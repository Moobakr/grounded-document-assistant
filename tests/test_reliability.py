import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np

from scripts.evaluate import load_cases, main as evaluate, score_answer, summarize
from src.cli import parser
from src.documents_loader import DocumentChunk
from src.generation import ABSTENTION, LocalGenerator, evidence_units, normalize_quote, render_excerpts
from src.vector_store import SearchMatch, VectorStore


class EvidenceTests(unittest.TestCase):
    sources = [("lecture.pdf, page 2", "Linear\nRegression and Logistic Regression"),
               ("lecture.pdf, page 3", "Data cleaning")]

    def test_whitespace_only_normalization_and_application_citations(self):
        result = {"excerpts": [{"quote": "Linear Regression", "source_id": 1}]}
        self.assertEqual(render_excerpts(result, self.sources), '"Linear Regression" [1]')

    def test_invented_or_paraphrased_quote_abstains(self):
        for quote in ("Random Forest", "linear regression", "", " ", "Linear Regression is best"):
            self.assertEqual(render_excerpts({"excerpts": [{"quote": quote, "source_id": 1}]}, self.sources),
                             ABSTENTION)

    def test_quote_must_occur_in_its_cited_source(self):
        result = {"excerpts": [{"quote": "Data cleaning", "source_id": 1}]}
        self.assertEqual(render_excerpts(result, self.sources), ABSTENTION)

    def test_one_invalid_quote_rejects_whole_answer(self):
        result = {"excerpts": [{"quote": "Linear Regression", "source_id": 1},
                               {"quote": "fabricated", "source_id": 2}]}
        self.assertEqual(render_excerpts(result, self.sources), ABSTENTION)

    def test_boolean_negative_and_out_of_range_references_rejected(self):
        for reference in (True, 0, -1, 3, "1"):
            with self.assertRaisesRegex(ValueError, "source reference"):
                render_excerpts({"excerpts": [{"quote": "Linear Regression", "source_id": reference}]}, self.sources)

    def test_malformed_shapes_rejected(self):
        for result in (None, [], {}, {"excerpts": "text"}, {"excerpts": [{}]},
                       {"excerpts": [], "answer": "unverified"}, {"excerpts": [None]}):
            with self.assertRaises(ValueError):
                render_excerpts(result, self.sources)

    def test_empty_excerpts_abstain(self):
        self.assertEqual(render_excerpts({"excerpts": []}, self.sources), ABSTENTION)

    def test_duplicate_quotes_removed(self):
        excerpt = {"quote": "Linear Regression", "source_id": 1}
        self.assertEqual(render_excerpts({"excerpts": [excerpt, excerpt]}, self.sources), '"Linear Regression" [1]')

    def test_quotes_with_brackets_or_quotes_are_preserved(self):
        quote = 'A "quoted" term [99]'
        self.assertEqual(render_excerpts({"excerpts": [{"quote": quote, "source_id": 1}]}, [("p1", quote)]),
                         json.dumps(quote) + " [1]")

    def test_default_evidence_mode_calls_local_server(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1]})
            if "directly_answers_question" in json.loads(request.content)["response_format"]["schema"]["properties"]:
                return httpx.Response(200, json={"choices": [{"message": {
                    "content": '{"directly_answers_question": true}'}, "finish_reason": "stop"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
                "excerpt_ids": [1],
            })}, "finish_reason": "stop"}]})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            answer = LocalGenerator(client=client).generate("Which method?", self.sources)
        self.assertEqual(answer, '"Linear Regression and Logistic Regression" [1]')
        payload = json.loads(requests[1].content)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertIn("excerpt_ids", payload["response_format"]["schema"]["properties"])

    def test_units_preserve_source_text_and_split_bullets(self):
        sources = [("page2", "Contents\n\u2751 Linear Regression\n\u2751 Clustering"),
                   ("page3", "A sentence. Another sentence.")]
        units = evidence_units(sources)
        self.assertEqual([unit["quote"] for unit in units],
                         ["Contents", "Linear Regression", "Clustering", "A sentence.", "Another sentence."])
        for unit in units:
            self.assertIn(unit["quote"], normalize_quote(sources[unit["source_id"] - 1][1]))

    def test_long_units_remain_bounded_and_source_backed(self):
        sources = [("page1", "word " * 400), ("page2", "x" * 1201)]
        for unit in evidence_units(sources):
            self.assertLessEqual(len(unit["quote"]), 600)
            self.assertIn(unit["quote"], sources[unit["source_id"] - 1][1])

    def test_invalid_excerpt_selections_rejected(self):
        for result in ({"excerpt_ids": [99]}, {"excerpt_ids": [True]}, {"excerpt_ids": "1"},
                       {"answer": "speculation"}, {"excerpt_ids": [1] * 17}):
            def handler(request):
                if request.url.path == "/tokenize":
                    return httpx.Response(200, json={"tokens": [1]})
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                              "finish_reason": "stop"}]})
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(ValueError, "excerpt"):
                    LocalGenerator(client=client).generate("Question", self.sources)

    def test_excerpt_number_maps_to_correct_passage_citation(self):
        sources = [("page2", "First sentence. Second sentence."), ("page3", "Third sentence.")]

        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1]})
            if "directly_answers_question" in json.loads(request.content)["response_format"]["schema"]["properties"]:
                return httpx.Response(200, json={"choices": [{"message": {
                    "content": '{"directly_answers_question": true}'}, "finish_reason": "stop"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"excerpt_ids": [3]}'},
                                                          "finish_reason": "stop"}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(LocalGenerator(client=client).generate("Question", sources), '"Third sentence." [2]')

    def test_relevance_rejection_becomes_abstention(self):
        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1]})
            review = "directly_answers_question" in json.loads(request.content)["response_format"]["schema"]["properties"]
            result = {"directly_answers_question": False} if review else {"excerpt_ids": [1]}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                          "finish_reason": "stop"}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(LocalGenerator(client=client).generate("Question", self.sources), ABSTENTION)

    def test_malformed_relevance_review_never_publishes_answer(self):
        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1]})
            review = "directly_answers_question" in json.loads(request.content)["response_format"]["schema"]["properties"]
            result = {"directly_answers_question": "yes"} if review else {"excerpt_ids": [1]}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                          "finish_reason": "stop"}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(ValueError, "relevance review"):
                LocalGenerator(client=client).generate("Question", self.sources)

    def test_truncated_and_invalid_json_rejected(self):
        for content, reason, message in (("{", "stop", "invalid JSON"), ("{}", "length", "output limit")):
            def handler(request):
                if request.url.path == "/tokenize":
                    return httpx.Response(200, json={"tokens": [1]})
                return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                              "finish_reason": reason}]})
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(ValueError, message):
                    LocalGenerator(client=client).generate("Question", self.sources)

    def test_no_evidence_needs_no_server(self):
        def handler(request):
            self.fail("Empty evidence should not contact the model server.")
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(LocalGenerator(client=client).generate("Question", []), ABSTENTION)

    def test_invalid_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "Answer mode"):
            LocalGenerator(answer_mode="unknown")

    def test_cli_defaults_to_evidence_and_allows_explicit_draft(self):
        self.assertEqual(parser().parse_args(["ask", "Question"]).answer_mode, "evidence")
        self.assertEqual(parser().parse_args(["ask", "Question", "--answer-mode", "draft"]).answer_mode, "draft")


class EvaluationTests(unittest.TestCase):
    case = {"id": "category", "question": "Which regression method?", "answerable": True,
            "relevant_pages": [{"source": "lecture.pdf", "page": 2}],
            "required_patterns": ["Linear Regression"], "forbidden_patterns": ["Clustering"]}
    matches = [SearchMatch(DocumentChunk("a", "lecture.pdf", 2, 0, "Linear Regression; Clustering"), 0.9)]

    def test_verified_quote_has_separate_checks(self):
        checks = score_answer(self.case, '"Linear Regression" [1]', self.matches, "evidence")
        for field in ("answer_check_pass", "citation_in_range", "citation_expected_page", "quote_grounding"):
            self.assertTrue(checks[field])

    def test_exact_quote_can_still_be_wrong_answer(self):
        checks = score_answer(self.case, '"Linear Regression; Clustering" [1]', self.matches, "evidence")
        self.assertTrue(checks["quote_grounding"])
        self.assertFalse(checks["answer_check_pass"])

    def test_invalid_citation_and_invented_quote_fail_grounding(self):
        for text in ('"Linear Regression" [99]', '"Invented" [1]', "unstructured answer"):
            self.assertFalse(score_answer(self.case, text, self.matches, "evidence")["quote_grounding"])

    def test_brackets_inside_quote_not_counted_as_citations(self):
        quote = 'Literal [99] and "text"'
        matches = [SearchMatch(DocumentChunk("a", "lecture.pdf", 2, 0, quote), 1)]
        checks = score_answer(self.case, json.dumps(quote) + " [1]", matches, "evidence")
        self.assertTrue(checks["quote_grounding"])
        self.assertTrue(checks["citation_in_range"])

    def test_draft_does_not_claim_verified_grounding(self):
        self.assertIsNone(score_answer(self.case, "Linear Regression [1]", self.matches, "draft")["quote_grounding"])

    def test_abstention_for_answerable_case_is_failure(self):
        checks = score_answer(self.case, ABSTENTION, self.matches, "evidence")
        self.assertFalse(checks["answer_check_pass"])
        self.assertTrue(checks["abstained"])
        self.assertIsNone(checks["citation_in_range"])

    def test_missing_information_requires_controlled_abstention(self):
        case = {"answerable": False}
        self.assertTrue(score_answer(case, ABSTENTION, self.matches, "evidence")["answer_check_pass"])
        self.assertFalse(score_answer(case, "I don't know [1]", self.matches, "draft")["answer_check_pass"])

    def test_failed_cases_stay_in_denominators(self):
        rows = [{"answerable": True, "generation_requested": True, "error": "server failed"},
                {"answerable": True, "generation_requested": True, "retrieval_hit": True,
                 "reciprocal_rank": 1, "checks": score_answer(self.case, '"Linear Regression" [1]',
                                                               self.matches, "evidence")}]
        summary = summarize(rows)
        self.assertEqual(summary["answer_checks"], {"passed": 1, "total": 2, "rate": 0.5})
        self.assertEqual(summary["retrieval_hit_at_k"]["total"], 2)
        self.assertEqual(summary["operational_errors"], 1)

    def test_dataset_validation_rejects_duplicate_ids_and_unlabelled_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            for cases in ([self.case, self.case], [{**self.case, "relevant_pages": []}],
                          [{**self.case, "required_patterns": []}], [{**self.case, "answerable": "yes"}]):
                path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_runner_reports_retrieval_and_generation_errors_without_losing_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, cases, output = root / "index.sqlite3", root / "cases.json", root / "report.json"
            store = VectorStore(index)
            try:
                store.add_document([DocumentChunk("a", "lecture.pdf", 2, 0, "Linear Regression")],
                                   np.array([[1, 0]]), "model-a")
            finally:
                store.close()
            cases.write_text(json.dumps({"cases": [self.case], "version": 1}), encoding="utf-8")
            embedder = Mock(identity="model-a")
            embedder.encode.return_value = np.array([[1, 0]])
            args = ["--index", str(index), "--cases", str(cases), "--output", str(output),
                    "--embedding-model", str(root / "model"), "--generation-model-dir", str(root / "gguf"),
                    "--server-key-file", str(root / "server.key")]
            with patch("scripts.evaluate.LocalEmbeddings", return_value=embedder), patch("builtins.print"), \
                    patch("scripts.evaluate.version", return_value="test-version"):
                self.assertEqual(evaluate(args), 0)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["summary"]["retrieval_hit_at_k"]["passed"], 1)
                self.assertIn("code_sha256", report)
                generator = Mock()
                generator.generate.side_effect = RuntimeError("local server unavailable")
                with patch("scripts.evaluate.LocalGenerator", return_value=generator):
                    self.assertEqual(evaluate(args + ["--generate"]), 1)
                generator.close.assert_called_once()
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(len(report["cases"]), 1)
                self.assertEqual(report["summary"]["operational_errors"], 1)
                self.assertEqual(report["summary"]["answer_checks"]["total"], 1)
            store = VectorStore(index)
            try:
                self.assertEqual(store.documents()[0]["chunks"], 1)
            finally:
                store.close()

    def test_runner_refuses_to_overwrite_index(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.sqlite3"
            index.write_bytes(b"sentinel")
            with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                evaluate(["--index", str(index), "--output", str(index)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(index.read_bytes(), b"sentinel")


if __name__ == "__main__":
    unittest.main()
