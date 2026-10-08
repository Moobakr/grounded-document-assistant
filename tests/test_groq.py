import json
import os
import unittest
from unittest.mock import patch

import httpx

from src.generation import ABSTENTION
from src.groq_generation import DEFAULT_MODEL, GroqGenerator


class GroqTests(unittest.TestCase):
    def generator(self, result=None, status=200, finish="stop"):
        self.requests = []

        def respond(request):
            self.requests.append(request)
            return httpx.Response(status, json={"choices": [{"finish_reason": finish, "message": {
                "content": json.dumps(result or {"answer": "Regression predicts a number. [1]",
                                              "source_ids": [1], "abstained": False})}}]})

        generator = GroqGenerator("private-test-key", transport=httpx.MockTransport(respond))
        self.addCleanup(generator.close)
        return generator

    def test_retrieved_evidence_task_and_history_go_to_fixed_https_endpoint(self):
        generator = self.generator()
        history = [{"question": "What is regression?", "answer": "Predict a number. [1]"}]
        answer = generator.generate("Give an example", [("course.pdf, page 3", "Regression relates variables.")],
                                    task="explain", history=history)
        self.assertIn("predicts", answer)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer private-test-key")
        body = json.loads(request.content)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        content = json.loads(body["messages"][1]["content"])
        self.assertEqual(content["conversation"], history)
        self.assertEqual(content["passages"][0]["source_id"], 1)
        self.assertIn("untrusted data", body["messages"][0]["content"])
        self.assertIn("Teach the concept", body["messages"][0]["content"])

    def test_quiz_and_summary_have_distinct_prompts(self):
        generator = self.generator()
        for task, wording in (("quiz", "answer key"), ("summary", "Summarize")):
            generator.generate("regression", [("source", "text")], task=task)
            self.assertIn(wording, json.loads(self.requests[-1].content)["messages"][0]["content"])

    def test_no_evidence_means_no_cloud_request(self):
        self.assertEqual(self.generator().generate("hello", []), ABSTENTION)
        self.assertEqual(self.requests, [])

    def test_missing_key_and_disconnect_do_not_use_environment_fallback(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "private-test-key"}):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                GroqGenerator("")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                GroqGenerator()

    def test_invalid_key_and_model(self):
        for key in ("key\nheader", "key with spaces", "x" * 513):
            with self.assertRaises(ValueError):
                GroqGenerator(key)
        with self.assertRaises(ValueError):
            GroqGenerator("valid-key", model="arbitrary-service/model")

    def test_error_messages_do_not_include_provider_body_or_key(self):
        for status in (301, 400, 401, 403, 404, 429, 500):
            with self.subTest(status=status):
                generator = self.generator(status=status)
                with self.assertRaises(RuntimeError) as caught:
                    generator.generate("private question", [("private filename", "private passage")])
                for private in ("private-test-key", "private question", "private passage"):
                    self.assertNotIn(private, str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_timeout_is_controlled_without_retries(self):
        def timeout(request):
            raise httpx.ReadTimeout("private-test-key", request=request)

        generator = GroqGenerator("private-test-key", transport=httpx.MockTransport(timeout))
        self.addCleanup(generator.close)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            generator.generate("question", [("source", "text")])

    def test_invalid_outputs_and_citations_fail_closed(self):
        cases = [
            {"answer": "Unsupported [2]", "source_ids": [2], "abstained": False},
            {"answer": "Unsupported [2]", "source_ids": [1], "abstained": False},
            {"answer": "No citations", "source_ids": [1], "abstained": False},
            {"answer": "Claim [1]", "source_ids": [True], "abstained": False},
            {"answer": "Claim [1]", "source_ids": [1, 1], "abstained": False},
            {"answer": "Claim [1]", "source_ids": [1], "abstained": "false"},
            {"answer": "Refusal [1]", "source_ids": [1], "abstained": True},
            {"answer": "", "source_ids": [], "abstained": True},
            {"answer": "Claim [1]", "source_ids": [1], "abstained": False, "extra": "data"},
        ]
        for result in cases:
            with self.subTest(result=result):
                with self.assertRaisesRegex(RuntimeError, "invalid citations"):
                    self.generator(result).generate("question", [("source", "text")])

    def test_abstention_and_truncation(self):
        generator = self.generator({"answer": "Not covered.", "source_ids": [], "abstained": True})
        self.assertEqual(generator.generate("question", [("source", "text")]), ABSTENTION)
        with self.assertRaises(RuntimeError):
            self.generator(finish="length").generate("question", [("source", "text")])

    def test_invalid_json_is_controlled(self):
        generator = GroqGenerator("test-key", transport=httpx.MockTransport(lambda request: httpx.Response(200, text="broken")))
        self.addCleanup(generator.close)
        with self.assertRaises(RuntimeError):
            generator.generate("question", [("source", "text")])


if __name__ == "__main__":
    unittest.main()
