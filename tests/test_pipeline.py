import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np

from src.cli import main
from src.documents_loader import DocumentChunk, chunk_text, load_chunks
from src.generation import LocalGenerator
from src.rag import RAGPipeline
from src.vector_store import VectorStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "index.sqlite3"
        self.store = VectorStore(self.path)
        self.chunks = [DocumentChunk("a", "lecture.pdf", 2, 0, "Regression methods"),
                       DocumentChunk("a", "lecture.pdf", 3, 1, "Data cleaning")]
        self.vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_persistence_and_page_metadata(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        self.store.close()
        self.store = VectorStore(self.path)
        matches = self.store.search(np.array([1, 0]), "model-a")
        self.assertEqual(matches[0].chunk.page, 2)
        self.assertEqual(matches[0].chunk.source, "lecture.pdf")
        self.assertAlmostEqual(matches[0].score, 1)

    def test_duplicate_does_not_add_chunks(self):
        self.assertTrue(self.store.add_document(self.chunks, self.vectors, "model-a"))
        self.assertFalse(self.store.add_document(self.chunks, self.vectors, "model-a"))
        self.assertEqual(self.store.documents()[0]["chunks"], 2)

    def test_multiple_documents_and_filtering(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        other = [DocumentChunk("b", "other.pdf", 1, 0, "Regression details")]
        self.store.add_document(other, np.array([[1, 0]]), "model-a")
        matches = self.store.search(np.array([1, 0]), "model-a", document_id="b")
        self.assertEqual([m.chunk.source for m in matches], ["other.pdf"])
        self.assertEqual(len(self.store.documents()), 2)

    def test_model_mismatch_rejected_even_with_same_dimensions(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.store.search(np.array([1, 0]), "model-b")

    def test_invalid_embeddings_never_publish_document(self):
        for vectors in (np.array([[1, 0]]), np.zeros((2, 2)),
                        np.array([[np.nan, 0], [1, 0]])):
            with self.assertRaises(ValueError):
                self.store.add_document(self.chunks, vectors, "model-a")
            self.assertEqual(self.store.documents(), [])

    def test_sql_failure_rolls_back_document_and_metadata(self):
        duplicate_chunk_indices = [self.chunks[0], self.chunks[0]]
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_document(duplicate_chunk_indices, self.vectors, "model-a")
        self.assertFalse(self.store.contains("a"))
        self.assertEqual(self.store.documents(), [])
        self.assertEqual(self.store._metadata(), {})

    def test_delete_cascades_and_empty_store_accepts_new_model(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        self.assertTrue(self.store.delete_document("a"))
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
        self.assertTrue(self.store.add_document(self.chunks, self.vectors, "model-b"))

    def test_invalid_top_k_and_query_dimensions(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        with self.assertRaises(ValueError):
            self.store.search(np.array([1, 0]), "model-a", top_k=0)
        with self.assertRaises(ValueError):
            self.store.search(np.array([1, 0, 0]), "model-a")

    def test_streaming_ranking_matches_full_sort(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(100, 8)).astype(np.float32)
        chunks = [DocumentChunk("c", "many.pdf", 1, i, f"Passage {i}") for i in range(100)]
        self.store.add_document(chunks, vectors, "model-a")
        query = rng.normal(size=8).astype(np.float32)
        expected = np.argsort((vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
                              @ (query / np.linalg.norm(query)))[::-1][:7]
        found = self.store.search(query, "model-a", top_k=7)
        self.assertEqual([m.chunk.chunk_index for m in found], expected.tolist())

    def test_pipeline_forwards_numbered_source_metadata(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        embedder = Mock(identity="model-a")
        embedder.encode.return_value = np.array([[1, 0]])
        generator = Mock()
        generator.generate.return_value = "Regression methods [1]."
        pipeline = RAGPipeline(embedder, self.store, generator)
        answer = pipeline.ask("Regression?", top_k=1)
        self.assertEqual(answer.sources[0].chunk.page, 2)
        generator.generate.assert_called_once_with("Regression?", [("lecture.pdf, page 2", "Regression methods")])

    def test_pipeline_duplicate_skips_embedding(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        embedder = Mock(identity="model-a")
        with patch("src.rag.document_hash", return_value="a"):
            self.assertFalse(RAGPipeline(embedder, self.store).ingest_document("duplicate.pdf"))
        embedder.encode.assert_not_called()

    def test_failed_embedding_leaves_old_documents_usable(self):
        self.store.add_document(self.chunks, self.vectors, "model-a")
        embedder = Mock(identity="model-a")
        embedder.encode.side_effect = RuntimeError("embedding failed")
        incoming = [DocumentChunk("b", "broken.pdf", 1, 0, "Some text")]
        with patch("src.rag.document_hash", return_value="b"), patch("src.rag.load_chunks", return_value=incoming):
            with self.assertRaises(RuntimeError):
                RAGPipeline(embedder, self.store).ingest_document("broken.pdf")
        self.assertFalse(self.store.contains("b"))
        self.assertEqual(len(self.store.documents()), 1)


class DocumentTests(unittest.TestCase):
    def test_invalid_overlap_cannot_hang_chunker(self):
        for size, overlap in ((0, 0), (5, 5), (5, -1), (5, 6)):
            with self.assertRaises(ValueError):
                chunk_text("text", size, overlap)

    def test_page_numbers_include_blank_pages(self):
        pages = [Mock(), Mock(), Mock()]
        for page, text in zip(pages, ("First page", None, "Third page")):
            page.extract_text.return_value = text
        reader = Mock(is_encrypted=False, pages=pages)
        with patch("src.documents_loader.document_hash", return_value="hash"), \
                patch("src.documents_loader.PdfReader", return_value=reader):
            chunks = load_chunks("lecture.pdf", lambda text: [text] if text else [])
        self.assertEqual([chunk.page for chunk in chunks], [1, 3])

    def test_no_extractable_text_is_rejected(self):
        page = Mock()
        page.extract_text.return_value = None
        reader = Mock(is_encrypted=False, pages=[page])
        with patch("src.documents_loader.document_hash", return_value="hash"), \
                patch("src.documents_loader.PdfReader", return_value=reader):
            with self.assertRaisesRegex(ValueError, "OCR"):
                load_chunks("scan.pdf", lambda text: [])


class GenerationTests(unittest.TestCase):
    def test_remote_urls_rejected(self):
        for url in ("https://api.groq.com", "http://example.com", "http://127.0.0.1.evil.com",
                    "http://user:password@localhost:8080", "http://localhost:8080/v1"):
            with self.assertRaises(ValueError):
                LocalGenerator(url)

    def test_sources_reach_local_server_and_answer_is_returned(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
                "answer": "Linear regression.", "source_ids": [1],
            })},
                                                        "finish_reason": "stop"}]})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            generator = LocalGenerator(client=client, answer_mode="draft")
            answer = generator.generate("Which method?", [("lecture.pdf, page 4", "Linear regression")])
        self.assertEqual(answer, "Linear regression. [1]")
        payload = json.loads(requests[-1].content)
        self.assertIn("[1] lecture.pdf, page 4", payload["messages"][0]["content"])
        self.assertEqual(payload["max_tokens"], 384)
        self.assertEqual(payload["response_format"]["schema"]["properties"]["source_ids"]["items"]["enum"], [1])

    def test_invalid_source_numbers_are_rejected(self):
        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1]})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
                "answer": "Unsupported answer", "source_ids": [99],
            })}, "finish_reason": "stop"}]})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(ValueError, "invalid source"):
                LocalGenerator(client=client, answer_mode="draft").generate("Question", [("page 1", "text")])

    def test_no_sources_returns_controlled_abstention(self):
        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1]})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
                "answer": "Unverified speculation", "source_ids": [],
            })}, "finish_reason": "stop"}]})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            answer = LocalGenerator(client=client, answer_mode="draft").generate("Question", [("page 1", "text")])
        self.assertIn("do not provide enough information", answer)
        self.assertNotIn("speculation", answer)

    def test_context_overflow_is_rejected_before_generation(self):
        paths = []

        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(200, json={"tokens": list(range(4000))})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(ValueError, "budget"):
                LocalGenerator(client=client).generate("Question", [("page 1", "text")])
        self.assertEqual(paths, ["/tokenize"])

    def test_unavailable_server_has_actionable_error(self):
        def handler(request):
            raise httpx.ConnectError("not running", request=request)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "Start llama-server"):
                LocalGenerator(client=client).generate("Question", [("page 1", "text")])

    def test_cli_listing_does_not_require_model_files(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            self.assertEqual(main(["--index", str(Path(directory) / "db.sqlite3"), "list"]), 0)


if __name__ == "__main__":
    unittest.main()
