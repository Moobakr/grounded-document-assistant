import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.documents_loader import document_hash
from src.groq_generation import GroqGenerator
from src.web import create_app, main


def sample_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(width=500, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 18 Tf 30 160 Td (Linear regression relates variables.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FakeEmbeddings:
    identity = "web-test-model"

    def encode(self, texts):
        return np.array([[1, 0] for _ in texts], dtype=np.float32)

    def split_text(self, text, size, overlap):
        return [text]


class WebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.embedder = FakeEmbeddings()
        self.generator = Mock()
        self.generator.generate.return_value = '"Linear regression relates variables." [1]'
        self.app = create_app(root=self.root, embedder=self.embedder, generator_factory=lambda: self.generator)
        self.client = TestClient(self.app, base_url="http://127.0.0.1")
        self.headers = {"X-Workspace-Token": self.client.get("/api/status").json()["token"]}
        self.pdf = sample_pdf()

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def upload(self, filename="sample.pdf", data=None):
        return self.client.post("/api/documents", files={"file": (filename, self.pdf if data is None else data, "application/pdf")}, headers=self.headers)

    def test_library_does_not_load_embeddings_and_does_not_expose_model_key(self):
        key = self.root / "data/local-server.key"
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text("secret-model-key")
        self.assertEqual(self.client.get("/api/documents").json(), [])
        self.assertNotIn("secret-model-key", self.client.get("/api/status").text)

    def test_upload_persists_document_and_original_for_preview(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        doc = response.json()["document"]
        self.assertEqual(doc["source"], "sample.pdf")
        self.assertTrue(doc["has_file"])
        self.assertEqual(doc["pages"], 1)
        archive = self.root / "data/uploads" / (doc["id"] + ".pdf")
        self.assertEqual(document_hash(archive), doc["id"])
        self.assertEqual(self.client.get(f"/api/documents/{doc['id']}/file").content, self.pdf)

    def test_preview_is_real_png_and_missing_page_is_404(self):
        doc = self.upload().json()["document"]
        response = self.client.get(f"/api/documents/{doc['id']}/preview/1")
        self.assertEqual(response.status_code, 200, response.text[:100])
        self.assertTrue(response.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get(f"/api/documents/{doc['id']}/preview/2").status_code, 404)

    def test_duplicate_upload_skips_indexing(self):
        self.upload()
        response = self.upload("renamed.pdf")
        self.assertEqual(response.json()["status"], "duplicate")
        docs = self.client.get("/api/documents").json()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["source"], "sample.pdf")

    def test_mutations_require_session_header_and_matching_origin(self):
        self.assertEqual(self.client.post("/api/query", json={"question": "test"}).status_code, 403)
        self.assertEqual(self.client.post("/api/query", json={"question": "test"}, headers={**self.headers, "Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.get("/api/documents", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
        self.assertEqual(self.client.get("/api/status", headers={"Host": "evil.example"}).status_code, 400)

    def test_invalid_uploads_never_publish_document(self):
        for filename, content in (("sample.txt", self.pdf), ("../sample.pdf", self.pdf),
                                  ("sample.pdf", b""), ("sample.pdf", b"not a pdf")):
            response = self.upload(filename, content)
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.client.get("/api/documents").json(), [])

    def test_upload_limits_reject_before_indexing(self):
        response = self.client.post("/api/documents", headers={**self.headers, "Content-Length": str(27 * 1024 * 1024)})
        self.assertEqual(response.status_code, 413)
        with patch("src.web.MAX_UPLOAD", 16):
            self.assertEqual(self.upload(data=b"x" * 17).status_code, 413)
        self.assertEqual(self.client.get("/api/documents").json(), [])

    def test_embedding_failure_rolls_back_managed_file(self):
        with patch.object(self.embedder, "encode", side_effect=RuntimeError("Embedding failure")):
            self.assertEqual(self.upload().status_code, 503)
        self.assertEqual(list((self.root / "data/uploads").glob("*.pdf")), [])
        self.assertEqual(self.client.get("/api/documents").json(), [])

    def test_search_returns_page_references_without_generation(self):
        self.upload()
        response = self.client.post("/api/query", json={"question": "regression"}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["sources"][0]["chunk"]["page"], 1)
        self.assertIsNone(response.json()["answer"])
        self.generator.generate.assert_not_called()

    def test_answer_calls_generator_and_closes_client(self):
        self.upload()
        response = self.client.post("/api/query", json={"question": "regression", "operation": "ask", "provider": "local"}, headers=self.headers)
        self.assertEqual(response.json()["answer"], self.generator.generate.return_value)
        self.generator.close.assert_called_once()

    def test_failed_generation_retains_retrieved_passages(self):
        self.upload()
        self.generator.generate.side_effect = RuntimeError("Local server unavailable")
        response = self.client.post("/api/query", json={"question": "regression", "operation": "ask", "provider": "local"}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json()["generation_error"])
        self.assertEqual(len(response.json()["sources"]), 1)

    def test_query_input_and_deleted_scope_validation(self):
        for body in ({"question": ""}, {"question": "hi", "top_k": 100}, {"question": "hi", "operation": "remote"}):
            self.assertEqual(self.client.post("/api/query", json=body, headers=self.headers).status_code, 422)
        response = self.client.post("/api/query", json={"question": "hi", "document_id": "a" * 64}, headers=self.headers)
        self.assertEqual(response.status_code, 404)

    def test_busy_operation_is_rejected_not_queued(self):
        lock = Mock()
        lock.locked.return_value = True
        self.app.state.work_lock = lock
        self.assertEqual(self.client.post("/api/query", json={"question": "hi"}, headers=self.headers).status_code, 409)

    def test_delete_removes_index_and_managed_copy_not_original(self):
        original = self.root / "documents/sample.pdf"
        original.parent.mkdir()
        original.write_bytes(self.pdf)
        doc = self.upload().json()["document"]
        response = self.client.delete(f"/api/documents/{doc['id']}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/documents").json(), [])
        self.assertTrue(original.is_file())
        self.assertFalse((self.root / "data/uploads" / (doc["id"] + ".pdf")).exists())

    def test_changed_original_cannot_be_served_as_indexed_pdf(self):
        doc = self.upload().json()["document"]
        archive = self.root / "data/uploads" / (doc["id"] + ".pdf")
        archive.write_bytes(b"changed content")
        self.assertEqual(self.client.get(f"/api/documents/{doc['id']}/file").status_code, 404)

    def test_private_paths_and_env_are_not_static_assets(self):
        self.assertEqual(self.client.get("/static/../.env").status_code, 404)
        self.assertEqual(self.client.get("/data/index.sqlite3").status_code, 404)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_groq_key_is_memory_only_not_in_status_or_validation_responses(self):
        key = "private-groq-key"
        self.assertEqual(self.client.put("/api/ai", json={"api_key": key}).status_code, 403)
        result = self.client.put("/api/ai", json={"api_key": key}, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(self.client.get("/api/ai").json()["configured"])
        self.assertEqual(self.app.state.workspace.groq_key, key)
        for response in (result, self.client.get("/api/ai"), self.client.get("/api/status"),
                         self.client.put("/api/ai", json={"api_key": {"secret": key}}, headers=self.headers)):
            self.assertNotIn(key, response.text)
        self.assertFalse(any(key in path.read_text() for path in self.root.rglob("*") if path.is_file()))
        self.assertEqual(self.client.delete("/api/ai", headers=self.headers).status_code, 200)
        self.assertFalse(self.client.get("/api/ai").json()["configured"])

    def test_default_assistant_uses_real_groq_adapter_and_followup_retrieval_context(self):
        self.upload()
        self.client.put("/api/ai", json={"api_key": "test-key"}, headers=self.headers)
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "answer": "Regression predicts a number. [1]", "source_ids": [1], "abstained": False})}}]})

        history = [{"question": "What is regression?", "answer": "It relates variables. [1]"}]
        with patch("src.web.GroqGenerator", side_effect=lambda key, model: GroqGenerator(key, model, httpx.MockTransport(respond))):
            with patch.object(self.embedder, "encode", wraps=self.embedder.encode) as encode:
                result = self.client.post("/api/query", json={"question": "Give me an example", "operation": "ask",
                                                             "task": "explain", "history": history}, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["provider"], "groq")
        self.assertEqual(result.json()["answer"], "Regression predicts a number. [1]")
        self.assertIn("What is regression?", encode.call_args.args[0][0])
        self.assertEqual(json.loads(requests[0]["messages"][1]["content"])["conversation"], history)
        self.generator.generate.assert_not_called()

    def test_groq_errors_keep_sources_and_search_never_creates_groq_client(self):
        self.upload()
        with patch("src.web.GroqGenerator", side_effect=RuntimeError("Groq rate limit reached")) as factory:
            result = self.client.post("/api/query", json={"question": "regression", "operation": "ask"}, headers=self.headers)
            self.assertEqual(len(result.json()["sources"]), 1)
            self.assertIsNone(result.json()["answer"])
            self.assertIn("limit", result.json()["generation_error"])
            factory.reset_mock()
            self.client.post("/api/query", json={"question": "regression"}, headers=self.headers)
            factory.assert_not_called()

    def test_history_provider_and_task_limits(self):
        for body in ({"provider": "other"}, {"task": "other"},
                     {"history": [{"question": "q", "answer": "a"}] * 3},
                     {"history": [{"question": "q", "answer": "x" * 3001}]}):
            result = self.client.post("/api/query", json={"question": "regression", **body}, headers=self.headers)
            self.assertEqual(result.status_code, 422)

    def test_disconnect_stays_disconnected_with_environment_key(self):
        self.upload()
        with patch.dict(os.environ, {"GROQ_API_KEY": "environment-secret"}):
            self.client.delete("/api/ai", headers=self.headers)
            result = self.client.post("/api/query", json={"question": "regression", "operation": "ask"}, headers=self.headers)
            self.assertIn("not configured", result.json()["generation_error"])

    def test_configuring_key_during_query_is_rejected(self):
        lock = Mock()
        lock.locked.return_value = True
        self.app.state.work_lock = lock
        self.assertEqual(self.client.put("/api/ai", json={"api_key": "test-key"}, headers=self.headers).status_code, 409)

    def test_health_does_not_expose_tokens_or_keys_or_load_model(self):
        self.app.state.workspace.groq_key = "private-health-key"
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "embedding_ready": True})
        self.assertNotIn("private-health-key", response.text)
        self.assertNotIn("token", response.text)
        with tempfile.TemporaryDirectory() as root:
            with TestClient(create_app(root=root), base_url="http://127.0.0.1") as client:
                self.assertEqual(client.get("/api/health").json(), {"status": "ok", "embedding_ready": False})

    def test_server_defaults_to_loopback_and_container_binding_is_explicit(self):
        with patch("src.web.create_app", return_value=self.app), patch("uvicorn.run") as run:
            main([])
            self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
            main(["--host", "0.0.0.0", "--port", "8767"])
            self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
            self.assertEqual(run.call_args.kwargs["port"], 8767)


if __name__ == "__main__":
    unittest.main()
