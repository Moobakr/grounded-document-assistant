"""Single-user local workspace with offline search and opt-in Groq tutoring."""

import argparse
import asyncio
from contextlib import closing, contextmanager
from dataclasses import asdict
import io
import os
from pathlib import Path, PureWindowsPath
import re
import secrets
import shutil
import tempfile
import threading
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
import httpx

from src.documents_loader import document_hash
from src.embeddings import LocalEmbeddings
from src.generation import ABSTENTION, LocalGenerator
from src.groq_generation import DEFAULT_MODEL, MODELS, GroqGenerator, validate_key
from src.rag import RAGPipeline
from src.vector_store import VectorStore


ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 25 * 1024 * 1024
PDF_LOCK = threading.Lock()


class UploadLimit:
    def __init__(self, app, limit=MAX_UPLOAD + 256 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            oversized = int(headers.get(b"content-length", b"0")) > self.limit
        except ValueError:
            return await JSONResponse({"detail": "Invalid content length."}, status_code=400)(scope, receive, send)
        if oversized:
            return await JSONResponse({"detail": "Upload exceeds the 25 MB limit."}, status_code=413)(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                raise HTTPException(413, "Upload exceeds the 25 MB limit.")
            return message

        await self.app(scope, bounded_receive, send)


class Turn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=3000)


class AISettings(BaseModel):
    api_key: SecretStr
    model: str = Field(default=DEFAULT_MODEL, pattern=r"^openai/gpt-oss-(120b|20b)$")


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)
    document_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    operation: str = Field(default="search", pattern=r"^(search|ask)$")
    provider: str = Field(default="groq", pattern=r"^(groq|local)$")
    task: str = Field(default="explain", pattern=r"^(answer|explain|summary|quiz)$")
    history: list[Turn] = Field(default_factory=list, max_length=2)


class Workspace:
    def __init__(self, root: Path, index: Path, model: Path, threads=2, embedder=None, generator_factory=None):
        self.root, self.index, self.model, self.threads = root.resolve(), index, model, threads
        self.uploads = self.root / "data/uploads"
        self.previews = self.root / "data/previews"
        self.embedder = embedder
        self.generator_factory = generator_factory or self.generator
        self.groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_model = DEFAULT_MODEL
        self.hash_cache = {}
        self.page_counts = {}

    @contextmanager
    def store(self):
        store = VectorStore(self.index)
        try:
            yield store
        finally:
            store.close()

    def embedding(self):
        if self.embedder is None:
            self.embedder = LocalEmbeddings(self.model, threads=self.threads)
        return self.embedder

    def generator(self):
        key = self.root / "data/local-server.key"
        return LocalGenerator(api_key=key.read_text().strip() if key.is_file() else None)

    def file(self, document):
        identity = document["id"]
        if not re.fullmatch(r"[a-f0-9]{64}", identity):
            return None
        candidates = [self.uploads / (identity + ".pdf")]
        source = document["source"]
        if Path(source).name == source and PureWindowsPath(source).name == source:
            candidates.append(self.root / "documents" / source)
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            digest = self.hash_cache.get(key)
            if digest is None:
                digest = document_hash(path)
                if len(self.hash_cache) >= 512:
                    self.hash_cache.clear()
                self.hash_cache[key] = digest
            if digest == identity:
                return path
        return None

    def documents(self):
        with self.store() as store:
            documents = store.documents()
        result = []
        for doc in documents:
            path = self.file(doc)
            pages = None
            if path is not None:
                if doc["id"] not in self.page_counts:
                    try:
                        self.page_counts[doc["id"]] = len(PdfReader(path).pages)
                    except (PdfReadError, ValueError, OSError):
                        self.page_counts[doc["id"]] = None
                pages = self.page_counts[doc["id"]]
            result.append({**doc, "has_file": path is not None, "pages": pages})
        return result

    def document(self, identity):
        for doc in self.documents():
            if doc["id"] == identity:
                return doc
        raise HTTPException(404, "Document is no longer indexed.")

    def ingest(self, path):
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")
        if not 0 < len(reader.pages) <= 300:
            raise ValueError("Upload a PDF with between 1 and 300 pages.")
        identity = document_hash(path)
        self.uploads.mkdir(parents=True, exist_ok=True)
        archive = self.uploads / (identity + ".pdf")
        created = False
        try:
            if not archive.exists():
                with archive.open("xb") as destination, path.open("rb") as source:
                    created = True
                    shutil.copyfileobj(source, destination)
            elif archive.is_symlink() or document_hash(archive) != identity:
                raise ValueError("Stored PDF failed its content-hash check.")
            with self.store() as store:
                added = RAGPipeline(self.embedding(), store).ingest_document(path)
            return {"status": "indexed" if added else "duplicate", "document": self.document(identity)}
        except Exception:
            if created:
                with self.store() as store:
                    published = store.contains(identity)
                if not published:
                    archive.unlink(missing_ok=True)
            raise

    def delete(self, identity):
        self.document(identity)
        with self.store() as store:
            if not store.delete_document(identity):
                raise HTTPException(404, "Document is no longer indexed.")
        archive = self.uploads / (identity + ".pdf")
        archive.unlink(missing_ok=True)
        for preview in self.previews.glob(identity + "-*.png"):
            preview.unlink(missing_ok=True)
        return {"deleted": identity}

    def query(self, query):
        generator = None
        started = perf_counter()
        try:
            with self.store() as store:
                if query.document_id is not None and not store.contains(query.document_id):
                    raise HTTPException(404, "Selected document is no longer indexed.")
                rag = RAGPipeline(self.embedding(), store)
                retrieval_question = query.question
                if query.operation == "ask" and query.provider == "groq" and query.history:
                    # Carry the topic into short follow-ups without embedding generated answers as evidence.
                    retrieval_question = "\n".join(turn.question[:500] for turn in query.history) + "\n" + query.question
                matches = rag.retrieve(retrieval_question, query.top_k, query.document_id)
                answer, generation_error = None, None
                if query.operation == "ask":
                    try:
                        generator = (GroqGenerator(self.groq_key, self.groq_model) if query.provider == "groq"
                                     else self.generator_factory())
                        sources = [(f"{match.chunk.source}, page {match.chunk.page}", match.chunk.text) for match in matches]
                        if query.provider == "groq":
                            answer = generator.generate(query.question, sources, task=query.task,
                                                        history=[turn.model_dump() for turn in query.history])
                        else:
                            answer = generator.generate(query.question, sources)
                    except (RuntimeError, ValueError, httpx.HTTPError) as error:
                        generation_error = str(error)
                return {"question": query.question.strip(), "operation": query.operation,
                        "provider": query.provider if query.operation == "ask" else "local",
                        "task": query.task,
                        "answer": answer, "abstained": answer == ABSTENTION,
                        "generation_error": generation_error, "sources": [asdict(match) for match in matches],
                        "seconds": round(perf_counter() - started, 3)}
        finally:
            if generator is not None:
                generator.close()

    def preview(self, identity, number):
        path = self.file(self.document(identity))
        if path is None:
            raise HTTPException(404, "Original PDF is unavailable. Upload it to restore the preview.")
        target = self.previews / f"{identity}-{number}.png"
        with PDF_LOCK:
            if not target.is_file():
                import pypdfium2 as pdfium

                with closing(pdfium.PdfDocument(path)) as pdf:
                    if not 1 <= number <= len(pdf):
                        raise HTTPException(404, "PDF page does not exist.")
                    with closing(pdf[number - 1]) as page:
                        width, height = page.get_size()
                        if width <= 0 or height <= 0:
                            raise ValueError("PDF page dimensions are invalid.")
                        scale = min(640 / width, 900 / height, 1.5)
                        with closing(page.render(scale=scale)) as bitmap:
                            image = bitmap.to_pil()
                            buffer = io.BytesIO()
                            image.save(buffer, format="PNG")
                            image.close()
                self.previews.mkdir(parents=True, exist_ok=True)
                target.write_bytes(buffer.getvalue())
        return target


def create_app(root=ROOT, index=None, model=None, threads=2, embedder=None, generator_factory=None):
    root = Path(root)
    workspace = Workspace(root, Path(index or root / "data/index.sqlite3"),
                          Path(model or root / "models/embeddings"), threads, embedder, generator_factory)
    app = FastAPI(title="Document Assistant", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.workspace = workspace
    app.state.work_lock = asyncio.Lock()
    token = secrets.token_urlsafe(32)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    app.add_middleware(UploadLimit)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site requests are not permitted."}, status_code=403)
            if request.method not in ("GET", "HEAD"):
                origin = request.headers.get("origin")
                expected = f"{request.url.scheme}://{request.headers.get('host')}"
                if origin and origin != expected:
                    return JSONResponse({"detail": "Origin is not permitted."}, status_code=403)
                if not secrets.compare_digest(request.headers.get("x-workspace-token", ""), token):
                    return JSONResponse({"detail": "Reload the workspace and retry."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        if request.url.path == "/api/ai":
            return JSONResponse({"detail": "Enter a valid key and supported model."}, status_code=422)
        return await request_validation_exception_handler(request, error)

    @app.exception_handler(PdfReadError)
    async def pdf_error(request, error):
        return JSONResponse({"detail": "Invalid PDF. Select a readable PDF with extractable text."}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=503)

    @app.exception_handler(OSError)
    async def disk_error(request, error):
        return JSONResponse({"detail": "Local file access failed. Check available disk space and file permissions."}, status_code=500)

    @app.get("/api/status")
    async def status():
        return {"token": token, "embedding_ready": embedder is not None or (workspace.model / "modules.json").is_file(),
                "busy": app.state.work_lock.locked(), "threads": threads}

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "embedding_ready": embedder is not None or (workspace.model / "modules.json").is_file()}

    @app.get("/api/model")
    async def model_status():
        key_path = root / "data/local-server.key"
        headers = {"Authorization": "Bearer " + key_path.read_text().strip()} if key_path.is_file() else {}
        try:
            async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as client:
                result = await client.get("http://127.0.0.1:8080/health", headers=headers)
            return {"online": result.status_code == 200}
        except httpx.HTTPError:
            return {"online": False}

    @app.get("/api/ai")
    async def ai_status():
        return {"configured": bool(workspace.groq_key), "model": workspace.groq_model,
                "models": list(MODELS), "provider": "groq"}

    @app.put("/api/ai")
    async def configure_ai(body: AISettings):
        check_busy()
        key = validate_key(body.api_key.get_secret_value())
        workspace.groq_key, workspace.groq_model = key, body.model
        return {"configured": True, "model": workspace.groq_model}

    @app.delete("/api/ai")
    async def disconnect_ai():
        check_busy()
        workspace.groq_key = ""
        return {"configured": False}

    @app.get("/api/documents")
    async def documents():
        return await run_in_threadpool(workspace.documents)

    def check_busy():
        if app.state.work_lock.locked():
            raise HTTPException(409, "Workspace is busy. Wait for the current operation to finish.")

    @app.post("/api/documents")
    async def upload(file: UploadFile):
        check_busy()
        async with app.state.work_lock:
            name = file.filename or ""
            if (PureWindowsPath(name).name != name or Path(name).name != name or len(name) > 120
                    or Path(name).suffix.lower() != ".pdf" or Path(name).is_reserved()
                    or re.search(r'[<>:"|?*\x00-\x1f]', name)):
                raise HTTPException(400, "Select a PDF with a valid filename, without directory paths.")
            workspace.uploads.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(dir=workspace.uploads) as temporary:
                    path = Path(temporary) / name
                    size = 0
                    with path.open("wb") as destination:
                        while block := await file.read(1024 * 1024):
                            size += len(block)
                            if size > MAX_UPLOAD:
                                raise HTTPException(413, "Upload exceeds the 25 MB limit.")
                            destination.write(block)
                    if size == 0:
                        raise HTTPException(400, "The selected file is empty.")
                    return await run_in_threadpool(workspace.ingest, path)
            finally:
                await file.close()

    @app.delete("/api/documents/{identity}")
    async def delete(identity: str):
        check_busy()
        async with app.state.work_lock:
            return await run_in_threadpool(workspace.delete, identity)

    @app.post("/api/query")
    async def query(body: Query):
        check_busy()
        async with app.state.work_lock:
            return await run_in_threadpool(workspace.query, body)

    @app.get("/api/documents/{identity}/file")
    async def pdf(identity: str):
        document = await run_in_threadpool(workspace.document, identity)
        path = await run_in_threadpool(workspace.file, document)
        if path is None:
            raise HTTPException(404, "Original PDF is unavailable. Upload it to restore access.")
        return FileResponse(path, media_type="application/pdf", filename=document["source"], content_disposition_type="inline")

    @app.get("/api/documents/{identity}/preview/{number}")
    async def preview(identity: str, number: int):
        if not 1 <= number <= 300:
            raise HTTPException(404, "PDF page does not exist.")
        return FileResponse(await run_in_threadpool(workspace.preview, identity, number), media_type="image/png")

    @app.get("/")
    async def home():
        return FileResponse(ROOT / "src/static/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "src/static"), name="static")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="Use 0.0.0.0 only inside a container with host-loopback port publishing.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.sqlite3")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    import uvicorn

    uvicorn.run(create_app(index=args.index), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
