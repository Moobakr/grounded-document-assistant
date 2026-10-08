import argparse
from dataclasses import asdict
import json
from pathlib import Path
import secrets
import subprocess
import sys

import httpx
from pypdf.errors import PdfReadError

from src.embeddings import LocalEmbeddings
from src.generation import LocalGenerator
from src.groq_generation import DEFAULT_MODEL, GroqGenerator
from src.rag import RAGPipeline
from src.vector_store import VectorStore


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Offline PDF search and optional local answers.")
    root.add_argument("--index", type=Path, default=Path("data/index.sqlite3"))
    root.add_argument("--embedding-model", type=Path, default=Path("models/embeddings"))
    root.add_argument("--threads", type=positive, default=2)
    root.add_argument("--server-key-file", type=Path, default=Path("data/local-server.key"))
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Explicit one-time online embedding model download")
    prepare.add_argument("--revision", default="main")
    generation = commands.add_parser("prepare-generation", help="Download free Windows CPU runtime and 1.12 GB model")
    generation.add_argument("--model-dir", type=Path, default=Path("models/generation"))
    generation.add_argument("--runtime-dir", type=Path, default=Path("tools/llama"))
    ingest = commands.add_parser("ingest", help="Index PDFs; unchanged duplicates are skipped")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.add_argument("--chunk-size", type=positive, default=180, help="embedding tokenizer tokens")
    ingest.add_argument("--overlap", type=int, default=30)
    for command in ("search", "ask"):
        query = commands.add_parser(command)
        query.add_argument("question")
        query.add_argument("--top-k", type=positive, default=3)
        query.add_argument("--document", help="full document ID from list")
        query.add_argument("--json", action="store_true")
        if command == "ask":
            query.add_argument("--answer-mode", choices=("evidence", "draft"), default="evidence",
                               help="verified excerpts (default), or experimental free-form draft")
            query.add_argument("--server", default="http://127.0.0.1:8080")
            query.add_argument("--model", default="local")
            query.add_argument("--provider", choices=("local", "groq"), default="local",
                               help="groq sends the question and retrieved passages to GroqCloud")
            query.add_argument("--groq-model", default=DEFAULT_MODEL)
            query.add_argument("--task", choices=("answer", "explain", "summary", "quiz"), default="explain")
    commands.add_parser("list", help="List persisted documents without loading a model")
    delete = commands.add_parser("delete", help="Remove a document and all its embeddings")
    delete.add_argument("document_id")
    serve = commands.add_parser("serve", help="Start an installed llama-server with CPU limits")
    serve.add_argument("--binary", type=Path, default=Path("tools/llama/llama-server.exe"))
    serve.add_argument("--model", type=Path,
                       default=Path("models/generation/qwen2.5-1.5b-instruct-q4_k_m.gguf"), help="local GGUF file")
    serve.add_argument("--port", type=positive, default=8080)
    return root


def prepare_model(path: Path, revision: str):
    from huggingface_hub import HfApi, snapshot_download

    manifest = path / "download.json"
    if manifest.is_file():
        prepared = json.loads(manifest.read_text())
        if revision not in ("main", prepared["revision"]):
            raise ValueError("Use a new --embedding-model folder to download a different revision.")
        print(f"Embedding model already prepared in {path}.")
        return
    if path.exists() and any(path.iterdir()) and not (path / ".cache/huggingface").is_dir():
        raise ValueError("Model folder is not empty. Use a new --embedding-model folder.")
    repo = "sentence-transformers/all-MiniLM-L6-v2"
    resolved = HfApi().model_info(repo, revision=revision).sha
    snapshot_download(
        repo_id=repo, revision=resolved, local_dir=path,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "README.md"],
        ignore_patterns=["onnx/*", "openvino/*"], max_workers=2,
    )
    (path / "download.json").write_text(json.dumps({"repository": repo, "revision": resolved}, indent=2))
    print(f"Embedding model downloaded to {path}. Runtime commands now load local files only.")


def show_matches(matches):
    for number, match in enumerate(matches, 1):
        print(f"\n[{number}] {match.chunk.source}, page {match.chunk.page} (similarity {match.score:.3f})")
        print(match.chunk.text)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = None
    generator = None
    try:
        if args.command == "prepare":
            prepare_model(args.embedding_model, args.revision)
            return 0
        if args.command == "prepare-generation":
            from src.setup_models import prepare_generation

            prepare_generation(args.model_dir, args.runtime_dir)
            return 0
        if args.command == "serve":
            if not args.binary.is_file() or not args.model.is_file():
                raise ValueError("Provide an existing llama-server binary and GGUF model file.")
            if args.port > 65535:
                raise ValueError("Port must be at most 65535.")
            if not args.server_key_file.exists():
                args.server_key_file.parent.mkdir(parents=True, exist_ok=True)
                with args.server_key_file.open("x") as handle:
                    handle.write(secrets.token_urlsafe(32) + "\n")
            return subprocess.call([
                str(args.binary.resolve()), "-m", str(args.model.resolve()),
                "--host", "127.0.0.1", "--port", str(args.port),
                "--threads", str(args.threads), "--threads-batch", str(args.threads),
                "--ctx-size", "4096", "--parallel", "1", "--n-gpu-layers", "0",
                "--alias", "local", "--n-predict", "384",
                "--api-key-file", str(args.server_key_file.resolve()),
                "--cors-origins", "localhost", "--no-webui", "--no-slots",
                "--sleep-idle-seconds", "120", "--prio", "-1", "--poll", "0",
            ])
        store = VectorStore(args.index)
        if args.command == "list":
            print(json.dumps(store.documents(), indent=2))
            return 0
        if args.command == "delete":
            if not store.delete_document(args.document_id):
                raise ValueError("Document ID was not found.")
            print("Document and embeddings deleted.")
            return 0
        embedder = LocalEmbeddings(args.embedding_model, threads=args.threads)
        if args.command == "ask":
            if args.provider == "groq":
                generator = GroqGenerator(model=args.groq_model)
            else:
                api_key = args.server_key_file.read_text().strip() if args.server_key_file.is_file() else None
                generator = LocalGenerator(args.server, model=args.model, api_key=api_key,
                                           answer_mode=args.answer_mode)
        rag = RAGPipeline(embedder, store, generator)
        if args.command == "ingest":
            paths = []
            for path in args.paths:
                paths.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
            if not paths:
                raise ValueError("No PDF files found.")
            for path in paths:
                print(f"Indexing {path.name}...", flush=True)
                added = rag.ingest_document(path, args.chunk_size, args.overlap)
                print("Indexed." if added else "Already indexed; skipped.", flush=True)
        elif args.command == "search":
            matches = rag.retrieve(args.question, args.top_k, args.document)
            if args.json:
                print(json.dumps([asdict(match) for match in matches], indent=2))
            elif matches:
                show_matches(matches)
            else:
                print("No passages found for the selected document.")
        elif args.command == "ask":
            if args.provider == "groq":
                from src.rag import Answer

                matches = rag.retrieve(args.question, args.top_k, args.document)
                text = generator.generate(args.question, [(f"{item.chunk.source}, page {item.chunk.page}", item.chunk.text)
                                                          for item in matches], task=args.task)
                answer = Answer(text=text, sources=matches)
            else:
                answer = rag.ask(args.question, args.top_k, args.document)
            if args.json:
                print(json.dumps(asdict(answer), indent=2))
            else:
                print(answer.text)
                show_matches(answer.sources)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, httpx.HTTPError, PdfReadError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    finally:
        if generator is not None:
            generator.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
