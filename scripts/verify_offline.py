"""Exercise real PDF ingestion and retrieval while rejecting external TCP connections."""

import argparse
from dataclasses import asdict
import ipaddress
import json
from importlib.metadata import version
from pathlib import Path
import socket
import tempfile
from time import perf_counter
from unittest.mock import patch

from src.embeddings import LocalEmbeddings
from src.generation import LocalGenerator
from src.rag import RAGPipeline
from src.vector_store import VectorStore


original_connect = socket.socket.connect


def local_connect(connection, address):
    try:
        allowed = ipaddress.ip_address(address[0]).is_loopback
    except (ValueError, TypeError):
        allowed = False
    if not allowed:
        raise AssertionError("Offline check blocked an external TCP connection.")
    return original_connect(connection, address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    pdfs = sorted(Path("documents").glob("*.pdf"))
    if not pdfs:
        raise ValueError("No PDFs found in documents/.")
    Path("data").mkdir(exist_ok=True)
    report = {"external_tcp_connections_blocked": True, "cpu_threads": 2}
    report["packages"] = {name: version(name) for name in (
        "numpy", "pypdf", "sentence-transformers", "transformers", "torch", "huggingface-hub", "httpx",
    )}
    key_path = Path("data/local-server.key")
    api_key = key_path.read_text().strip() if key_path.is_file() else None
    generator = LocalGenerator(args.server, api_key=api_key) if args.generate else None
    try:
        with patch.object(socket.socket, "connect", local_connect), \
                tempfile.TemporaryDirectory(dir="data") as directory:
            index = Path(directory) / "index.sqlite3"
            store = VectorStore(index)
            try:
                rag = RAGPipeline(LocalEmbeddings("models/embeddings"), store)
                started = perf_counter()
                for pdf in pdfs:
                    assert rag.ingest_document(pdf), "First ingestion unexpectedly skipped a PDF."
                report["ingestion_seconds"] = round(perf_counter() - started, 3)
                for pdf in pdfs:
                    assert not rag.ingest_document(pdf), "Duplicate PDF was indexed again."
                report["documents"] = len(store.documents())
                report["chunks"] = sum(document["chunks"] for document in store.documents())
            finally:
                store.close()
            store = VectorStore(index)
            try:
                rag = RAGPipeline(LocalEmbeddings("models/embeddings"), store, generator)
                question = "What topics are listed under Course Contents?"
                report["question"] = question
                started = perf_counter()
                matches = rag.retrieve(question)
                assert matches, "Reopened index returned no results."
                assert any("regression" in match.chunk.text.lower() for match in matches)
                report["retrieval_after_reopen_seconds"] = round(perf_counter() - started, 3)
                report["sources"] = [asdict(match) for match in matches]
                if args.generate:
                    started = perf_counter()
                    report["answer"] = rag.ask(question).text
                    report["answer_seconds"] = round(perf_counter() - started, 3)
                    missing_question = "What is the lecturer's favorite color?"
                    missing_answer = rag.ask(missing_question).text
                    assert missing_answer == (
                        "The indexed documents do not provide enough information to answer this question."
                    ), "Local model failed the unanswerable-question smoke check."
                    report["unanswerable_question"] = missing_question
                    report["unanswerable_response"] = missing_answer
            finally:
                store.close()
    finally:
        if generator is not None:
            generator.close()
    Path("data/verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
