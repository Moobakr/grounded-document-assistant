"""Run a versioned development evaluation against an existing local index."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import socket
import sys
from time import perf_counter
from unittest.mock import patch

from scripts.verify_offline import local_connect
from src.cli import positive
from src.embeddings import LocalEmbeddings
from src.generation import ABSTENTION, LocalGenerator, normalize_quote
from src.rag import RAGPipeline
from src.vector_store import VectorStore


def load_cases(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list) or not dataset["cases"]:
        raise ValueError("Evaluation dataset must contain a nonempty cases list.")
    seen = set()
    for case in dataset["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("Each evaluation case needs a nonempty ID.")
        if case["id"] in seen:
            raise ValueError("Evaluation case IDs must be unique.")
        seen.add(case["id"])
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("Each case needs a question.")
        if type(case.get("answerable")) is not bool:
            raise ValueError("Each case needs a boolean answerable label.")
        if case["answerable"]:
            references = case.get("relevant_pages")
            if not isinstance(references, list) or not references:
                raise ValueError("Answerable cases need relevant_pages labels.")
            for reference in references:
                if (not isinstance(reference, dict) or not isinstance(reference.get("source"), str)
                        or type(reference.get("page")) is not int or reference["page"] < 1):
                    raise ValueError("Relevant pages need a source filename and positive PDF page.")
            if not case.get("required_patterns"):
                raise ValueError("Answerable cases need required_patterns for answer checks.")
        for field in ("required_patterns", "forbidden_patterns"):
            patterns = case.get(field, [])
            if not isinstance(patterns, list) or not all(isinstance(value, str) and value for value in patterns):
                raise ValueError("Answer patterns must be nonempty strings in lists.")
            for value in patterns:
                re.compile(value, re.IGNORECASE)
    return dataset


def relevant(case: dict, match) -> bool:
    return any(match.chunk.source == page["source"] and match.chunk.page == page["page"]
               for page in case.get("relevant_pages", []))


def score_answer(case: dict, text: str, matches: list, mode: str) -> dict:
    abstained = text == ABSTENTION
    checks = {"abstained": abstained}
    if case["answerable"]:
        checks["required_patterns_found"] = [bool(re.search(pattern, text, re.IGNORECASE))
                                             for pattern in case["required_patterns"]]
        checks["forbidden_patterns_found"] = [bool(re.search(pattern, text, re.IGNORECASE))
                                              for pattern in case.get("forbidden_patterns", [])]
        checks["answer_check_pass"] = (not abstained and all(checks["required_patterns_found"])
                                       and not any(checks["forbidden_patterns_found"]))
    else:
        checks["answer_check_pass"] = abstained
    checks.update(citation_in_range=None, citation_expected_page=None, quote_grounding=None)
    if abstained:
        return checks
    if mode == "evidence":
        references, grounded = [], True
        for line in text.splitlines():
            found = re.fullmatch(r'("(?:[^"\\]|\\.)*") \[(\d+)\]', line)
            if not found:
                grounded = False
                continue
            quote, reference = json.loads(found[1]), int(found[2])
            references.append(reference)
            if (not 1 <= reference <= len(matches)
                    or not normalize_quote(quote)
                    or normalize_quote(quote) not in normalize_quote(matches[reference - 1].chunk.text)):
                grounded = False
        checks["quote_grounding"] = grounded and bool(references)
    else:
        references = [int(value) for value in re.findall(r"\[(\d+)\]", text)]
    checks["citation_in_range"] = bool(references) and all(1 <= value <= len(matches) for value in references)
    if case["answerable"]:
        checks["citation_expected_page"] = (checks["citation_in_range"] and any(
            relevant(case, matches[value - 1]) for value in references
        ))
    return checks


def rate(values: list) -> dict:
    return {"passed": sum(bool(value) for value in values), "total": len(values),
            "rate": round(sum(bool(value) for value in values) / len(values), 4) if values else None}


def summarize(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    generated = [row for row in rows if "checks" in row]
    summary = {
        "cases": len(rows), "operational_errors": sum("error" in row for row in rows),
        "retrieval_hit_at_k": rate([row.get("retrieval_hit", False) for row in answerable]),
        "mean_reciprocal_page_rank": (round(sum(row.get("reciprocal_rank", 0) for row in answerable)
                                            / len(answerable), 4) if answerable else None),
    }
    if any(row.get("generation_requested") for row in rows):
        summary["answer_checks"] = rate([row.get("checks", {}).get("answer_check_pass", False) for row in rows])
        summary["answerable_checks"] = rate([row.get("checks", {}).get("answer_check_pass", False)
                                              for row in answerable])
        missing = [row for row in rows if not row["answerable"]]
        summary["unanswerable_abstention"] = rate([row.get("checks", {}).get("abstained", False) for row in missing])
        summary["false_abstention"] = rate([row.get("checks", {}).get("abstained", False) for row in answerable])
        summary["answered_outputs"] = sum(not row["checks"]["abstained"] for row in generated)
        for field in ("citation_in_range", "citation_expected_page", "quote_grounding"):
            summary[field] = rate([row["checks"][field] for row in generated if row["checks"][field] is not None])
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index.sqlite3"))
    parser.add_argument("--embedding-model", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--generation-model-dir", type=Path, default=Path("models/generation"))
    parser.add_argument("--threads", type=positive, default=2)
    parser.add_argument("--top-k", type=positive, default=3)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--answer-mode", choices=("evidence", "draft"), default="evidence")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--server-key-file", type=Path, default=Path("data/local-server.key"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation.json"))
    parser.add_argument("--strict", action="store_true", help="exit nonzero for failed development checks")
    args = parser.parse_args(argv)
    if not args.index.is_file():
        parser.error("Index does not exist. Ingest the evaluation documents first.")
    if args.output.resolve() in (args.index.resolve(), args.cases.resolve(), args.server_key_file.resolve()):
        parser.error("Report output must not overwrite the index, dataset, or server key.")
    dataset = load_cases(args.cases)
    store, generator = VectorStore(args.index), None
    try:
        documents = store.documents()
        available = {document["source"] for document in documents}
        required = {page["source"] for case in dataset["cases"] for page in case.get("relevant_pages", [])}
        if required - available:
            raise ValueError("Missing evaluation documents: " + ", ".join(sorted(required - available)))
        api_key = args.server_key_file.read_text().strip() if args.server_key_file.is_file() else None
        if args.generate:
            generator = LocalGenerator(args.server, api_key=api_key, answer_mode=args.answer_mode)
        embedder = LocalEmbeddings(args.embedding_model, threads=args.threads)
        rows = []
        with patch.object(socket.socket, "connect", local_connect):
            rag = RAGPipeline(embedder, store, generator)
            for case in dataset["cases"]:
                row = {"id": case["id"], "question": case["question"], "answerable": case["answerable"],
                       "generation_requested": args.generate, "rubric": case.get("rubric")}
                try:
                    started = perf_counter()
                    matches = rag.retrieve(case["question"], args.top_k)
                    row["retrieval_seconds"] = round(perf_counter() - started, 3)
                    row["sources"] = [asdict(match) for match in matches]
                    if case["answerable"]:
                        ranks = [i for i, match in enumerate(matches, 1) if relevant(case, match)]
                        row["retrieval_hit"] = bool(ranks)
                        row["reciprocal_rank"] = 1 / ranks[0] if ranks else 0
                    if generator is not None:
                        started = perf_counter()
                        sources = [(f"{match.chunk.source}, page {match.chunk.page}", match.chunk.text)
                                   for match in matches]
                        text = generator.generate(case["question"], sources)
                        row["generation_seconds"] = round(perf_counter() - started, 3)
                        row["answer"] = text
                        row["checks"] = score_answer(case, text, matches, args.answer_mode)
                except (ValueError, RuntimeError, OSError) as error:
                    row["error"] = str(error)
                rows.append(row)
                print(f"Evaluated {case['id']}", flush=True)
        manifests = {}
        for name, directory in (("embeddings", args.embedding_model), ("generation", args.generation_model_dir)):
            manifest = directory / "download.json"
            if manifest.is_file():
                manifests[name] = json.loads(manifest.read_text(encoding="utf-8"))
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {"path": str(args.cases), "sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                        "purpose": dataset.get("purpose"), "version": dataset.get("version")},
            "configuration": {"top_k": args.top_k, "threads": args.threads, "generation": args.generate,
                              "answer_mode": args.answer_mode, "seed": 42, "temperature": 0},
            "external_tcp_blocked_in_evaluator": True,
            "embedding_identity": embedder.identity, "models": manifests, "documents": documents,
            "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                "src/generation.py", "src/rag.py", "src/vector_store.py", "scripts/evaluate.py",
            )},
            "packages": {name: version(name) for name in ("numpy", "sentence-transformers", "torch", "httpx")},
            "limitations": ["Development cases, not a held-out benchmark.",
                            "Regex answer checks and page labels are proxies, not semantic correctness.",
                            "Quote matching proves text provenance, not relevance or entailment.",
                            "TCP blocking covers this Python process, not the separate model server.",
                            "Local artifact manifests do not attest which model a separately started server loaded.",
                            "First retrieval includes model load; timings are not a performance benchmark."],
            "summary": summarize(rows), "cases": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(args.output), "summary": report["summary"]}, indent=2))
        failed = report["summary"]["operational_errors"] > 0
        if args.strict:
            failed = failed or not all(row.get("retrieval_hit", True) for row in rows)
            if args.generate:
                failed = failed or not all(row.get("checks", {}).get("answer_check_pass", False) for row in rows)
                failed = failed or any(row.get("checks", {}).get("citation_in_range") is False for row in rows)
                failed = failed or any(row.get("checks", {}).get("citation_expected_page") is False for row in rows)
                failed = failed or any(row.get("checks", {}).get("quote_grounding") is False for row in rows)
        return int(failed)
    finally:
        store.close()
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        raise SystemExit(1)
