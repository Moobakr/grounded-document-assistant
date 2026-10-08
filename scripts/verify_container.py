"""Smoke-test real CPU retrieval in the Compose workspace, cleaning up only its fixture.

No Groq key is needed and no cloud inference is performed. Run with the app/model
already provisioned. An optional explicit restart checks persistence of the fixture.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageStat

from scripts.verify_web import fixture_pdf


ROOT = Path(__file__).resolve().parent.parent


def wait_ready(client):
    for attempt in range(60):
        try:
            response = client.get("/api/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("Container web server did not become healthy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--restart", action="store_true", help="Restart this repository's Compose app to check persistence.")
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or url.username or url.password or url.path not in ("", "/"):
        raise ValueError("Verification accepts only a local HTTP workspace URL.")
    report = {"checks": [], "cloud_inference": False}

    def checked(label):
        report["checks"].append(label)
        print("Verified " + label, flush=True)

    payload = fixture_pdf()
    fixture_id = hashlib.sha256(payload).hexdigest()
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False, follow_redirects=False) as client:
        health = wait_ready(client)
        assert health["embedding_ready"], "Run docker compose run --rm model-setup first."
        assert not client.get("/api/ai").json()["configured"], "Use an unconfigured verification workspace; no real cloud key should be active."
        original = {doc["id"] for doc in client.get("/api/documents").json()}
        assert fixture_id not in original, "Verification fixture already exists; it will not be modified."
        token = client.get("/api/status").json()["token"]
        headers = {"X-Workspace-Token": token}
        checked("healthy application and separately provisioned model")
        try:
            uploaded = client.post("/api/documents", files={"file": ("container-verification.pdf", payload, "application/pdf")}, headers=headers)
            uploaded.raise_for_status()
            assert uploaded.json()["document"]["id"] == fixture_id
            checked("real local model loads and indexes a PDF under container limits")
            duplicate = client.post("/api/documents", files={"file": ("renamed-verification.pdf", payload, "application/pdf")}, headers=headers)
            duplicate.raise_for_status()
            assert duplicate.json()["status"] == "duplicate"
            result = client.post("/api/query", json={"question": "Local upload and indexing", "document_id": fixture_id}, headers=headers)
            result.raise_for_status()
            sources = result.json()["sources"]
            assert sources and sources[0]["chunk"]["document_id"] == fixture_id and sources[0]["chunk"]["page"] == 1
            assert "Local upload" in sources[0]["chunk"]["text"]
            assert result.json()["answer"] is None and result.json()["provider"] == "local"
            report["search_seconds"] = result.json()["seconds"]
            checked("content deduplication, scoped cosine retrieval, and correct page metadata")
            preview = client.get(f"/api/documents/{fixture_id}/preview/1")
            preview.raise_for_status()
            with Image.open(io.BytesIO(preview.content)) as image:
                assert max(ImageStat.Stat(image).stddev) > 5, "Rendered page is blank."
            assert client.get(f"/api/documents/{fixture_id}/file").content == payload
            checked("real PDF rendering and verified original download")
            response = client.post("/api/query", json={"question": "Explain local upload", "operation": "ask", "document_id": fixture_id}, headers=headers)
            response.raise_for_status()
            assert response.json()["generation_error"] and response.json()["sources"]
            checked("missing Groq key retains retrieved passages without cloud inference")
            if args.restart:
                identity = subprocess.run(["docker", "compose", "ps", "-q", "app"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
                if not identity:
                    raise RuntimeError("No project Compose app is running; restart was not attempted.")
                bindings = json.loads(subprocess.run(["docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", identity],
                                                    check=True, capture_output=True, text=True).stdout)
                assert any(binding["HostIp"] == "127.0.0.1" and int(binding["HostPort"]) == (url.port or 80)
                           for binding in bindings.get("8765/tcp", [])), "URL is not this Compose app; no restart attempted."
                subprocess.run(["docker", "compose", "restart", "app"], cwd=ROOT, check=True)
                wait_ready(client)
                token = client.get("/api/status").json()["token"]
                headers = {"X-Workspace-Token": token}
                docs = {doc["id"] for doc in client.get("/api/documents").json()}
                assert docs == original | {fixture_id}
                assert client.get(f"/api/documents/{fixture_id}/file").content == payload
                checked("document/index/original persistence across explicit container restart")
        finally:
            token = client.get("/api/status").json()["token"]
            headers = {"X-Workspace-Token": token}
            docs = {doc["id"] for doc in client.get("/api/documents").json()}
            if fixture_id in docs:
                client.delete(f"/api/documents/{fixture_id}", headers=headers).raise_for_status()
            assert {doc["id"] for doc in client.get("/api/documents").json()} == original
            checked("fixture cleanup preserves the original workspace")
    output = ROOT / "data/container-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
