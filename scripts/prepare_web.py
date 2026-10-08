"""One-time online download of pinned Lucide browser icons; runtime is offline."""

import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

import httpx


def main():
    destination = Path("src/static/vendor")
    destination.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        metadata = client.get("https://registry.npmjs.org/lucide/0.468.0")
        metadata.raise_for_status()
        dist = metadata.json()["dist"]
        response = client.get(dist["tarball"])
        response.raise_for_status()
    algorithm, expected = dist["integrity"].split("-", 1)
    if algorithm != "sha512" or base64.b64encode(hashlib.sha512(response.content).digest()).decode() != expected:
        raise ValueError("Lucide package integrity check failed.")
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for source, target in (("package/dist/umd/lucide.min.js", "lucide.min.js"),
                               ("package/LICENSE", "LICENSE")):
            member = archive.extractfile(source)
            if member is None:
                raise ValueError("Expected Lucide asset is missing.")
            (destination / target).write_bytes(member.read())
    (destination / "manifest.json").write_text(json.dumps({
        "package": "lucide", "version": "0.468.0", "integrity": dist["integrity"],
    }, indent=2), encoding="utf-8")
    print("Local Lucide assets ready.")


if __name__ == "__main__":
    main()
