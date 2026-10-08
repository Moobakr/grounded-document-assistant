import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile

import httpx


GENERATION_REPOSITORY = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GENERATION_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLAMA_RELEASE = "b11499"
RUNTIME_SHA256 = "5e4b71e4f48780b354d2c30668c2b4a1b6d21b441e969a18b5028b886179dfa8"


def download_file(url: str, path: Path, expected_hash: str):
    if path.is_file() and file_hash(path) == expected_hash:
        print(f"Already downloaded: {path.name}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if offset and response.status_code != 206:
                offset = 0
            transferred = offset
            last_report = transferred
            with partial.open("ab" if offset else "wb") as handle:
                for block in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(block)
                    transferred += len(block)
                    if transferred - last_report >= 50 * 1024 * 1024:
                        print(f"{path.name}: {transferred // (1024 * 1024)} MB downloaded", flush=True)
                        last_report = transferred
    if file_hash(partial) != expected_hash:
        raise ValueError(f"Download checksum does not match: {path.name}. Remove the .part file and retry.")
    partial.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_generation(model_dir: Path, runtime_dir: Path):
    """Explicit online setup; ordinary generation never downloads files."""
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError("Automatic runtime setup supports Windows x64. Install llama-server manually on other platforms.")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import HfApi

    model_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(GENERATION_REPOSITORY, files_metadata=True)
    revision = info.sha
    model_file = next(file for file in info.siblings if file.rfilename == GENERATION_FILE)
    expected_hash = model_file.lfs.sha256
    print("Downloading the free 1.12 GB instruction model...", flush=True)
    download_file(
        f"https://huggingface.co/{GENERATION_REPOSITORY}/resolve/{revision}/{GENERATION_FILE}?download=true",
        model_dir / GENERATION_FILE, expected_hash,
    )
    (model_dir / "download.json").write_text(json.dumps({
        "repository": GENERATION_REPOSITORY, "revision": revision,
        "file": GENERATION_FILE, "sha256": expected_hash,
    }, indent=2))
    url = (f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/"
           f"llama-{LLAMA_RELEASE}-bin-win-cpu-x64.zip")
    print("Downloading the 20 MB Windows CPU runtime...", flush=True)
    with tempfile.TemporaryDirectory(dir=runtime_dir.parent) as temporary:
        archive = Path(temporary) / "runtime.zip"
        download_file(url, archive, RUNTIME_SHA256)
        with zipfile.ZipFile(archive) as zipped:
            root = runtime_dir.resolve()
            for member in zipped.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Runtime archive contains an invalid path.")
            zipped.extractall(root)
    (runtime_dir / "download.json").write_text(json.dumps({
        "repository": "ggml-org/llama.cpp", "release": LLAMA_RELEASE, "sha256": RUNTIME_SHA256,
    }, indent=2))
    print("Local generation setup complete. Run: python -m src.cli serve")
