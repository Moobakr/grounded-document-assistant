import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def split_text(self, text: str, size: int, overlap: int) -> list[str]: ...


class LocalEmbeddings:
    def __init__(self, model_path: str | Path, threads: int = 2, batch_size: int = 8):
        self.model_path = Path(model_path).resolve()
        if not (self.model_path / "modules.json").is_file():
            raise ValueError("Embedding model is missing. Run: python -m src.cli prepare")
        if threads < 1 or batch_size < 1:
            raise ValueError("Thread count and batch size must be positive.")
        self.threads = threads
        self.batch_size = batch_size
        self._model = None
        self._identity = None

    @property
    def model(self):
        if self._model is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["USE_TF"] = "0"
            os.environ["USE_FLAX"] = "0"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.threads)
            self._model = SentenceTransformer(
                str(self.model_path), device="cpu", local_files_only=True, trust_remote_code=False,
            )
        return self._model

    @property
    def identity(self) -> str:
        if self._identity is None:
            digest = hashlib.sha256()
            files = sorted(path for path in self.model_path.rglob("*")
                           if path.is_file() and ".cache" not in path.parts
                           and path.suffix in {".json", ".txt", ".safetensors", ".bin", ".model"})
            for path in files:
                digest.update(path.relative_to(self.model_path).as_posix().encode())
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            self._identity = "sentence-transformers:normalized-v1:" + digest.hexdigest()
        return self._identity

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty batch.")
        return np.asarray(self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ), dtype=np.float32)

    def split_text(self, text: str, size: int = 180, overlap: int = 30) -> list[str]:
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        if not 0 < size <= limit or not 0 <= overlap < size:
            raise ValueError(f"Use chunk size 1..{limit} and overlap smaller than chunk size.")
        text = text.strip()
        if not text:
            return []
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)
        offsets = encoded["offset_mapping"]
        chunks = []
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            chunk = text[offsets[start][0]:offsets[end - 1][1]].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(offsets):
                break
        return chunks

