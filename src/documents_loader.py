from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import hashlib

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    source: str
    page: int
    chunk_index: int
    text: str


def load_pdf(file_path: str | Path) -> str:
    reader = PdfReader(file_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    return [chunk for start in range(0, len(text), chunk_size - chunk_overlap)
            if (chunk := text[start:start + chunk_size].strip())]


def document_hash(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(file_path: str | Path, splitter: Callable[[str], list[str]]) -> list[DocumentChunk]:
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported.")
    document_id = document_hash(path)
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported. Use a decrypted copy.")
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        for text in splitter(page.extract_text() or ""):
            chunks.append(DocumentChunk(document_id, path.name, page_number, len(chunks), text))
    if not chunks:
        raise ValueError("The PDF has no extractable text. Scanned PDFs need OCR first.")
    return chunks
