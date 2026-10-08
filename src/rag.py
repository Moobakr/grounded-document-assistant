from dataclasses import dataclass
from pathlib import Path

from src.documents_loader import document_hash, load_chunks
from src.embeddings import EmbeddingProvider
from src.generation import GenerationProvider
from src.vector_store import SearchMatch, VectorStore


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchMatch]


class RAGPipeline:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore,
                 generator: GenerationProvider | None = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator

    def ingest_document(self, pdf_path: str | Path, chunk_size: int = 180, chunk_overlap: int = 30) -> bool:
        self.vector_store.check_identity(self.embedder.identity)
        if self.vector_store.contains(document_hash(pdf_path)):
            return False
        chunks = load_chunks(pdf_path, lambda text: self.embedder.split_text(text, chunk_size, chunk_overlap))
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        return self.vector_store.add_document(chunks, vectors, self.embedder.identity)

    def retrieve(self, query: str, top_k: int = 3, document_id: str | None = None) -> list[SearchMatch]:
        query = query.strip()
        if not query:
            raise ValueError("Question must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not self.vector_store.documents():
            raise ValueError("No documents indexed. Use the ingest command first.")
        self.vector_store.check_identity(self.embedder.identity)
        vector = self.embedder.encode([query])[0]
        candidates = self.vector_store.search(vector, self.embedder.identity, top_k * 3, document_id)
        matches = []
        seen = set()
        for match in candidates:
            key = " ".join(match.chunk.text.split())
            if key not in seen:
                seen.add(key)
                matches.append(match)
            if len(matches) == top_k:
                break
        return matches

    def ask(self, query: str, top_k: int = 3, document_id: str | None = None) -> Answer:
        matches = self.retrieve(query, top_k, document_id)
        if not matches:
            return Answer("No passages found for the selected document.", [])
        if self.generator is None:
            raise ValueError("No generation provider configured. Use search or configure llama-server.")
        sources = [(f"{m.chunk.source}, page {m.chunk.page}", m.chunk.text) for m in matches]
        return Answer(self.generator.generate(query.strip(), sources), matches)
