from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np

from src.documents_loader import DocumentChunk


@dataclass(frozen=True)
class SearchMatch:
    chunk: DocumentChunk
    score: float


class VectorStore:
    """Persistent exact cosine search for a small, single-user document collection."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY (document_id, chunk_index)
            );
        """)

    def close(self):
        self.connection.close()

    def contains(self, document_id: str) -> bool:
        return self.connection.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone() is not None

    def _metadata(self) -> dict[str, str]:
        return dict(self.connection.execute("SELECT key, value FROM metadata"))

    def check_identity(self, identity: str):
        stored = self._metadata().get("embedding_identity")
        if stored is not None and stored != identity:
            raise ValueError("Embedding model differs from the index. Create a new index and re-ingest.")

    def add_document(self, chunks: list[DocumentChunk], vectors: np.ndarray, identity: str) -> bool:
        if not chunks:
            raise ValueError("Cannot index a document without chunks.")
        self.check_identity(identity)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
            raise ValueError("Embedding count or dimensions do not match the chunks.")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embeddings must be finite, nonzero vectors.")
        dimension = self._metadata().get("dimension")
        if dimension is not None and int(dimension) != vectors.shape[1]:
            raise ValueError("Embedding dimension differs from the index.")
        first = chunks[0]
        if any(c.document_id != first.document_id or c.source != first.source for c in chunks):
            raise ValueError("Chunks must belong to the same document.")
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        # Publish document metadata and all vectors together, or roll everything back.
        with self.connection:
            inserted = self.connection.execute("INSERT OR IGNORE INTO documents(id, source) VALUES (?, ?)",
                                               (first.document_id, first.source))
            if inserted.rowcount == 0:
                return False
            self.connection.executemany(
                "INSERT INTO chunks(document_id, page, chunk_index, text, vector) VALUES (?, ?, ?, ?, ?)",
                [(c.document_id, c.page, c.chunk_index, c.text, v.astype("<f4").tobytes())
                 for c, v in zip(chunks, vectors)],
            )
            self.connection.executemany("INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                                        [("embedding_identity", identity), ("dimension", str(vectors.shape[1]))])
        return True

    def search(self, query_vector: np.ndarray, identity: str, top_k: int = 3,
               document_id: str | None = None) -> list[SearchMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.check_identity(identity)
        query = np.asarray(query_vector, dtype=np.float32)
        dimension = self._metadata().get("dimension")
        if query.ndim != 1 or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("Query embedding must be a finite, nonzero vector.")
        if dimension is not None and query.size != int(dimension):
            raise ValueError("Query embedding dimension differs from the index.")
        query = query / np.linalg.norm(query)
        sql = """SELECT c.document_id, d.source, c.page, c.chunk_index, c.text, c.vector
                 FROM chunks c JOIN documents d ON d.id = c.document_id"""
        params = ()
        if document_id is not None:
            sql += " WHERE c.document_id = ?"
            params = (document_id,)
        matches = []
        for row in self.connection.execute(sql, params):
            vector = np.frombuffer(row[5], dtype="<f4")
            if vector.size != query.size:
                raise ValueError("Stored embedding dimension is invalid. Rebuild the index.")
            matches.append(SearchMatch(DocumentChunk(*row[:5]), float(np.dot(query, vector))))
            if len(matches) >= top_k * 2:
                matches = sorted(matches, key=lambda m: (-m.score, m.chunk.document_id, m.chunk.chunk_index))[:top_k]
        return sorted(matches, key=lambda m: (-m.score, m.chunk.document_id, m.chunk.chunk_index))[:top_k]

    def documents(self) -> list[dict]:
        rows = self.connection.execute("""
            SELECT d.id, d.source, d.created_at, COUNT(c.chunk_index)
            FROM documents d JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.source, d.id
        """)
        return [dict(zip(("id", "source", "created_at", "chunks"), row)) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self.connection:
            deleted = self.connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            if not self.connection.execute("SELECT 1 FROM documents LIMIT 1").fetchone():
                self.connection.execute("DELETE FROM metadata")
        return deleted.rowcount > 0
