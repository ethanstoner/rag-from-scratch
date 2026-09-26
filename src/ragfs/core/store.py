"""In-memory vector store on numpy. Vectors are unit length, so cosine is a dot product."""
import json
from pathlib import Path
from typing import Callable

import numpy as np

from ragfs.core.types import Chunk


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def __len__(self):
        return len(self.chunks)

    def add(self, chunks: list[Chunk], texts: list[str] | None = None):
        """Add chunks. `texts` overrides what gets embedded (e.g. a summary standing in for a doc)."""
        vecs = self.embedder.embed_documents(texts if texts is not None else [c.text for c in chunks])
        self.vectors = vecs if not self.chunks else np.vstack([self.vectors, vecs])
        self.chunks.extend(chunks)

    def search_vector(self, qvec: np.ndarray, k: int, where: Callable[[Chunk], bool] | None = None):
        if not self.chunks:
            return []
        scores = self.vectors @ qvec
        if where is not None:
            mask = np.array([where(c) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def search(self, query: str, k: int, where=None) -> list[tuple[Chunk, float]]:
        return self.search_vector(self.embedder.embed_query(query), k, where)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.__dict__) for c in self.chunks) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder):
        store = cls(embedder)
        store.vectors = np.load(path / "vectors.npy")
        store.chunks = [Chunk(**json.loads(ln)) for ln in
                        (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        return store
