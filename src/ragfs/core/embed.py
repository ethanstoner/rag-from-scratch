"""Ollama embedding client with a per-text disk cache (one jsonl file per model)."""
import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from ragfs.core.config import (CACHE_DIR, DOC_PREFIX, EMBED_BATCH, EMBED_MODEL,
                               OLLAMA_HOST, QUERY_PREFIX)


def _normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class OllamaEmbedder:
    def __init__(self, model=EMBED_MODEL, host=OLLAMA_HOST, cache_dir: Path | None = CACHE_DIR / "embed"):
        self.model, self.host = model, host
        self.cache_path = cache_dir / f"{model.replace(':', '_')}.jsonl" if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_path:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_path.exists():
                for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                    k, v = json.loads(line)
                    self._cache[k] = v
        self.seconds = 0.0

    @staticmethod
    def _key(text):
        return hashlib.sha256(text.encode()).hexdigest()

    def _fetch(self, texts):
        t0 = time.perf_counter()
        r = httpx.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts}, timeout=300.0)
        r.raise_for_status()
        self.seconds += time.perf_counter() - t0
        return r.json()["embeddings"]

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        new_lines = []
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i:i + EMBED_BATCH]
            for t, v in zip(batch, self._fetch(batch)):
                self._cache[self._key(t)] = v
                new_lines.append(json.dumps([self._key(t), v]))
        if new_lines and self.cache_path:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.asarray([self._cache[self._key(t)] for t in texts], dtype=np.float32))

    def embed_documents(self, texts):
        return self.embed([DOC_PREFIX + t for t in texts])

    def embed_query(self, text):
        return self.embed([QUERY_PREFIX + text])[0]
