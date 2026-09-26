import hashlib
import json

import numpy as np
import pytest

from ragfs.core.bm25 import tokenize


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared words => similar vectors."""
    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self.embed([text])[0]


class FakeLLM:
    """Replies via `responder(prompt) -> str | dict`; records every prompt."""

    def __init__(self, responder=lambda p: "stub answer"):
        self.responder, self.prompts, self.calls, self.model_seconds, self.real_seconds = responder, [], 0, 0.0, 0.0

    def reset_counters(self):
        self.calls, self.model_seconds, self.real_seconds = 0, 0.0, 0.0

    def complete(self, prompt, system=None, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        out = self.responder(prompt)
        return json.dumps(out) if isinstance(out, dict) else out

    def json(self, prompt, schema, system=None, retries=1):
        return json.loads(self.complete(prompt, system=system))


@pytest.fixture
def embedder():
    return FakeEmbedder()
