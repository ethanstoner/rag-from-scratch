"""Ollama chat client. Every LLM call in the project goes through here.

Responses are cached on disk keyed by the full request, so eval reruns are free and
deterministic. The cache stores how long the original call took; `model_seconds`
accumulates that figure, so timing stays honest on a cache hit.
"""
import hashlib
import json
import time
from pathlib import Path

import httpx

from ragfs.core.config import CACHE_DIR, CHAT_MODEL, NUM_CTX, OLLAMA_HOST


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model=CHAT_MODEL, host=OLLAMA_HOST, cache_dir: Path | None = CACHE_DIR / "llm"):
        self.model, self.host = model, host
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.model_seconds = 0.0  # time the calls took when first made (cache hits included)
        self.real_seconds = 0.0   # time actually spent waiting on Ollama in this process

    def reset_counters(self):
        self.calls, self.model_seconds, self.real_seconds = 0, 0.0, 0.0

    def _cache_path(self, body):
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def chat(self, messages, temperature=0.0, format=None) -> str:
        body = {"model": self.model, "messages": messages, "stream": False,
                "think": False, "options": {"temperature": temperature, "num_ctx": NUM_CTX}}
        if format is not None:
            body["format"] = format
        self.calls += 1
        path = self._cache_path(body) if self.cache_dir else None
        if path and path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
            self.model_seconds += hit["elapsed"]
            return hit["content"]
        t0 = time.perf_counter()
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=600.0)
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        content = r.json()["message"]["content"]
        self.model_seconds += elapsed
        self.real_seconds += elapsed
        if path:
            path.write_text(json.dumps({"content": content, "elapsed": elapsed}), encoding="utf-8")
        return content

    def complete(self, prompt: str, system: str | None = None, **kw) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw).strip()

    def json(self, prompt: str, schema: dict, system: str | None = None, retries: int = 1) -> dict:
        """Structured output constrained by a JSON schema, validated for required keys."""
        last = None
        for attempt in range(retries + 1):
            nudge = "" if attempt == 0 else f"\n\nYour previous reply was invalid ({last}). Reply with JSON only."
            raw = self.complete(prompt + nudge, system=system, format=schema)
            try:
                data = json.loads(raw)
                missing = [k for k in schema.get("required", []) if k not in data]
                if missing:
                    raise ValueError(f"missing keys {missing}")
                return data
            except (json.JSONDecodeError, ValueError) as e:
                last = e
        raise LLMError(f"no valid JSON after {retries + 1} attempts: {last}")
