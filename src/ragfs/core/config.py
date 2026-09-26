"""Configuration. Single source of truth — do not hardcode these elsewhere."""
import os
from pathlib import Path

_DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"


def normalize_host(raw: str | None) -> str:
    """Turn an OLLAMA_HOST value into a URL a client can request.

    Ollama's OLLAMA_HOST is a *bind* address (often `0.0.0.0:11435`, no scheme).
    0.0.0.0 refuses connections when dialled, so rewrite it to loopback.
    """
    host = (raw or "").strip()
    if not host:
        return _DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = "http://" + host
    return host.replace("://0.0.0.0", "://127.0.0.1").rstrip("/")


OLLAMA_HOST = normalize_host(os.environ.get("OLLAMA_HOST"))

CHAT_MODEL = os.environ.get("RAGFS_CHAT_MODEL", "gemma4:latest")
JUDGE_MODEL = os.environ.get("RAGFS_JUDGE_MODEL", "qwen3:8b")
EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH = 32

# nomic-embed-text was trained with these prefixes; retrieval degrades without them.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# The corpus: Lilian Weng posts the course draws on.
POSTS = [
    "2023-06-23-agent",
    "2023-03-15-prompt-engineering",
    "2023-10-25-adv-attack-llm",
    "2024-02-05-human-data-quality",
    "2024-07-07-hallucination",
]
POST_URL = "https://lilianweng.github.io/posts/{slug}/"

# Chunking, in characters (the course's RecursiveCharacterTextSplitter defaults).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5
RRF_K = 60
