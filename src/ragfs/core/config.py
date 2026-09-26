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
# Set explicitly: Ollama's default window silently truncates long prompts (whole
# articles in multi-representation, many chunks in decomposition).
NUM_CTX = 16384
EMBED_BATCH = 32

# nomic-embed-text was trained with these prefixes; retrieval degrades without them.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/{file}"

# Chunking, in characters (the course's RecursiveCharacterTextSplitter defaults;
# MultiHop-RAG's own baselines used 256-token chunks, about the same).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Evidence for one query spans 2-4 articles, so generation sees more than the usual 4-5.
TOP_K = 8
RRF_K = 60
