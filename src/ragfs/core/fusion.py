"""Reciprocal rank fusion (Cormack et al., 2009) — the core of RAG-Fusion."""
from ragfs.core.config import RRF_K
from ragfs.core.types import Chunk


def rrf(rankings: list[list[Chunk]], k: int = RRF_K) -> list[tuple[Chunk, float]]:
    """Fuse ranked lists. Score encodes rank agreement across lists, not relevance."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(chunk.id, chunk)
    return [(by_id[i], s) for i, s in sorted(scores.items(), key=lambda x: -x[1])]


def unique(chunks: list[Chunk]) -> list[Chunk]:
    seen, out = set(), []
    for c in chunks:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out
