"""Retrieval and answer metrics, plus bootstrap confidence intervals."""
import re
import string
from collections import Counter

import numpy as np

from ragfs.core.types import Chunk

MIN_COVER = 0.5  # a chunk "contains" an evidence fact if it covers at least half its characters


def covers(chunk: Chunk, ev: dict) -> bool:
    if chunk.doc_id != ev["doc_id"] or "start" not in chunk.metadata:
        return False
    overlap = min(chunk.metadata["end"], ev["end"]) - max(chunk.metadata["start"], ev["start"])
    return overlap >= MIN_COVER * (ev["end"] - ev["start"])


def retrieval_metrics(contexts: list[Chunk], evidence: list[dict]) -> dict:
    """Evidence recall, all-evidence hit, MRR and document recall over the returned contexts."""
    if not evidence:
        return {}
    found = [any(covers(c, ev) for c in contexts) for ev in evidence]
    first = next((r for r, c in enumerate(contexts, 1) if any(covers(c, ev) for ev in evidence)), None)
    gold_docs = {ev["doc_id"] for ev in evidence}
    got_docs = {c.doc_id for c in contexts}
    return {"evidence_recall": sum(found) / len(found),
            "all_evidence": float(all(found)),
            "mrr": 1.0 / first if first else 0.0,
            "doc_recall": len(gold_docs & got_docs) / len(gold_docs)}


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def contains(pred: str, gold: str) -> float:
    """The MultiHop-RAG paper's accuracy: the gold answer appears in the response (as whole words)."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    return float(any(p[i:i + len(g)] == g for i in range(len(p) - len(g) + 1))) if g else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return float(p == g)
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(values, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean and percentile CI of per-query values."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, **kw):
    """CI of mean(a - b) over the same queries: 'does technique a beat b, and by how much'."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), **kw)
