"""Okapi BM25 over a fixed list of chunks."""
import math
import re
from collections import Counter

from ragfs.core.types import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("a an and are as at be by for from has have how in is it its of on or that the "
                "this to was were what when where which who why with do does did can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for d, dl in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scored = sorted(zip(self.chunks, self.scores(query)), key=lambda x: -x[1])
        return [(c, s) for c, s in scored[:k] if s > 0]
