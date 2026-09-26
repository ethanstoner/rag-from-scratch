"""Lessons 1-4: index, retrieve top-k by cosine similarity, generate."""
from ragfs.core.fusion import rrf
from ragfs.techniques.base import Technique


class Baseline(Technique):
    name = "baseline"
    lesson = "1-4"
    summary = "Dense top-k retrieval, then a grounded answer"

    def retrieve(self, question):
        return self.dense(question)


class Hybrid(Technique):
    """Not in the course: dense + BM25 fused with RRF, as a stronger reference point."""
    name = "hybrid"
    lesson = "extra"
    summary = "Dense + BM25, fused with reciprocal rank fusion"

    def retrieve(self, question):
        dense = self.dense(question, k=20)
        sparse = [c for c, _ in self.kit.bm25.search(question, 20)]
        return [c for c, _ in rrf([dense, sparse])][:self.k]
