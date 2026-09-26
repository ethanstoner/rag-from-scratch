"""Lessons 12-13: change what gets indexed.

Multi-representation indexes an LLM summary of each article but hands the generator
the article itself. RAPTOR clusters summaries recursively into a tree and searches
every level of it at once ("collapsed tree").

Deviation from RAPTOR: level 1 is one summary per article rather than summaries of
clusters of leaf chunks, which suits a corpus of short news articles and reuses the
multi-representation summaries.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from ragfs.core.store import VectorStore
from ragfs.core.types import Chunk
from ragfs.techniques.base import Technique

SUMMARY_INPUT_CHARS = 12000  # longest articles exceed the context window; summarise the opening

DOC_SUMMARY_PROMPT = """Summarise this news article in 3-5 sentences. Name the outlet, the people, companies,
teams and products involved, the key events, and any figures or dates.

Outlet: {source} ({date})
Title: {title}

{text}

Summary:"""

CLUSTER_SUMMARY_PROMPT = """Here are summaries of related news articles. Write one detailed summary (5-8 sentences)
covering the themes, people, companies and events they share, and where they differ.

{text}

Summary:"""


def article_summaries(kit) -> list[Chunk]:
    # Not memoised: the LLM disk cache makes repeats cheap, and going through the LLM
    # each time keeps every technique's reported index cost complete.
    out = []
    for d in kit.docs:
        s = kit.llm.complete(DOC_SUMMARY_PROMPT.format(
            source=d.source, date=d.date, title=d.title, text=d.text[:SUMMARY_INPUT_CHARS]))
        out.append(Chunk(id=f"{d.id}#summary", text=s, doc_id=d.id,
                         metadata={**d.metadata(), "level": 1}))
    return out


class MultiRepresentation(Technique):
    name = "multi_representation"
    lesson = "12"
    summary = "Search article summaries, return whole articles"
    n_docs = 3
    max_doc_chars = 10000

    def prepare(self):
        self.summary_store()

    def summary_store(self):
        if "summary_store" not in self.kit.extras:
            store = VectorStore(self.kit.embedder)
            store.add(article_summaries(self.kit))
            self.kit.extras["summary_store"] = store
        return self.kit.extras["summary_store"]

    def retrieve(self, question):
        hits = self.summary_store().search(question, self.n_docs)
        out = []
        for summary, _ in hits:
            d = self.kit.doc(summary.doc_id)
            text = d.text[:self.max_doc_chars]
            out.append(Chunk(id=f"{d.id}#full", text=text, doc_id=d.id,
                             metadata={**d.metadata(), "start": 0, "end": len(text)}))
        return out


def _cluster(vectors: np.ndarray, seed: int = 0, threshold: float = 0.1) -> list[list[int]]:
    """GMM on PCA-reduced vectors; cluster count by BIC; soft membership above `threshold`."""
    n = len(vectors)
    if n <= 3:
        return [list(range(n))]
    dims = min(10, n - 2)
    reduced = PCA(n_components=dims, random_state=seed).fit_transform(vectors)
    candidates = range(2, min(max(3, n // 4), 60) + 1)
    best = min((GaussianMixture(k, random_state=seed).fit(reduced) for k in candidates),
               key=lambda g: g.bic(reduced))
    probs = best.predict_proba(reduced)
    clusters = [list(np.where(probs[:, j] > threshold)[0]) for j in range(best.n_components)]
    return [c for c in clusters if c]


class Raptor(Technique):
    name = "raptor"
    lesson = "13"
    summary = "Recursive cluster-and-summarise tree, searched at every level"
    max_levels = 3
    cluster_input_chars = 12000

    def prepare(self):
        self.tree_store()

    def tree_store(self):
        if "raptor_store" not in self.kit.extras:
            kit = self.kit
            nodes = list(article_summaries(kit))
            level_nodes = nodes
            for level in range(2, self.max_levels + 2):
                if len(level_nodes) <= 4:
                    break
                vecs = kit.embedder.embed_documents([c.text for c in level_nodes])
                parents = []
                for j, members in enumerate(_cluster(vecs)):
                    text = "\n\n".join(level_nodes[i].text for i in members)[:self.cluster_input_chars]
                    s = kit.llm.complete(CLUSTER_SUMMARY_PROMPT.format(text=text))
                    parents.append(Chunk(id=f"L{level}#{j}", text=s, doc_id=f"L{level}#{j}", metadata={
                        "level": level, "source": "cluster summary", "date": "",
                        "title": f"Summary of {len(members)} level-{level - 1} nodes",
                        "children": [level_nodes[i].id for i in members]}))
                nodes.extend(parents)
                if len(parents) >= len(level_nodes):
                    break
                level_nodes = parents
            store = VectorStore(kit.embedder)
            store.chunks, store.vectors = list(kit.store.chunks), kit.store.vectors
            store.add(nodes)
            kit.extras["raptor_store"] = store
            kit.extras["raptor_levels"] = {lv: sum(1 for c in nodes if c.metadata.get("level") == lv)
                                           for lv in range(1, self.max_levels + 2)}
        return self.kit.extras["raptor_store"]

    def retrieve(self, question):
        hits = self.tree_store().search(question, self.k)
        self.trace["levels"] = [c.metadata.get("level", 0) for c, _ in hits]
        return [c for c, _ in hits]
