"""Lessons 10-11: route the question, or turn it into a structured metadata query.

The course routes between per-language doc indexes and builds filters over YouTube
metadata. Here the "indexes" are the corpus's news categories and the metadata is each
article's source and publication date, which MultiHop-RAG questions name explicitly
("...as reported by TechCrunch on October 7, 2023").
"""
from datetime import date, timedelta

import numpy as np

from ragfs.core.fusion import unique
from ragfs.techniques.base import Technique

CATEGORIES = ["business", "entertainment", "health", "science", "sports", "technology"]


class LogicalRouting(Technique):
    name = "routing_logical"
    lesson = "10"
    summary = "LLM routes the question to one or more category indexes"

    PROMPT = """You route questions about news articles to the right category indexes.
Categories: {cats}.
Pick every category that the articles needed to answer this question could belong to.

Question: {question}"""

    SCHEMA = {"type": "object", "required": ["categories"], "properties": {
        "categories": {"type": "array", "items": {"type": "string", "enum": CATEGORIES}}}}

    def route(self, question):
        try:
            cats = self.llm.json(self.PROMPT.format(cats=", ".join(CATEGORIES), question=question),
                                 self.SCHEMA)["categories"]
        except Exception as e:
            self.trace["route_error"] = str(e)
            cats = []
        cats = [c for c in cats if c in CATEGORIES]
        self.trace["categories"] = cats
        return set(cats)

    def retrieve(self, question):
        cats = self.route(question)
        if not cats:
            return self.dense(question)
        return self.dense(question, where=lambda c: c.metadata["category"] in cats)


class SemanticRouting(Technique):
    """Route by embedding similarity to each category's centroid, no LLM call."""
    name = "routing_semantic"
    lesson = "10"
    summary = "Route to category indexes by similarity to category centroids"
    margin = 0.02  # keep every category within this cosine of the best one

    def centroids(self):
        if "centroids" not in self.kit.extras:
            store = self.kit.store
            cats = np.array([c.metadata["category"] for c in store.chunks])
            cents = {}
            for cat in sorted(set(cats)):
                v = store.vectors[cats == cat].mean(axis=0)
                cents[cat] = v / np.linalg.norm(v)
            self.kit.extras["centroids"] = cents
        return self.kit.extras["centroids"]

    def retrieve(self, question):
        q = self.kit.embedder.embed_query(question)
        sims = {cat: float(v @ q) for cat, v in self.centroids().items()}
        best = max(sims.values())
        cats = {c for c, s in sims.items() if s >= best - self.margin}
        self.trace["categories"] = sorted(cats)
        return self.dense(question, where=lambda c: c.metadata["category"] in cats)


class QueryConstruction(Technique):
    """NL -> {source, date} mentions -> one filtered search per mention."""
    name = "query_construction"
    lesson = "11"
    summary = "LLM extracts source/date filters; one filtered search per mention"

    PROMPT = """Extract every news outlet the question mentions, with the publication date it gives for
that outlet's article, if any. Map each outlet to the closest name from this list:
{sources}

Return JSON: {{"mentions": [{{"source": "<name from list>", "date": "YYYY-MM-DD or null"}}]}}.
Return an empty list if no outlet is named. Articles are from September-December 2023.

Question: {question}"""

    def sources(self):
        return sorted({c.metadata["source"] for c in self.kit.chunks})

    def schema(self):
        return {"type": "object", "required": ["mentions"], "properties": {"mentions": {"type": "array", "items": {
            "type": "object", "required": ["source"], "properties": {
                "source": {"type": "string", "enum": self.sources()},
                "date": {"type": ["string", "null"]}}}}}}

    def extract(self, question):
        try:
            mentions = self.llm.json(self.PROMPT.format(sources="\n".join(self.sources()), question=question),
                                     self.schema())["mentions"]
        except Exception as e:
            self.trace["extract_error"] = str(e)
            mentions = []
        valid = set(self.sources())
        out = []
        for m in mentions:
            if m.get("source") in valid:
                out.append((m["source"], _parse_date(m.get("date"))))
        out = list(dict.fromkeys(out))
        self.trace["mentions"] = [(s, str(d) if d else None) for s, d in out]
        return out

    def retrieve(self, question):
        mentions = self.extract(question)
        if not mentions:
            return self.dense(question)
        per = max(2, self.k // len(mentions))
        picked = []
        for source, day in mentions:
            hits = []
            if day:
                window = {str(day + timedelta(days=d)) for d in (-1, 0, 1)}
                hits = self.dense(question, k=per, where=lambda c, s=source, w=window:
                                  c.metadata["source"] == s and c.metadata["date"] in w)
            if not hits:
                hits = self.dense(question, k=per, where=lambda c, s=source: c.metadata["source"] == s)
            picked.extend(hits)
        # Unfilled budget goes to plain dense retrieval.
        return unique(picked + self.dense(question))[:self.k]


def _parse_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None
