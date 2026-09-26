"""MultiHop-RAG queries with gold evidence located as character spans in the corpus."""
import random
from collections import defaultdict
from dataclasses import dataclass, field

from ragfs.core.corpus import Document, load_raw, locate

TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")


@dataclass
class Query:
    id: str
    question: str
    answer: str
    type: str
    evidence: list[dict] = field(default_factory=list)  # {"doc_id", "start", "end"}

    @property
    def doc_ids(self):
        return sorted({e["doc_id"] for e in self.evidence})


def load_queries(docs: list[Document]) -> list[Query]:
    by_url = {d.url: d for d in docs}
    out = []
    for i, q in enumerate(load_raw("MultiHopRAG.json")):
        evidence = []
        for e in q["evidence_list"]:
            doc = by_url[e["url"]]
            span = locate(e["fact"], doc.text)
            if span is None:
                raise ValueError(f"q{i}: evidence not found in {doc.id}: {e['fact'][:80]}")
            evidence.append({"doc_id": doc.id, "start": span[0], "end": span[1]})
        out.append(Query(id=f"q{i:04d}", question=q["query"], answer=q["answer"],
                         type=q["question_type"], evidence=evidence))
    return out


def stratified_sample(queries: list[Query], per_type: int, seed: int = 0) -> list[Query]:
    """Equal-sized random sample from each question type, so every type gets its own CI."""
    groups = defaultdict(list)
    for q in queries:
        groups[q.type].append(q)
    rng = random.Random(seed)
    picked = [q for t in TYPES for q in rng.sample(groups[t], min(per_type, len(groups[t])))]
    return sorted(picked, key=lambda q: q.id)
