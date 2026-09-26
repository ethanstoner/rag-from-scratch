"""Everything a technique needs, built once and shared: corpus, chunks, indexes, models."""
from dataclasses import dataclass, field

from ragfs.core.bm25 import BM25
from ragfs.core.chunker import chunk_document
from ragfs.core.corpus import Document, load_corpus
from ragfs.core.store import VectorStore
from ragfs.core.types import Chunk


@dataclass
class Kit:
    docs: list[Document]
    chunks: list[Chunk]
    store: VectorStore
    bm25: BM25
    llm: object
    embedder: object
    judge: object = None
    extras: dict = field(default_factory=dict)  # lazily built indexes (summaries, RAPTOR tree, ...)

    def doc(self, doc_id: str) -> Document:
        return next(d for d in self.docs if d.id == doc_id)


def build_kit(llm=None, embedder=None, docs=None) -> Kit:
    if llm is None or embedder is None:
        from ragfs.core.embed import OllamaEmbedder
        from ragfs.core.llm import OllamaLLM
        llm = llm or OllamaLLM()
        embedder = embedder or OllamaEmbedder()
    docs = docs if docs is not None else load_corpus()
    chunks = [c for d in docs for c in chunk_document(d.id, d.text, d.metadata())]
    store = VectorStore(embedder)
    store.add(chunks)
    return Kit(docs=docs, chunks=chunks, store=store, bm25=BM25(chunks), llm=llm, embedder=embedder)
