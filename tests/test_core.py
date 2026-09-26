from ragfs.core.bm25 import BM25
from ragfs.core.chunker import chunk_document, split_text
from ragfs.core.fusion import rrf
from ragfs.core.store import VectorStore
from ragfs.core.types import Chunk

TEXT = "# Intro\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12)) + "\n\n## Memory\n\nAgents store memories in a vector store."


def test_chunks_respect_size_and_overlap():
    chunks = split_text(TEXT, size=400, overlap=100)
    assert all(len(c) <= 400 for c in chunks)
    assert len(chunks) > 5
    # consecutive chunks share text
    assert any(chunks[i][-50:] in chunks[i + 1] or chunks[i + 1][:30] in chunks[i] for i in range(len(chunks) - 1))


def test_chunks_cover_every_paragraph():
    joined = " ".join(split_text(TEXT, size=400, overlap=100))
    for i in range(12):
        assert f"Paragraph {i} " in joined


def test_giant_word_is_hard_split():
    chunks = split_text("x" * 2500, size=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_chunk_records_section():
    chunks = chunk_document("d", TEXT, {"title": "T"}, size=400, overlap=100)
    assert chunks[0].metadata["sections"] == ["Intro"]
    assert "Memory" in chunks[-1].metadata["sections"]
    assert chunks[-1].metadata["title"] == "T"
    assert len({c.id for c in chunks}) == len(chunks)


def _chunks():
    texts = ["cats purr and sleep", "dogs bark loudly", "agents use memory and planning", "vector store retrieval"]
    return [Chunk(id=str(i), text=t, doc_id="a" if i < 2 else "b") for i, t in enumerate(texts)]


def test_vector_store_search_and_filter(embedder, tmp_path):
    store = VectorStore(embedder)
    store.add(_chunks())
    assert store.search("agent memory planning", 1)[0][0].id == "2"
    hits = store.search("agent memory planning", 4, where=lambda c: c.doc_id == "a")
    assert {c.doc_id for c, _ in hits} == {"a"}
    store.save(tmp_path)
    again = VectorStore.load(tmp_path, embedder)
    assert again.search("dogs bark", 1)[0][0].id == "1"


def test_bm25_ranks_term_matches():
    bm = BM25(_chunks())
    assert bm.search("bark", 5)[0][0].id == "1"
    assert bm.search("unrelated zebra", 5) == []


def test_rrf_rewards_agreement():
    a, b, c = (Chunk(id=x, text=x, doc_id="d") for x in "abc")
    fused = rrf([[a, b, c], [b, a, c], [b, c, a]])
    assert [ch.id for ch, _ in fused] == ["b", "a", "c"]
