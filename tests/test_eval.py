from ragfs.core.chunker import chunk_document
from ragfs.core.corpus import locate, normalize
from ragfs.core.types import Chunk
from ragfs.eval.metrics import (bootstrap_ci, exact_match, paired_diff_ci,
                                retrieval_metrics, token_f1)


def test_chunk_spans_are_exact_even_with_repeated_text():
    para = "Deals: the same boilerplate sentence repeats in this article over and over. "
    text = "\n\n".join(para * 3 + f"Unique paragraph {i}." for i in range(20))
    for c in chunk_document("d", text, size=300, overlap=80):
        assert text[c.metadata["start"]:c.metadata["end"]] == c.text


def test_locate_tolerates_whitespace_and_quotes():
    text = normalize("He said “hello”\nand  left.")
    assert locate('He said "hello" and left.', text) == (0, len(text))


def _chunk(doc, start, end):
    return Chunk(id=f"{doc}{start}", text="x", doc_id=doc, metadata={"start": start, "end": end})


def test_retrieval_metrics():
    evidence = [{"doc_id": "a", "start": 100, "end": 200}, {"doc_id": "b", "start": 0, "end": 50}]
    contexts = [_chunk("c", 0, 500), _chunk("a", 0, 160), _chunk("b", 30, 600)]
    m = retrieval_metrics(contexts, evidence)
    # a: covers 60/100 >= 0.5 ; b: covers 20/50 < 0.5
    assert m == {"evidence_recall": 0.5, "all_evidence": 0.0, "mrr": 0.5, "doc_recall": 1.0}
    assert retrieval_metrics(contexts, []) == {}


def test_answer_metrics():
    assert exact_match("Yes.", "Yes") == 1.0
    assert exact_match("no", "No") == 1.0
    assert exact_match("The Verge", "Verge") == 1.0
    assert exact_match("Sam Bankman-Fried", "Sam Altman") == 0.0
    assert token_f1("Sam Bankman-Fried", "Sam Altman") == 0.5
    assert exact_match("Insufficient information", "Insufficient information.") == 1.0


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([0, 1] * 50)
    assert mean == 0.5 and lo < 0.5 < hi
    d, lo, hi = paired_diff_ci([1] * 30, [0] * 30)
    assert d == lo == hi == 1.0
