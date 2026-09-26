import pytest

from ragfs.core.corpus import Document
from ragfs.core.kit import build_kit
from ragfs.techniques import registry
from ragfs.techniques.base import REFUSAL
from tests.conftest import FakeLLM

ARTICLES = [
    ("TechCrunch", "technology", "2023-10-07", "Flexport chief Dave Clark resigned after clashing with founder Ryan Petersen."),
    ("The Verge", "technology", "2023-11-01", "Google shaped how the internet looks, critics of its search dominance say."),
    ("Sporting News", "sports", "2023-12-01", "Tyreek Hill leads the league in receiving yards for the Dolphins."),
    ("Fortune", "business", "2023-10-30", "Ryan Petersen returned as Flexport chief executive and reversed layoffs."),
]


def responder(prompt):
    if '"queries"' in prompt:
        return {"queries": ["Flexport leadership change", "Dave Clark Flexport resignation"]}
    if "Categories:" in prompt:
        return {"categories": ["technology"]}
    if '"mentions"' in prompt:
        return {"mentions": [{"source": "Fortune", "date": "2023-10-30"}, {"source": "Nonexistent", "date": None}]}
    if "Step-back:" in prompt:
        return "Who has led Flexport?"
    if "Passage:" in prompt:
        return "Dave Clark left Flexport as chief executive."
    return "Yes"


@pytest.fixture
def kit(embedder):
    docs = [Document(id=f"a{i}", title=f"t{i}", url=f"u{i}", source=s, category=c, author="x",
                     published_at=d + "T00:00:00", text=t) for i, (s, c, d, t) in enumerate(ARTICLES)]
    return build_kit(llm=FakeLLM(responder), embedder=embedder, docs=docs)


@pytest.mark.parametrize("name", [n for n in registry.names() if n not in registry.NEURAL])
def test_every_technique_answers_within_budget(kit, name):
    tech = registry.get(name)(kit, k=3)
    res = tech.answer("Did Flexport's leadership change after Dave Clark left, per TechCrunch and Fortune?")
    assert res.answer
    assert 0 < len(res.contexts) <= 3
    assert len({c.id for c in res.contexts}) == len(res.contexts)


def test_logical_routing_filters_category(kit):
    tech = registry.get("routing_logical")(kit, k=3)
    assert {c.metadata["category"] for c in tech.retrieve("Flexport news")} == {"technology"}


def test_query_construction_drops_unknown_sources_and_filters(kit):
    tech = registry.get("query_construction")(kit, k=3)
    ctx = tech.retrieve("What did Fortune report on October 30, 2023 about Flexport?")
    assert tech.trace["mentions"] == [("Fortune", "2023-10-30")]
    assert ctx[0].metadata["source"] == "Fortune"


def test_structured_output_failure_falls_back(embedder):
    docs = [Document(id="a0", title="t", url="u", source="S", category="sports", author="x",
                     published_at="2023-10-01", text="some text about sports")]
    kit = build_kit(llm=FakeLLM(lambda p: "not json" if "Categories" in p else REFUSAL), embedder=embedder, docs=docs)
    tech = registry.get("routing_logical")(kit, k=3)
    res = tech.answer("anything")
    assert "route_error" in tech.trace and res.contexts


def test_self_rag_rewrites_then_gives_up_when_nothing_is_relevant(kit):
    kit.llm.responder = lambda p: {"score": "no"} if '"score"' in p else "rewritten query"
    tech = registry.get("self_rag")(kit, k=3)
    res = tech.answer("unanswerable question")
    assert res.answer == REFUSAL
    assert [r["query"] for r in tech.trace["rounds"]] == ["unanswerable question", "rewritten query", "rewritten query"]


def test_crag_keeps_only_relevant_chunks(kit):
    kit.llm.responder = lambda p: ({"score": "yes" if "Tyreek" in p else "no"} if '"score"' in p
                                   else "Tyreek Hill receiving")
    tech = registry.get("crag")(kit, k=3)
    res = tech.answer("Who leads the league in receiving yards?")
    assert res.contexts and all("Tyreek" in c.text for c in res.contexts)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(registry.NEURAL))
def test_neural_techniques(kit, name):
    tech = registry.get(name)(kit, k=2)
    ctx = tech.retrieve("Which receiver leads the league for the Dolphins?")
    assert "Tyreek" in ctx[0].text
