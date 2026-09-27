import json

import pytest

import ragfs.core.kit
import ragfs.core.llm
from ragfs.__main__ import main
from ragfs.core.llm import OllamaLLM


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "an answer"}}


@pytest.fixture
def posts(monkeypatch):
    sent = []
    monkeypatch.setattr(ragfs.core.llm.httpx, "post", lambda url, json, timeout: sent.append(json) or FakeResponse())
    return sent


def test_cache_serves_repeats_without_calling_the_model(tmp_path, posts):
    llm = OllamaLLM(cache_dir=tmp_path)
    assert llm.complete("q") == llm.complete("q") == "an answer"
    assert len(posts) == 1


def test_uncached_client_calls_the_model_every_time(posts):
    llm = OllamaLLM(cache_dir=None)
    llm.complete("q")
    llm.complete("q")
    assert len(posts) == 2 and llm.real_seconds == llm.model_seconds


class Built(Exception):
    pass


@pytest.mark.parametrize("flag, cached", [([], True), (["--no-cache"], False)])
def test_eval_no_cache_builds_an_uncached_client(monkeypatch, flag, cached):
    seen = {}

    def fake_build_kit(llm=None, **kw):
        seen["llm"] = llm
        raise Built

    monkeypatch.setattr(ragfs.core.kit, "build_kit", fake_build_kit)
    with pytest.raises(Built):
        main(["eval", "--techniques", "baseline", *flag])
    if cached:
        assert seen["llm"] is None  # build_kit falls back to the default, cached client
    else:
        assert isinstance(seen["llm"], OllamaLLM) and seen["llm"].cache_dir is None


def test_report_reads_and_writes_results_dir(monkeypatch, tmp_path):
    import types

    import ragfs.core.corpus
    import ragfs.eval.dataset
    import ragfs.eval.run

    queries = [types.SimpleNamespace(id="a"), types.SimpleNamespace(id="b")]
    monkeypatch.setattr(ragfs.core.corpus, "load_corpus", lambda: [])
    monkeypatch.setattr(ragfs.eval.dataset, "load_queries", lambda docs: queries)
    monkeypatch.setattr(ragfs.eval.dataset, "stratified_sample", lambda qs, per_type, seed: qs)
    seen = {}

    def fake_summarise(results):
        seen["results"] = results
        return {}

    monkeypatch.setattr(ragfs.eval.run, "summarise", fake_summarise)
    monkeypatch.setattr(ragfs.eval.run, "to_markdown", lambda summary, meta: "# stub\n")
    rows = [{"id": "a"}, {"id": "b"}, {"id": "not-sampled"}]
    (tmp_path / "baseline.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

    main(["report", "--results-dir", str(tmp_path)])
    assert list(seen["results"]) == ["baseline"] and len(seen["results"]["baseline"]) == 2
    assert (tmp_path / "results.md").read_text(encoding="utf-8") == "# stub\n"
    assert (tmp_path / "summary.json").exists()
