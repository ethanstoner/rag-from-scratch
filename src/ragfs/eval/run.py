"""Run techniques over the query sample, one resumable jsonl per technique, then summarise.

Latency is reported as `latency_s` = wall time with cached LLM calls credited back at
the duration they took when first made, so a resumed or cached run reports the same
cost as a cold one.
"""
import json
import time
from pathlib import Path
from statistics import median

from ragfs.core.config import RESULTS_DIR
from ragfs.eval.dataset import TYPES, Query
from ragfs.eval.metrics import (bootstrap_ci, exact_match, paired_diff_ci,
                                retrieval_metrics, token_f1)
from ragfs.techniques.base import REFUSAL, is_refusal


def run_one(tech, q: Query) -> dict:
    llm = tech.llm
    llm.reset_counters()
    t0 = time.perf_counter()
    try:
        res = tech.answer(q.question)
        error = None
    except Exception as e:  # a technique crash scores as a wrong, empty answer
        res, error = None, f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    answer = res.answer if res else ""
    contexts = res.contexts if res else []
    gold = q.answer
    row = {"id": q.id, "type": q.type, "technique": tech.name, "question": q.question,
           "gold": gold, "answer": answer,
           "em": exact_match(answer, gold), "f1": token_f1(answer, gold),
           "refused": is_refusal(answer),
           "llm_calls": llm.calls,
           "latency_s": wall - llm.real_seconds + llm.model_seconds,
           "n_contexts": len(contexts), "context_chars": sum(len(c.text) for c in contexts),
           "context_ids": [c.id for c in contexts],
           "trace": res.trace if res else {}, "error": error}
    row.update(retrieval_metrics(contexts, q.evidence))
    return row


def run_technique(tech, queries: list[Query], out: Path, log=print) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    done = {r["id"] for r in rows}
    todo = [q for q in queries if q.id not in done]
    if todo:
        tech.llm.reset_counters()
        t0 = time.perf_counter()
        tech.prepare()
        wall = time.perf_counter() - t0
        index_cost = {"llm_calls": tech.llm.calls,
                      "seconds": wall - tech.llm.real_seconds + tech.llm.model_seconds}
        if index_cost["llm_calls"] or index_cost["seconds"] > 1:
            out.with_suffix(".index.json").write_text(json.dumps(index_cost), encoding="utf-8")
            log(f"  {tech.name}: index built with {index_cost['llm_calls']} LLM calls in {index_cost['seconds']:.0f}s")
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            row = run_one(tech, q)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                em = sum(r["em"] for r in rows) / len(rows)
                log(f"  {tech.name}: {len(rows)}/{len(queries)}  EM so far {em:.3f}  "
                    f"({time.perf_counter() - t0:.0f}s this session)")
    wanted = {q.id for q in queries}
    return [r for r in rows if r["id"] in wanted]


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else float("nan")


def summarise(results: dict[str, list[dict]], baseline: str = "baseline") -> dict:
    """Per-technique metrics with bootstrap CIs, and paired EM deltas against the baseline."""
    base = {r["id"]: r for r in results.get(baseline, [])}
    out = {}
    for name, rows in results.items():
        answerable = [r for r in rows if r["type"] != "null_query"]
        nulls = [r for r in rows if r["type"] == "null_query"]
        s = {"n": len(rows),
             "em": bootstrap_ci([r["em"] for r in rows]),
             "em_by_type": {t: _mean([r for r in rows if r["type"] == t], "em") for t in TYPES},
             "f1": _mean(rows, "f1"),
             "false_refusal": _mean([{"x": float(r["refused"])} for r in answerable], "x"),
             "null_refusal": _mean([{"x": float(r["refused"])} for r in nulls], "x"),
             "evidence_recall": bootstrap_ci([r["evidence_recall"] for r in answerable]),
             "all_evidence": _mean(answerable, "all_evidence"),
             "mrr": _mean(answerable, "mrr"),
             "doc_recall": _mean(answerable, "doc_recall"),
             "llm_calls": _mean(rows, "llm_calls"),
             "latency_median_s": median(r["latency_s"] for r in rows) if rows else float("nan"),
             "context_chars": _mean(rows, "context_chars"),
             "errors": sum(1 for r in rows if r.get("error"))}
        paired = [(r["em"], base[r["id"]]["em"]) for r in rows if r["id"] in base]
        if paired and name != baseline:
            s["em_vs_baseline"] = paired_diff_ci([a for a, _ in paired], [b for _, b in paired])
            s["recall_vs_baseline"] = paired_diff_ci(
                [r["evidence_recall"] for r in answerable if r["id"] in base],
                [base[r["id"]]["evidence_recall"] for r in answerable if r["id"] in base])
        out[name] = s
    return out


def _ci(t):
    m, lo, hi = t
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _delta(t):
    if not t:
        return "—"
    m, lo, hi = t
    sig = "**" if lo > 0 or hi < 0 else ""
    return f"{sig}{m:+.3f}{sig} [{lo:+.3f}, {hi:+.3f}]"


def to_markdown(summary: dict, meta: dict) -> str:
    order = sorted(summary, key=lambda n: -summary[n]["em"][0])
    lines = [f"# Results\n",
             f"{meta.get('n')} MultiHop-RAG queries ({meta.get('per_type')} per type, sample seed "
             f"{meta.get('seed')}), generator `{meta.get('chat_model')}`, embeddings `{meta.get('embed_model')}`, "
             f"temperature 0, top-k {meta.get('top_k')}. 95% bootstrap CIs; deltas are paired against "
             f"`baseline` and bold where the CI excludes zero.\n",
             "## Answer quality\n",
             "| technique | EM | EM Δ vs baseline | inference | comparison | temporal | null (refusal) | false refusals | LLM calls/q | median latency |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        s = summary[n]
        bt = s["em_by_type"]
        lines.append(f"| {n} | {_ci(s['em'])} | {_delta(s.get('em_vs_baseline'))} | {bt['inference_query']:.2f} | "
                     f"{bt['comparison_query']:.2f} | {bt['temporal_query']:.2f} | {bt['null_query']:.2f} | "
                     f"{s['false_refusal']:.2f} | {s['llm_calls']:.1f} | {s['latency_median_s']:.1f}s |")
    lines += ["\n## Retrieval (answerable queries)\n",
              "| technique | evidence recall | Δ vs baseline | all evidence found | MRR | doc recall | context chars |",
              "|---|---|---|---|---|---|---|"]
    for n in sorted(summary, key=lambda n: -summary[n]["evidence_recall"][0]):
        s = summary[n]
        lines.append(f"| {n} | {_ci(s['evidence_recall'])} | {_delta(s.get('recall_vs_baseline'))} | "
                     f"{s['all_evidence']:.3f} | {s['mrr']:.3f} | {s['doc_recall']:.3f} | {s['context_chars']:.0f} |")
    errs = {n: s["errors"] for n, s in summary.items() if s["errors"]}
    if errs:
        lines.append(f"\nTechnique errors (scored as wrong): {errs}")
    lines.append(f"\nRefusal string: `{REFUSAL}`")
    return "\n".join(lines) + "\n"


def results_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.jsonl"
