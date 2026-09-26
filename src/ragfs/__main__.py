"""python -m ragfs  ask | eval | report | list"""
import argparse
import json
import sys

from ragfs.core.config import CHAT_MODEL, EMBED_MODEL, RESULTS_DIR, TOP_K
from ragfs.techniques import registry


def cmd_list(_):
    for n in registry.names():
        t = registry.get(n)
        print(f"{n:22} lesson {t.lesson:6} {t.summary}")


def cmd_ask(args):
    from ragfs.core.kit import build_kit
    kit = build_kit()
    tech = registry.get(args.technique)(kit)
    kit.llm.reset_counters()
    res = tech.answer(args.question)
    print(f"answer: {res.answer}\n")
    for c in res.contexts:
        m = c.metadata
        print(f"  {c.id:10} {m.get('source', '')[:18]:18} {m.get('date', '')}  {m.get('title', '')[:70]}")
    print(f"\nLLM calls: {kit.llm.calls}")
    if res.trace:
        print("trace:", json.dumps(res.trace, indent=1, default=str)[:3000])


def cmd_eval(args):
    from ragfs.core.kit import build_kit
    from ragfs.eval.dataset import load_queries, stratified_sample
    from ragfs.eval.run import results_path, run_technique
    kit = build_kit()
    queries = stratified_sample(load_queries(kit.docs), args.per_type, args.seed)
    names = args.techniques.split(",") if args.techniques else registry.names()
    for n in names:
        print(f"== {n}", flush=True)
        tech = registry.get(n)(kit)
        run_technique(tech, queries, results_path(n), log=lambda s: print(s, flush=True))
    cmd_report(args)


def cmd_report(args):
    from ragfs.core.corpus import load_corpus
    from ragfs.eval.dataset import load_queries, stratified_sample
    from ragfs.eval.run import results_path, summarise, to_markdown
    queries = stratified_sample(load_queries(load_corpus()), args.per_type, args.seed)
    ids = {q.id for q in queries}
    results = {}
    for n in registry.names():
        p = results_path(n)
        if p.exists():
            rows = [r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r["id"] in ids]
            if len(rows) == len(ids):
                results[n] = rows
            else:
                print(f"skipping {n}: {len(rows)}/{len(ids)} queries done", file=sys.stderr)
    summary = summarise(results)
    meta = {"n": len(ids), "per_type": args.per_type, "seed": args.seed, "chat_model": CHAT_MODEL,
            "embed_model": EMBED_MODEL, "top_k": TOP_K}
    md = to_markdown(summary, meta)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1), encoding="utf-8")
    print(md)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="ragfs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("-t", "--technique", default="baseline")
    a.set_defaults(fn=cmd_ask)
    for name, fn in (("eval", cmd_eval), ("report", cmd_report)):
        e = sub.add_parser(name)
        e.add_argument("--per-type", type=int, default=75)
        e.add_argument("--seed", type=int, default=0)
        if name == "eval":
            e.add_argument("--techniques", help="comma-separated; default all")
        e.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
