"""Forest plot of paired deltas vs the baseline (answer EM and evidence recall)."""
import json

from ragfs.core.config import RESULTS_DIR

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#8a8985", "#e4e3df"
SIGNIFICANT = "#2a78d6"


def plot(out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    summary, meta = data["summary"], data["meta"]
    names = [n for n in summary if "em_vs_baseline" in summary[n]]
    names.sort(key=lambda n: summary[n]["em_vs_baseline"][0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(names) + 1.6), sharey=True, facecolor=SURFACE)
    panels = [("em_vs_baseline", "Answer exact match, Δ vs baseline"),
              ("recall_vs_baseline", "Evidence recall, Δ vs baseline")]
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for y, n in enumerate(names):
            m, lo, hi = summary[n][key]
            color = SIGNIFICANT if lo > 0 or hi < 0 else MUTED
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.axvline(0, color=INK, lw=1, zorder=1)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    axes[0].set_yticks(range(len(names)), names, color=INK, fontsize=9.5)
    fig.suptitle(f"{len(names)} RAG techniques vs dense top-{meta['top_k']} baseline on {meta['n']} MultiHop-RAG "
                 f"queries (95% paired bootstrap CI; blue = CI excludes 0)", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    out = out or RESULTS_DIR / "deltas.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out
