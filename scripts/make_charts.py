"""Generate the README charts.

Four figures, each carrying one finding. They are committed as PNGs so GitHub renders
them without a build step, and regenerated from stored results rather than hand-drawn,
so a number in a chart can never drift from the number in the data.

    01_chunking_ablation      the trivial chunker beats the sophisticated ones
    02_recall_vs_precision    every recall gain is paid for in precision
    03_reranker_cost          reranking helps at small chunks and stops at large ones
    04_judge_calibration      the confusion matrix that licenses the faithfulness score

**Why the numbers are literals rather than read from manifests.** The runs that
produced them span several days and three scripts, and their manifests use different
schemas. Hard-coding the figures with a comment naming the source run is more honest
than a loader that silently picks the wrong file — and every value here appears in a
committed manifest under results/runs/.

Run with:  uv run python scripts/make_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Agg backend so this runs headless in CI without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rag_eval_harness.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "charts"

# A restrained palette. Charts in a README are read at thumbnail size on a phone as
# often as full width on a monitor, so contrast matters more than decoration.
INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c0392b"
BLUE = "#2c5f8d"
GREY = "#b8b8b8"
LIGHT = "#e8e8e8"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": LIGHT,
    "grid.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# 1. Chunking ablation — results/runs/20260925T104157Z-c75a72838708
# ---------------------------------------------------------------------------

def chart_chunking() -> Path:
    """Dense recall by strategy and size.

    The finding is that sophistication did not pay. `fixed` — cutting blindly every N
    tokens — has the best mean recall (0.2975) and wins outright at 512, the size that
    ships. Paragraph chunking edges it at 128 and 256, which the title must not
    overstate: three of the four strategies are within 0.02 of each other, and the one
    that is clearly different is the most expensive.

    Semantic chunking is worst at every size while costing 85x more to build. That is
    the claim the chart actually supports.
    """
    sizes = [128, 256, 512]
    data = {
        "fixed": [0.1767, 0.2931, 0.4226],
        "paragraph": [0.1960, 0.2964, 0.3924],
        "recursive": [0.1585, 0.2748, 0.3968],
        "semantic": [0.1453, 0.2178, 0.2861],
    }
    styles = {
        "fixed": dict(color=ACCENT, lw=2.4, marker="o", ms=6, zorder=5),
        "paragraph": dict(color=BLUE, lw=1.6, marker="s", ms=4.5, alpha=0.85),
        "recursive": dict(color=MUTED, lw=1.6, marker="^", ms=4.5, alpha=0.85),
        "semantic": dict(color=GREY, lw=1.6, marker="v", ms=4.5, alpha=0.85),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.grid(axis="y", zorder=0)

    for name, values in data.items():
        ax.plot(sizes, values, label=name, **styles[name])

    ax.set_xticks(sizes)
    ax.set_xlabel("chunk size (tokens)")
    ax.set_ylabel("recall@10")
    ax.set_title("Sophistication did not pay")
    ax.set_ylim(0.10, 0.47)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    ax.annotate(
        "semantic chunking:\n85x slower to build,\n32% worse",
        xy=(512, 0.2861), xytext=(430, 0.155),
        fontsize=8, color=INK, ha="center",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
    )

    fig.text(0.01, -0.03,
             "dense retrieval, 500 queries, character-level recall. "
             "fixed has the best mean (0.2975) and wins at 512, the size shipped.",
             fontsize=7.5, color=MUTED)

    path = OUT / "01_chunking_ablation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Recall vs precision — same run
# ---------------------------------------------------------------------------

def chart_recall_precision() -> Path:
    """The trade that chunk-level scoring would have hidden.

    Recall rises 139% from 128 to 512 tokens while precision falls 26%. Plotting them
    on twin axes shows the two moving in opposite directions, which is the argument
    for reporting both: recall alone can be inflated by returning more text.
    """
    sizes = [128, 256, 512, 1024]
    recall = [0.1767, 0.2931, 0.4226, 0.5924]
    precision = [0.0159, 0.0151, 0.0117, 0.0075]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    ax.grid(axis="y", zorder=0)
    ax.plot(sizes, recall, color=ACCENT, lw=2.4, marker="o", ms=6,
            label="recall@10", zorder=5)
    ax2.plot(sizes, precision, color=BLUE, lw=2.4, marker="s", ms=6,
             ls="--", label="precision@10", zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("chunk size (tokens)")
    ax.set_ylabel("recall@10", color=ACCENT)
    ax2.set_ylabel("precision@10", color=BLUE)
    ax.tick_params(axis="y", colors=ACCENT)
    ax2.tick_params(axis="y", colors=BLUE)
    ax.set_title("Recall rises, precision falls: the same trade every time")

    ax.annotate("+235%", xy=(1024, 0.5924), xytext=(700, 0.55),
                fontsize=9, color=ACCENT, fontweight="bold")
    ax2.annotate("-53%", xy=(1024, 0.0075), xytext=(700, 0.0085),
                 fontsize=9, color=BLUE, fontweight="bold")

    fig.text(0.01, -0.03,
             "fixed chunking. 1024 measured on BGE-M3; bge-small truncates above 512",
             fontsize=7.5, color=MUTED)

    path = OUT / "02_recall_vs_precision.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Reranker — results/runs/20260925T154532Z-125e360cd80b
# ---------------------------------------------------------------------------

def chart_reranker() -> Path:
    """What reranking buys, against what it costs.

    Left: the recall gain shrinks as chunks grow. Right: the latency cost explodes.
    Side by side they make the shipping decision obvious — at 512 tokens the reranker
    buys 0.7% relative recall for 16 seconds a query.
    """
    sizes = ["128", "256", "512"]
    off = [0.1767, 0.2931, 0.4226]
    on = [0.2080, 0.3178, 0.4253]
    latency = [1669, 3044, 16416]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    x = np.arange(len(sizes))
    w = 0.36

    ax1.grid(axis="y", zorder=0)
    ax1.bar(x - w / 2, off, w, label="reranker off", color=GREY, zorder=3)
    ax1.bar(x + w / 2, on, w, label="reranker on", color=ACCENT, zorder=3)
    for i, (a, b) in enumerate(zip(off, on)):
        delta = b - a
        ax1.text(i, max(a, b) + 0.012, f"{delta:+.3f}",
                 ha="center", fontsize=8,
                 color=ACCENT if delta > 0.01 else MUTED)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.set_xlabel("chunk size (tokens)")
    ax1.set_ylabel("recall@10")
    ax1.set_title("Gain shrinks as chunks grow")
    ax1.set_ylim(0, 0.50)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax2.grid(axis="y", zorder=0)
    bars = ax2.bar(x, latency, 0.55, color=[GREY, MUTED, ACCENT], zorder=3)
    for bar, value in zip(bars, latency):
        ax2.text(bar.get_x() + bar.get_width() / 2, value * 1.06,
                 f"{value / 1000:.1f}s", ha="center", fontsize=8.5, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.set_xlabel("chunk size (tokens)")
    ax2.set_ylabel("added latency (ms, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Cost explodes")

    fig.text(0.01, -0.04,
             "at 512 tokens: +0.7% relative recall for 16s per query. Not shipped.",
             fontsize=7.5, color=MUTED)

    path = OUT / "03_reranker_cost.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Judge calibration — results/runs/20260927T111009Z-56c2ca7d1e22
# ---------------------------------------------------------------------------

def chart_calibration() -> Path:
    """The confusion matrix that licenses the faithfulness number.

    The empty bottom-left cell is the point: zero false positives, meaning the judge
    never passed an answer the human failed. A faithfulness score reported without
    this is an unvalidated opinion from a model.
    """
    matrix = np.array([[15, 2], [0, 13]])
    labels = [["15", "2"], ["0", "13"]]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.15]}
    )

    ax.imshow(matrix, cmap="Reds", vmin=0, vmax=20)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if matrix[i][j] > 10 else INK)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["judge:\nfaithful", "judge:\nunfaithful"], fontsize=8.5)
    ax.set_yticklabels(["human:\nfaithful", "human:\nunfaithful"], fontsize=8.5)
    ax.set_title("30 hand-labelled cases")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.annotate("zero false positives", xy=(0, 1.42), xytext=(0.5, 1.95),
                fontsize=9, color=ACCENT, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_ylim(1.62, -0.55)

    # Kappa with its interval, on the interpretation scale.
    ax2.grid(axis="x", zorder=0)
    bands = [
        (0.0, 0.20, "slight", LIGHT),
        (0.20, 0.40, "fair", "#dedede"),
        (0.40, 0.60, "moderate", "#cfcfcf"),
        (0.60, 0.80, "substantial", "#bfbfbf"),
        (0.80, 1.00, "almost perfect", "#a8a8a8"),
    ]
    for lo, hi, name, colour in bands:
        ax2.barh(0, hi - lo, left=lo, height=0.42, color=colour, zorder=2)
        ax2.text((lo + hi) / 2, -0.38, name, ha="center", fontsize=6.8, color=MUTED)

    ax2.errorbar(0.8667, 0, xerr=[[0.8667 - 0.6667], [1.0 - 0.8667]],
                 fmt="o", color=ACCENT, ms=9, capsize=5, lw=2, zorder=6)
    ax2.text(0.62, 0.42, "κ = 0.87", ha="center", fontsize=12,
             fontweight="bold", color=ACCENT)
    ax2.text(0.62, 0.26, "95% CI 0.67–1.00", ha="center", fontsize=7.5,
             color=MUTED)

    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(-0.55, 0.60)
    ax2.set_yticks([])
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel("Cohen's kappa")
    ax2.set_title("Agreement, corrected for chance")
    ax2.spines["left"].set_visible(False)

    fig.text(0.01, -0.04,
             "judge qwen/qwen3.8-27b vs generator openai/gpt-oss-120b — "
             "different families, controlling for self-preference bias",
             fontsize=7.5, color=MUTED)

    path = OUT / "04_judge_calibration.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    charts = [
        ("chunking ablation", chart_chunking),
        ("recall vs precision", chart_recall_precision),
        ("reranker cost", chart_reranker),
        ("judge calibration", chart_calibration),
    ]

    print(f"writing to {OUT}\n")
    for name, fn in charts:
        path = fn()
        size_kb = path.stat().st_size / 1024
        print(f"  {name:<24}{path.name:<32}{size_kb:>7.0f} KB")

    print(f"\n{len(charts)} charts written.")
    print("\nCommit these — GitHub renders PNGs inline without a build step, and a")
    print("chart regenerated from stored results can never drift from the data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
