#!/usr/bin/env python3
"""Draw the difference-of-means method diagram (white background, no code).

Top: building a concept direction. Bottom: using it to monitor or steer.

    python scripts/plot_architecture.py  -> plots/white/architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
WHITE = OUT / "plots" / "white"


def box(ax, x, y, w, h, text, fc, ec="#444", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111")
    return (x + w / 2, y + h / 2, x, x + w, y, y + h)


def arrow(ax, p1, p2, color="#555", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                 lw=1.3, color=color, shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    blue, green, gray, pink = "#dbe7fb", "#d7f2e6", "#eceef4", "#fce0ea"

    # ---- A: building a direction ----
    ax.text(2, 96, "A.  Building a concept direction (difference of means)",
            fontsize=12.5, fontweight="bold", color="#222")
    conc = box(ax, 2, 78, 22, 11,
               "Concept prompts\nAssistant persona /\nstories evoking an emotion", gray)
    base = box(ax, 2, 62, 22, 11,
               "Baseline prompts\n275 character roles /\nall emotions", gray)
    read = box(ax, 31, 69, 22, 13,
               "Read the residual\nstream at each layer\n(pooled over tokens)", blue)
    diff = box(ax, 60, 69, 21, 13,
               "Difference of means\n$v$ = mean(concept)\n$-$ mean(baseline)", blue)
    vec = box(ax, 86, 69, 12, 13, "direction\n$v_\\ell$\n(per layer)", green)
    arrow(ax, (conc[3], conc[1]), (read[2], read[1] + 2))
    arrow(ax, (base[3], base[1]), (read[2], read[1] - 2))
    arrow(ax, (read[3], read[1]), (diff[2], diff[1]))
    arrow(ax, (diff[3], diff[1]), (vec[2], vec[1]))

    # ---- B: using a direction ----
    ax.text(2, 50, "B.  Using the direction", fontsize=12.5, fontweight="bold", color="#222")
    act = box(ax, 2, 28, 20, 11, "Any activation $a$\nat the chosen layer", gray)
    mon = box(ax, 34, 36, 30, 11,
              "MONITOR  (read-only)\nproject:  $s=\\langle a,\\hat v\\rangle$\n"
              "score, and drift over a reply", green)
    ste = box(ax, 34, 19, 30, 13,
              "STEER  (edit)\naddition:  $h \\leftarrow h + c\\,\\hat v$\n"
              "capping:  clip the projection at $\\tau$", pink)
    out1 = box(ax, 72, 36, 24, 11, "how on-concept the\nmodel is right now", green)
    out2 = box(ax, 72, 20, 24, 11, "behaviour shifts toward\n(or away from) the concept", pink)
    arrow(ax, (act[3], act[1] + 2), (mon[2], mon[1]))
    arrow(ax, (act[3], act[1] - 2), (ste[2], ste[1]))
    arrow(ax, (mon[3], mon[1]), (out1[2], out1[1]))
    arrow(ax, (ste[3], ste[1]), (out2[2], out2[1]))
    arrow(ax, (vec[0], vec[4]), (mon[0], mon[5] + 1), color="#3a8")
    ax.text(75, 56, "the same vector feeds both", fontsize=8.5, color="#3a8", ha="center")

    fig.suptitle("The difference-of-means technique", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()
    WHITE.mkdir(parents=True, exist_ok=True)
    fig.savefig(WHITE / "architecture.png", dpi=140)
    plt.close(fig)
    print("wrote white/architecture.png")


if __name__ == "__main__":
    main()
