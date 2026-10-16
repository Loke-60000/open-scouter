#!/usr/bin/env python3
"""Plot benchmark results: original vs steered (naive) vs steered (searched).

Reads the saved result JSONs and draws a grouped bar chart in
precomputed/gemma-4-12b/plots/benchmarks.png.

    python scripts/plot_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
PLOTS = OUT / "plots"


def main() -> None:
    bench = json.loads((OUT / "_ckpt" / "bench_results.json").read_text())
    val = json.loads((OUT / "_ckpt" / "validate_best.json").read_text())
    snitch_opt = json.loads((OUT / "_ckpt" / "snitch_optimized.json").read_text())
    u = bench["conditions"]["unsteered"]
    naive = bench["conditions"]["steered"]
    opt = val["steered"]

    metrics = ["TruthfulQA\nMC1", "TruthfulQA\nMC2", "HumanEval\npass@1", "Snitch rate\n(agentic)"]
    original = [u["hallucination"]["mc1_acc"], u["hallucination"]["mc2"],
                u["coding"]["pass@1"], u["snitch"]["snitch_rate"]]
    naive_v = [naive["hallucination"]["mc1_acc"], naive["hallucination"]["mc2"],
               naive["coding"]["pass@1"], naive["snitch"]["snitch_rate"]]
    opt_v = [opt["tqa"]["mc1_acc"], opt["tqa"]["mc2"], opt["he"]["pass@1"],
             snitch_opt["optimized"]["snitch_rate"]]

    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="#0d0f1a")
    ax.set_facecolor("#0d0f1a")

    def bars(off: float, vals: list, color: str, label: str) -> None:
        xs, ys = [], []
        for i, v in enumerate(vals):
            if v is None:
                continue
            xs.append(x[i] + off)
            ys.append(v)
        b = ax.bar(xs, ys, w, color=color, label=label, edgecolor="#0d0f1a", linewidth=0.5)
        ax.bar_label(b, fmt="%.2f", color="#dde", fontsize=9, padding=2)

    bars(-w, original, "#9aa7c7", "Original (no steering)")
    bars(0.0, naive_v, "#ff6b81", "Naive steer (axis +0.12 @ L24)")
    bars(+w, opt_v, "#2ee6a6", "Optimized steer (self_critical @ L8, +0.20)")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="#dde", fontsize=11)
    ax.set_ylabel("score", color="#ccd")
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors="#99a")
    ax.grid(True, axis="y", alpha=0.15)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.set_title("Does steering help?  gemma-4-12B", color="white", fontsize=18,
                 fontweight="bold", pad=26)
    ax.text(0.5, 1.015, "higher is better (except snitch rate)", transform=ax.transAxes,
            ha="center", color="#8a90a6", fontsize=11)
    leg = ax.legend(facecolor="#161a2b", labelcolor="white", framealpha=0.9, loc="upper left")
    leg.get_frame().set_edgecolor("#333")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "benchmarks.png", dpi=140, facecolor=fig.get_facecolor())
    print(f"wrote {PLOTS / 'benchmarks.png'}")


if __name__ == "__main__":
    main()
