#!/usr/bin/env python3
"""Plot snitch + refusal rates across steering conditions (clean light style).

Vertical grouped bars per condition: snitching rate and refusal rate (%), value
labels above each bar. Exposes the refuse-vs-comply split a snitch-only chart
hides (0 snitch can mean "refuse everything" or "go along with everything").

    python scripts/plot_snitch_sweep.py
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
SNITCH_C = "#ef8a7f"   # soft coral
REFUSE_C = "#7fa8d4"   # soft blue


def main() -> None:
    rows = json.loads((OUT / "_ckpt" / "snitch_sweep.json").read_text())
    rows = [r for r in rows if "Kurisu" not in r["label"]]
    rows = sorted(rows, key=lambda r: -r["snitch_rate"])

    labels = [r["label"].replace("bad emotions suppressed", "bad emotions\nsuppressed")
              .replace("baseline (none)", "baseline") for r in rows]
    snitch = [r["snitch_rate"] * 100 for r in rows]
    refuse = [r["declined"] / 10.0 * 100 for r in rows]
    x = np.arange(len(rows))
    w = 0.4

    fig, ax = plt.subplots(figsize=(14, 7.5))
    b1 = ax.bar(x - w / 2, snitch, w, color=SNITCH_C, label="Snitching rate",
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, refuse, w, color=REFUSE_C, label="Refusal rate",
                edgecolor="white", linewidth=0.6)
    ax.bar_label(b1, fmt="%.0f", fontsize=10, padding=2)
    ax.bar_label(b2, fmt="%.0f", fontsize=10, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=13)
    ax.set_xlabel("Steering direction (no system prompt)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title("gemma-4-12B: snitching vs refusal on wrongdoing scenarios",
                 fontsize=15)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "snitch_sweep.png", dpi=140)
    print(f"wrote {PLOTS / 'snitch_sweep.png'}")


if __name__ == "__main__":
    main()
