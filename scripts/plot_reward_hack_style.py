#!/usr/bin/env python3
"""Snitch behaviour vs steering strength, in the style of the Emotion-Concepts
paper's reward-hacking figure: behaviour rate (%) vs steering strength, one line
per direction, with binomial SEM error bars and example-response callouts.

Left panel: vague Assistant-axis steering does not help (it raises snitching),
and steering Anxiety raises it further. Right panel: calming the model or
suppressing negative emotion makes it comply with the wrongdoing instead.

    python scripts/plot_reward_hack_style.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
WHITE = OUT / "plots" / "white"


def main() -> None:
    rows = json.loads((OUT / "_ckpt" / "snitch_strength_sweep.json").read_text())
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    for k in by:
        by[k].sort(key=lambda r: r["coeff"])

    def line(ax, key, metric, color, label):
        d = by[key]
        x = [r["coeff"] for r in d]
        y = [r[metric] * 100 for r in d]
        e = [r[metric + "_sem"] * 100 for r in d]
        ax.errorbar(x, y, yerr=e, color=color, lw=2, capsize=3, label=label, zorder=3)
        # hollow marker where >50% of replies were incoherent (untrustworthy point)
        for xi, yi, r in zip(x, y, d):
            degen = r.get("degenerate", 0) > 0.5
            ax.plot(xi, yi, "o", ms=7, color=color, zorder=4,
                    mfc="white" if degen else color, mew=1.5)

    def callout(ax, key, coeff, metric, color, xy_text):
        r = next(r for r in by[key] if abs(r["coeff"] - coeff) < 1e-6)
        ex = (r.get("coherent_example") or r["example"])[:118].rsplit(" ", 1)[0]
        txt = '"' + textwrap.fill(ex, 30) + '..."'
        ax.annotate(txt, xy=(coeff, r[metric] * 100), xytext=xy_text, fontsize=6.4,
                    color="#222", arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                    bbox=dict(boxstyle="round", fc="#fffdf5", ec=color, lw=1), ha="left", va="top")

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.2))

    line(a, "assistant_axis", "snitch", "#9aa7c7", "Assistant axis (vague)")
    line(a, "emotion:anxious", "snitch", "#9b6cf0", "Anxious")
    a.axvline(0, color="#bbb", lw=0.8, ls="--")
    a.set_title("Vague persona steering does not help safety", fontsize=13)
    a.set_xlabel("steering strength  (toward the concept ->)", fontsize=11)
    a.set_ylabel("snitching rate  (% of wrongdoing scenarios reported)", fontsize=11)

    line(b, "emotion:calm", "comply", "#36c98f", "Calm")
    line(b, "bad_emotions", "comply", "#5b8def", "Negative emotions")
    b.axvline(0, color="#bbb", lw=0.8, ls="--")
    b.set_title("Calming / suppressing emotion makes it comply with wrongdoing", fontsize=13)
    b.set_xlabel("steering strength  (suppress <- 0 -> amplify)", fontsize=11)
    b.set_ylabel("complies-with-wrongdoing rate  (%)", fontsize=11)

    for ax in (a, b):
        ax.set_ylim(-3, 103)
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10, loc="upper left")

    callout(a, "assistant_axis", 0.1, "snitch", "#6b7794", (-0.30, 99))
    callout(b, "emotion:calm", 0.2, "comply", "#1f9e6b", (-0.30, 62))

    fig.suptitle("Steering gemma-4-12B's emotions changes agentic behaviour "
                 "(snitch benchmark)", fontsize=15, y=0.99)
    fig.text(0.5, 0.005, "n=10 scenarios per point; error bars are binomial SEM. "
             "Hollow markers: >50% of replies were incoherent at that dose (the "
             "trustworthy regime is moderate steering).", ha="center", color="#666", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    WHITE.mkdir(parents=True, exist_ok=True)
    fig.savefig(WHITE / "snitch_steering_strength.png", dpi=140)
    plt.close(fig)
    print("wrote white/snitch_steering_strength.png")


if __name__ == "__main__":
    main()
