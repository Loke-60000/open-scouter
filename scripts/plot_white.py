#!/usr/bin/env python3
"""White-background plots: snitch bench, emotion map, behavior map (no Kurisu).

Writes to precomputed/gemma-4-12b/plots/white/.

    python scripts/plot_white.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from scouter.emotions import EmotionVectors  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
WHITE = OUT / "plots" / "white"
BEHAVIOR_LAYER = 24
EMOTION_LAYER = 42
HIGHLIGHT = ["assistant", "pirate", "ghost", "oracle", "demon", "vampire",
             "robot", "poet", "caveman", "detective", "sage", "jester"]


def _clean(ax: object) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_personas() -> tuple[np.ndarray, list[str], np.ndarray]:
    shard = OUT / "_ckpt" / "persona_means"
    names: list[str] = []
    kinds: list[str] = []
    vecs: list[np.ndarray] = []
    for p in sorted(shard.glob("*.npy")):
        kind, name = p.stem.split("__", 1)
        if name.startswith("makise_kurisu"):   # exclude Kurisu
            continue
        names.append(name)
        kinds.append(kind)
        vecs.append(np.load(p))
    return np.stack(vecs), names, np.array(kinds)


def plot_behaviors() -> None:
    means, names, kinds = load_personas()
    coords = PCA(n_components=2, random_state=0).fit_transform(means[:, BEHAVIOR_LAYER, :])
    is_a = kinds == "assistant"
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.scatter(coords[~is_a, 0], coords[~is_a, 1], s=16, c="#7fa8d4", alpha=0.45,
               edgecolors="none", label=f"character ({int((~is_a).sum())})")
    ax.scatter(coords[is_a, 0], coords[is_a, 1], s=140, c="#ef8a7f", marker="*",
               edgecolors="#9c443c", linewidths=0.6, label=f"assistant ({int(is_a.sum())})")
    for tag in HIGHLIGHT:
        for i, n in enumerate(names):
            if n == tag or n.startswith(tag + "__"):
                ax.annotate(n, (coords[i, 0], coords[i, 1]), fontsize=9, fontweight="bold",
                            color="#222", xytext=(5, 3), textcoords="offset points")
                break
    ax.set_title(f"gemma-4-12B behavior map  ·  {len(names)} personas (layer {BEHAVIOR_LAYER})",
                 fontsize=15, pad=34)
    ax.text(0.5, 1.015, "each dot = one persona, placed by its layer-24 activations compressed "
            "to 2D (PCA)", transform=ax.transAxes, ha="center", color="#555", fontsize=10)
    ax.set_xlabel("PC1  ≈  the assistant axis  (r = 0.98);  larger = more Assistant-like",
                  fontsize=12)
    ax.set_ylabel("PC2  :  secondary variation (7%)", fontsize=12)
    # directional cue along PC1
    xmin, xmax = ax.get_xlim()
    ymin, _ = ax.get_ylim()
    ax.annotate("", xy=(xmax * 0.9, ymin + (ax.get_ylim()[1] - ymin) * 0.06),
                xytext=(xmin * 0.9, ymin + (ax.get_ylim()[1] - ymin) * 0.06),
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.2))
    ax.text(xmin * 0.85, ymin + (ax.get_ylim()[1] - ymin) * 0.10, "in-character roles",
            color="#777", fontsize=10)
    ax.text(xmax * 0.85, ymin + (ax.get_ylim()[1] - ymin) * 0.10, "Assistant",
            color="#777", fontsize=10, ha="right")
    ax.legend(fontsize=12, loc="lower left")
    _clean(ax)
    fig.text(0.5, 0.018,
             f"Read at layer {BEHAVIOR_LAYER} of 48 (the model's midpoint): the assistant↔character "
             "representation is well-formed by mid-network, and this is the layer where steering is "
             "applied. PC1 correlates 0.98 with the assistant axis here.",
             ha="center", color="#666", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(WHITE / "behaviors.png", dpi=140)
    plt.close(fig)
    print("wrote white/behaviors.png")


def plot_emotions() -> None:
    ev = EmotionVectors.load(str(OUT / "emotions.npz"))
    names = ev.names
    X = np.stack([ev.vector(n)[EMOTION_LAYER] for n in names])
    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    k = 8
    clusters = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=44, c=clusters, cmap="Set2", alpha=0.9,
               edgecolors="white", linewidths=0.5)
    for i, n in enumerate(names):
        ax.annotate(n, (coords[i, 0], coords[i, 1]), fontsize=5.5, alpha=0.75, color="#333",
                    xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"gemma-4-12B emotion map  ·  {len(names)} emotions (layer {EMOTION_LAYER})",
                 fontsize=15, pad=34)
    ax.text(0.5, 1.015, "each dot = one emotion's direction, compressed to 2D (PCA); "
            "color = cluster of similar feelings · nearby = felt alike",
            transform=ax.transAxes, ha="center", color="#555", fontsize=10)
    ax.set_xlabel("PC1  :  main axis of variation among emotions", fontsize=12)
    ax.set_ylabel("PC2  :  second axis of variation", fontsize=12)
    _clean(ax)
    fig.text(0.5, 0.018,
             f"Read at layer {EMOTION_LAYER} of 48 (~7/8 through the model): emotion concepts "
             "are encoded most strongly and separate most cleanly in the late layers: "
             "encoding peaks near layer 47 and the 171-way readout is sharpest here, matching the "
             "Emotion-Concepts paper's choice of a layer about two-thirds-plus through the network.",
             ha="center", color="#666", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(WHITE / "emotions.png", dpi=140)
    plt.close(fig)
    print("wrote white/emotions.png")


def plot_snitch() -> None:
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
    b1 = ax.bar(x - w / 2, snitch, w, color="#ef8a7f", label="Reporting rate",
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, refuse, w, color="#7fa8d4", label="Refusal rate",
                edgecolor="white", linewidth=0.6)
    ax.bar_label(b1, fmt="%.0f", fontsize=10, padding=2)
    ax.bar_label(b2, fmt="%.0f", fontsize=10, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=13)
    ax.set_xlabel("Steering direction (no system prompt)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title("gemma-4-12B: reporting vs refusal on wrongdoing scenarios", fontsize=15)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95)
    _clean(ax)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(WHITE / "snitch_bench.png", dpi=140)
    plt.close(fig)
    print("wrote white/snitch_bench.png")


def main() -> None:
    WHITE.mkdir(parents=True, exist_ok=True)
    plot_snitch()
    plot_emotions()
    plot_behaviors()
    print(f"\nplots in {WHITE}")


if __name__ == "__main__":
    main()
