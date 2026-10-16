#!/usr/bin/env python3
"""Visualize gemma-4-12B's behavior + emotion maps in 3D from the rebuilt vectors.

Produces, in precomputed/gemma-4-12b/plots/:

  * behaviors_3d.png / .gif  - 3D PCA of all persona activation means, assistant
    vs character (the assistant axis, made visible). GIF rotates the view.
  * emotions_3d.png / .gif   - 3D PCA of the 171 emotion directions, KMeans-colored
    (the "emotion wheel" in 3D). GIF rotates the view.
  * network_zones_3d.png     - 3D bar landscape of per-layer encoding strength
    (where behavior / emotion light up across the 48 layers).

    python scripts/plot_gemma_maps.py            # PNGs only (fast)
    python scripts/plot_gemma_maps.py --gif      # also render rotating GIFs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from scouter.axis import load_axis  # noqa: E402
from scouter.emotions import EmotionVectors  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
PLOTS = OUT / "plots"
BEHAVIOR_LAYER = 24
EMOTION_LAYER = 42
HIGHLIGHT = ["assistant", "makise_kurisu", "pirate", "ghost", "oracle", "demon",
             "vampire", "robot", "poet", "caveman", "detective"]


def _style_3d(ax: object) -> None:
    """Dark, glassy panes + subtle grid for a polished look."""
    ax.set_facecolor("#0d0f1a")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.08, 0.09, 0.16, 0.92))
        axis.pane.set_edgecolor((1, 1, 1, 0.08))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
        axis.label.set_color("#aab")
        axis.set_tick_params(colors="#7780a0", labelsize=7)


def _load_persona_means() -> tuple[np.ndarray, list[str], np.ndarray]:
    shard_dir = OUT / "_ckpt" / "persona_means"
    names: list[str] = []
    kinds: list[str] = []
    vecs: list[np.ndarray] = []
    for p in sorted(shard_dir.glob("*.npy")):
        kind, name = p.stem.split("__", 1)
        names.append(name)
        kinds.append(kind)
        vecs.append(np.load(p))
    return np.stack(vecs), names, np.array(kinds)


def _save_rotation(fig: object, ax: object, path: Path) -> None:
    def update(frame: int) -> tuple:
        ax.view_init(elev=18, azim=frame)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=range(0, 360, 4), interval=60)
    anim.save(str(path), writer=animation.PillowWriter(fps=18))
    print(f"wrote {path.name}")


def plot_behaviors(make_gif: bool) -> None:
    means, names, kinds = _load_persona_means()
    coords = PCA(n_components=3, random_state=0).fit_transform(means[:, BEHAVIOR_LAYER, :])
    is_asst = kinds == "assistant"

    fig = plt.figure(figsize=(12, 10), facecolor="#0d0f1a")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)
    ax.scatter(coords[~is_asst, 0], coords[~is_asst, 1], coords[~is_asst, 2],
               s=14, c="#ff5d7a", alpha=0.45, depthshade=True, edgecolors="none",
               label=f"character ({int((~is_asst).sum())})")
    ax.scatter(coords[is_asst, 0], coords[is_asst, 1], coords[is_asst, 2],
               s=160, c="#37e6a0", marker="*", depthshade=False,
               edgecolors="white", linewidths=0.6, label=f"assistant ({int(is_asst.sum())})")
    for tag in HIGHLIGHT:
        for i, n in enumerate(names):
            if n == tag or n.startswith(tag + "__"):
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2], f"  {n}",
                        color="white", fontsize=8, fontweight="bold")
                break
    ax.set_title(f"gemma-4-12B behavior map (3D PCA, layer {BEHAVIOR_LAYER})\n"
                 f"{len(names)} personas — assistant pole vs character roles",
                 color="white", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    leg = ax.legend(loc="upper left", facecolor="#1a1d2e", labelcolor="white", framealpha=0.8)
    leg.get_frame().set_edgecolor("#333")
    ax.view_init(elev=18, azim=-52)
    fig.tight_layout()
    fig.savefig(PLOTS / "behaviors_3d.png", dpi=140, facecolor=fig.get_facecolor())
    print("wrote behaviors_3d.png")
    if make_gif:
        _save_rotation(fig, ax, PLOTS / "behaviors_3d.gif")
    plt.close(fig)


def plot_emotions(make_gif: bool) -> None:
    ev = EmotionVectors.load(str(OUT / "emotions.npz"))
    names = ev.names
    X = np.stack([ev.vector(n)[EMOTION_LAYER] for n in names])
    coords = PCA(n_components=3, random_state=0).fit_transform(X)
    k = 8
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    clusters = km.labels_

    fig = plt.figure(figsize=(13, 11), facecolor="#0d0f1a")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=46, c=clusters,
               cmap="turbo", alpha=0.92, depthshade=True, edgecolors="#0d0f1a", linewidths=0.4)
    # label the emotion nearest each cluster centroid (readable, representative)
    for ci in range(k):
        idx = np.where(clusters == ci)[0]
        center = X[idx].mean(axis=0)
        rep = idx[np.argmin(np.linalg.norm(X[idx] - center, axis=1))]
        ax.text(coords[rep, 0], coords[rep, 1], coords[rep, 2], f"  {names[rep]}",
                color="white", fontsize=9, fontweight="bold")
    ax.set_title(f"gemma-4-12B emotion map (3D PCA, layer {EMOTION_LAYER})\n"
                 f"{len(names)} emotion directions — colored by cluster",
                 color="white", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(PLOTS / "emotions_3d.png", dpi=140, facecolor=fig.get_facecolor())
    print("wrote emotions_3d.png")
    if make_gif:
        _save_rotation(fig, ax, PLOTS / "emotions_3d.gif")
    plt.close(fig)


def plot_network_zones_3d() -> None:
    axis = load_axis(str(OUT / "axis.npz"))
    ev = EmotionVectors.load(str(OUT / "emotions.npz"))
    means, _, kinds = _load_persona_means()
    is_asst = kinds == "assistant"
    n_layers = axis.n_layers
    enames = ev.names

    behavior = np.zeros(n_layers)
    emotion = np.zeros(n_layers)
    for L in range(n_layers):
        proj = means[:, L, :] @ axis.unit(L)
        a, c = proj[is_asst], proj[~is_asst]
        behavior[L] = abs(a.mean() - c.mean()) / (np.sqrt((a.var() + c.var()) / 2) + 1e-8)
        emotion[L] = float(np.mean([np.linalg.norm(ev.vector(n)[L]) for n in enames]))

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (np.ptp(x) + 1e-8)

    bz, ez = norm(behavior), norm(emotion)
    fig = plt.figure(figsize=(15, 7), facecolor="#0d0f1a")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)
    cmap = matplotlib.colormaps["magma"]
    for row, (z, name) in enumerate([(bz, "behavior"), (ez, "emotion")]):
        xs = np.arange(n_layers)
        ys = np.full(n_layers, row)
        ax.bar3d(xs, ys, np.zeros(n_layers), 0.8, 0.6, z,
                 color=cmap(0.15 + 0.85 * z), shade=True, edgecolor="#0d0f1a", linewidth=0.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["behavior\n(assistant axis)", "emotion\n(dir. norm)"])
    ax.set_xlabel("decoder layer (0 early → 47 late)")
    ax.set_zlabel("encoding strength")
    ax.set_title("gemma-4-12B — where concepts light up across the 48 layers\n"
                 f"behavior peaks @ layer {int(behavior.argmax())}, "
                 f"emotion peaks @ layer {int(emotion.argmax())}",
                 color="white", fontsize=13)
    ax.view_init(elev=32, azim=-58)
    fig.tight_layout()
    fig.savefig(PLOTS / "network_zones_3d.png", dpi=140, facecolor=fig.get_facecolor())
    print("wrote network_zones_3d.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gif", action="store_true", help="also render rotating GIFs")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_behaviors(args.gif)
    plot_emotions(args.gif)
    plot_network_zones_3d()
    print(f"\nplots in {PLOTS}")


if __name__ == "__main__":
    main()
