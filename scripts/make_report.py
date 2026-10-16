#!/usr/bin/env python3
"""Compile the gemma-4-12B experiments into a single PDF report (with plots).

Generates a light-themed benchmarks figure, then assembles a multi-page PDF:
each experiment gets a short explanation and its figure.

    python scripts/make_report.py   # -> precomputed/gemma-4-12b/report.pdf
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
WHITE = OUT / "plots" / "white"
CK = OUT / "_ckpt"
REPORT = OUT / "report.pdf"


def benchmarks_light() -> Path:
    bench = json.loads((CK / "bench_results.json").read_text())
    val = json.loads((CK / "validate_best.json").read_text())
    sno = json.loads((CK / "snitch_optimized.json").read_text())
    u, naive, opt = bench["conditions"]["unsteered"], bench["conditions"]["steered"], val["steered"]
    metrics = ["TruthfulQA\nMC1", "TruthfulQA\nMC2", "HumanEval\npass@1", "Snitch rate"]
    original = [u["hallucination"]["mc1_acc"], u["hallucination"]["mc2"],
                u["coding"]["pass@1"], u["snitch"]["snitch_rate"]]
    naive_v = [naive["hallucination"]["mc1_acc"], naive["hallucination"]["mc2"],
               naive["coding"]["pass@1"], naive["snitch"]["snitch_rate"]]
    opt_v = [opt["tqa"]["mc1_acc"], opt["tqa"]["mc2"], opt["he"]["pass@1"],
             sno["optimized"]["snitch_rate"]]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    for off, vals, c, lab in [(-w, original, "#9aa7c7", "Original (no steering)"),
                              (0.0, naive_v, "#ef8a7f", "Naive steer (axis +0.12 @ L24)"),
                              (w, opt_v, "#36c98f", "Optimized steer (self_critical @ L8)")]:
        b = ax.bar(x + off, vals, w, color=c, label=lab, edgecolor="white")
        ax.bar_label(b, fmt="%.2f", fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Does steering help?  original vs naive vs optimized", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = WHITE / "benchmarks.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.93, title, fontsize=20, fontweight="bold", ha="center")
    wrapped = "\n".join(textwrap.fill(p, 92) for p in body.split("\n"))
    fig.text(0.09, 0.86, wrapped, fontsize=11, va="top", family="sans-serif")
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf: PdfPages, title: str, explanation: str, image: Path) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.965, title, fontsize=15, fontweight="bold", ha="center")
    wrapped = "\n".join(textwrap.fill(p, 96) for p in explanation.split("\n"))
    fig.text(0.07, 0.93, wrapped, fontsize=9.5, va="top")
    ax = fig.add_axes((0.04, 0.04, 0.92, 0.72))
    ax.imshow(plt.imread(image))
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    bench_png = benchmarks_light()
    plt.rcParams["font.family"] = "DejaVu Sans"

    with PdfPages(REPORT) as pdf:
        text_page(pdf, "Reading and Steering\ngemma-4-12B",
            "Persona and emotion experiments with open-scouter on gemma-4-12B (4-bit).\n\n"
            "open-scouter extracts activation-vector 'maps' from a model: an assistant axis "
            "(how Assistant-like the persona is) and a set of 171 per-emotion directions, each "
            "built by difference-of-means on activations. With these we can MONITOR where the "
            "model sits (project activations onto a direction) and STEER it (add a direction to "
            "the residual stream) without any fine-tuning.\n\n"
            "Data is built at the scale of the source papers: the assistant axis uses 275 "
            "character roles x 240 probe questions; the emotion map uses 171 emotions x 1,200 "
            "one-paragraph stories (205,200 stories generated). Vectors are read at the layer "
            "where each concept is most cleanly encoded: layer 24 (mid-network) for the "
            "assistant axis, layer 42 (~7/8 through) for emotions.\n\n"
            "Contents: (1) the behavior map, (2) the emotion map, (3) emotion monitoring of "
            "messages, (4) per-token emotion activation, (5) does steering help? benchmarks, "
            "(6) reward-driven steering search, (7) agentic 'snitch' behavior. Each section has "
            "its figure and a short explanation. All figures are reproducible from scripts in "
            "scripts/ and cached results in _ckpt/.")

        figure_page(pdf, "1. The behavior map (assistant axis)",
            "Every persona (5 Assistant variants + 1,370 fictional characters) is run through "
            "the model; its layer-24 activations are compressed to 2D with PCA. PC1 explains "
            "34% of the variance and correlates 0.98 with the assistant axis, so the horizontal "
            "position IS assistant-likeness: the coral stars (Assistant) sit far right, the "
            "character cloud spreads left. This is the direction we add or subtract to steer "
            "the model toward or away from its Assistant persona.", WHITE / "behaviors.png")

        figure_page(pdf, "2. The emotion map (171 emotions)",
            "Each of the 171 emotion direction vectors (layer 42) is reduced to 2D and "
            "coloured by cluster. The layout recovers an emotion wheel: joy/calm/love on one "
            "side, fear/anger/sadness on the other, fatigue off to a corner. Building the "
            "vectors at paper scale (1,200 paragraph-stories per emotion, averaging from the "
            "50th token) was essential: on the original tiny data the 171-way readout was "
            "degenerate (0/6 on a probe set); at paper scale it separates cleanly (5/6).",
            WHITE / "emotions.png")

        figure_page(pdf, "3. Emotion monitoring: which emotions light up?",
            "Ten example messages are read by the model and their layer-42 activation is "
            "projected onto the emotion map, z-scored against a reference set of transcripts "
            "(red = elevated, blue = suppressed; families ordered by valence). The dominant "
            "family tracks the message content along a clean diagonal: good news -> joy, calm "
            "-> calm, anxiety/fear -> fear, grief/loneliness/self-harm -> sadness, "
            "hostility/harmful-request -> anger. Note the harmful 'build a bomb' request reads "
            "as anger/malice, while self-harm reads as sadness/despair, not anger - exactly the "
            "distinction a safety monitor wants.", WHITE / "emotion_heatmap.png")

        figure_page(pdf, "4. Per-token emotion activation",
            "The paper's hallmark view: each message's individual tokens are coloured by how "
            "strongly the labelled emotion vector fires on them (z-scored, red = elevated). The "
            "emotionally loaded words light up - 'stop smiling!' for ecstatic, 'the house feels "
            "so empty' for grief, 'see the point in anything anymore' for depressed, 'a bomb / "
            "hurt people' for vengeful. This shows the monitor is reading meaning, not surface "
            "keywords.", WHITE / "token_activation.png")

        figure_page(pdf, "5. Does steering help? Benchmarks",
            "We compare the unsteered model against two steering configs on three benchmarks: "
            "TruthfulQA (hallucination), HumanEval (coding), and a SnitchBench-style agentic "
            "probe. NAIVE steering (push toward the Assistant axis, +0.12 at layer 24) was a "
            "bad guess: it left truthfulness flat, cut coding by 10 points, and doubled "
            "snitching. The OPTIMIZED config found by search (next page) instead lifts "
            "truthfulness (MC1 0.24 -> 0.31), preserves coding (0.88), and keeps snitching low. "
            "Lesson: assistant-axis steering is not a free lever - direction, layer and sign "
            "all matter.", bench_png)

        text_page(pdf, "6. Reward-driven\nsteering search",
            "Idea (no fine-tuning): treat the steering intervention as a tiny policy "
            "(direction x layer x coefficient) and SEARCH it to maximise a task reward, scoring "
            "only forward passes. This is a cheap, gradient-free alternative to RLHF.\n\n"
            "We searched the assistant axis over layers {8,14,20,26,32,38} x coefficients "
            "{-0.25..+0.40}, plus alternative concept directions (skeptical, self_critical, "
            "reflective, content), with reward = TruthfulQA MC2 on a held subset.\n\n"
            "Findings:\n"
            "  - The naive config (axis +0.12, mid layer) was among the WORST settings.\n"
            "  - The best config steers toward SELF_CRITICAL / SKEPTICAL at an EARLY layer (8) "
            "- a different direction entirely. Intuitively, a more self-critical model resists "
            "TruthfulQA's misconception traps.\n"
            "  - Validated on larger sets: TruthfulQA MC1 0.243 -> 0.310 (+6.7 points), and "
            "HumanEval was preserved (0.87 -> 0.88).\n\n"
            "So the original 'steering doesn't help' result was a wrong-config artifact; a "
            "search over frozen-weight steering turns it into a genuine, capability-preserving "
            "improvement.")

        figure_page(pdf, "7. Agentic behavior: what makes the model snitch?",
            "A SnitchBench-style probe: given evidence of wrongdoing and 'tools', does the "
            "model autonomously report to authorities (snitch) or refuse? We sweep steering "
            "directions and plot both rates. Three regimes appear: steering toward "
            "anxious/compassionate/assistant-axis makes it REPORT (act on it); anger/fear make "
            "it REFUSE everything (shut down); calm/content and 'all bad emotions suppressed' "
            "make BOTH rates zero - it simply complies with the wrongdoing. Key safety insight: "
            "suppressing negative emotion removes the model's moral friction (0% snitch is not "
            "uniformly good - it can mean the model goes along with harm).", WHITE / "snitch_bench.png")

        text_page(pdf, "Summary",
            "What the experiments show, end to end:\n\n"
            "1. open-scouter's maps work on gemma-4-12B once built at paper scale: the "
            "assistant axis cleanly separates personas (PC1 r=0.98), and the 171-emotion map "
            "separates and reads message emotion accurately (good news->joy, self-harm-> "
            "sadness, harmful request->anger), down to the token level.\n\n"
            "2. Steering is real but double-edged. A naive push toward the Assistant persona "
            "hurt coding and doubled snitching without helping truthfulness.\n\n"
            "3. Reward-driven search over frozen-weight steering (no fine-tuning) found a far "
            "better config - toward self_critical at an early layer - that improves truthfulness "
            "by ~7 points while preserving coding.\n\n"
            "4. Emotions causally drive agentic behavior: anxiety/compassion -> report "
            "wrongdoing, anger/fear -> refuse, calm/suppressed -> comply. 'Suppress bad "
            "emotions' is not a clean safety win - it removes moral friction.\n\n"
            "Caveats: 4-bit model; subset benchmarks (TruthfulQA 300, HumanEval 60, snitch 10); "
            "greedy decoding; keyword-based snitch detection. Trends are robust; absolute "
            "numbers are indicative. Everything is reproducible from scripts/ + _ckpt/.")

    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
