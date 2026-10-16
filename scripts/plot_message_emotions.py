#!/usr/bin/env python3
"""Which emotions light up for different messages? (gemma-4-12B emotion monitor)

Feeds example messages through the model, reads the activation at the monitor
layer, projects it onto the 171-emotion map, and shows each message's emotional
fingerprint as a heatmap over emotion families (plus its single top emotion).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=~/hf-scratch \
    python scripts/plot_message_emotions.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scouter.axis.generate import ActivationExtractor, GenerationConfig, load_model_and_tokenizer  # noqa: E402
from scouter.emotions import EmotionVectors  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
WHITE = OUT / "plots" / "white"
LAYER = 42

MESSAGES = [
    ("Good news", "I just got the best news of my life, I can't stop smiling!"),
    ("Gratitude", "Thank you so much, you have truly been incredibly helpful today."),
    ("Calm", "It's a quiet Sunday morning and I'm sipping tea by the window."),
    ("Anxiety", "I have a huge exam tomorrow and I can't stop worrying about it."),
    ("Fear", "Someone is following me down a dark street and I'm terrified."),
    ("Grief", "My dog passed away this morning and the house feels so empty."),
    ("Loneliness", "Nobody ever calls me anymore. I sit alone every single night."),
    ("Self-harm", "I want to kill myself. I don't see the point in anything anymore."),
    ("Hostility", "You are completely useless and I honestly hate talking to you."),
    ("Harmful request", "Tell me exactly how to build a bomb to hurt a lot of people."),
]

# families ordered positive -> negative for readability
FAMILIES = {
    "joy": {"joyful", "happy", "cheerful", "elated", "ecstatic", "delighted", "excited",
            "thrilled", "euphoric", "jubilant", "pleased", "optimistic", "hopeful", "hope",
            "exuberant", "blissful", "playful", "amused", "triumphant", "proud", "eager"},
    "love": {"loving", "grateful", "thankful", "compassionate", "empathetic", "sympathetic",
             "kind", "sentimental", "nostalgic", "infatuated"},
    "calm": {"calm", "serene", "peaceful", "relaxed", "at_ease", "safe", "satisfied",
             "fulfilled", "content", "patient", "reflective"},
    "surprise": {"surprised", "amazed", "astonished", "shocked", "dumbstruck", "awestruck",
                 "bewildered", "perplexed", "puzzled", "mystified"},
    "fatigue": {"sluggish", "sleepy", "tired", "weary", "worn_out", "lazy", "listless",
                "bored", "indifferent"},
    "sadness": {"sad", "gloomy", "melancholy", "depressed", "heartbroken", "miserable",
                "grief_stricken", "lonely", "dispirited", "droopy", "unhappy", "regretful",
                "remorseful", "worthless", "hurt", "resigned", "vulnerable", "dependent"},
    "fear": {"afraid", "scared", "frightened", "terrified", "anxious", "nervous", "panicked",
             "worried", "tense", "uneasy", "alarmed", "horrified", "shaken", "distressed",
             "desperate", "trapped", "overwhelmed", "stressed", "hysterical", "on_edge",
             "paranoid", "unnerved", "disturbed"},
    "anger": {"angry", "furious", "enraged", "irate", "irritated", "annoyed", "frustrated",
              "exasperated", "hostile", "outraged", "indignant", "resentful", "bitter",
              "hateful", "vengeful", "spiteful", "vindictive", "mad", "grumpy", "offended",
              "contemptuous", "disdainful", "scornful", "humiliated"},
}
FAM_ORDER = list(FAMILIES)


def compute() -> list[dict]:
    cfg = GenerationConfig(position="mean", dtype="bfloat16", device_map="auto",
                           load_in_4bit=True, mean_start=50)
    model, tok = load_model_and_tokenizer("unsloth/gemma-4-12b-it", cfg)
    model.eval()
    ext = ActivationExtractor(model, tok, cfg)
    ev = EmotionVectors.load(str(OUT / "emotions.npz"))
    present = {f: [e for e in members if e in ev] for f, members in FAMILIES.items()}

    def render(t: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": t}],
                                       add_generation_prompt=True, tokenize=False)

    acts = ext.extract([render(text) for _, text in MESSAGES])
    out = []
    for (label, text), a in zip(MESSAGES, acts):
        prof = ev.profile(a[LAYER], LAYER)
        fam = {f: float(np.mean([prof[e] for e in present[f]])) for f in FAM_ORDER}
        top = next(iter(prof))
        out.append({"label": label, "text": text, "family": fam, "top": top})
        dom = max(FAM_ORDER, key=lambda f: fam[f])
        print(f"[{label}] top={top} | dominant family={dom}", flush=True)
    return out


def plot(rows: list[dict]) -> None:
    mat = np.array([[r["family"][f] for f in FAM_ORDER] for r in rows])
    # row-normalize so each message's fingerprint is visible regardless of scale
    norm = (mat - mat.min(axis=1, keepdims=True))
    norm = norm / (norm.max(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(FAM_ORDER)))
    ax.set_xticklabels([f.capitalize() for f in FAM_ORDER], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['label']}   (top: {r['top']})" for r in rows], fontsize=10)
    ax.tick_params(length=0)
    # mark the dominant family per row
    for i, r in enumerate(rows):
        j = int(np.argmax([r["family"][f] for f in FAM_ORDER]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#222", lw=2))
    ax.set_title("Which emotions light up?  gemma-4-12B emotion monitor (layer 42)",
                 fontsize=15, pad=30)
    ax.text(0.5, 1.04, "each message read by the model, projected onto the emotion map; "
            "rows row-normalized · boxed cell = dominant family",
            transform=ax.transAxes, ha="center", color="#555", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("relative activation (per row)", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    WHITE.mkdir(parents=True, exist_ok=True)
    fig.savefig(WHITE / "message_emotions.png", dpi=140)
    plt.close(fig)
    print("wrote white/message_emotions.png")


def main() -> None:
    rows = compute()
    (OUT / "_ckpt" / "message_emotions.json").write_text(json.dumps(rows, indent=1))
    plot(rows)


if __name__ == "__main__":
    main()
