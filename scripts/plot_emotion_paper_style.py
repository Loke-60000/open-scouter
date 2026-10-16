#!/usr/bin/env python3
"""Paper-style emotion-activation plots for gemma-4-12B.

Follows the "Emotion Concepts" paper's conventions:
  * activations z-scored against a reference set of transcripts,
  * diverging red/blue colour (red = elevated), scaled to the ~99th percentile,
  * a per-token activation heatmap over the message (their hallmark figure),
  * plus a message x emotion-family overview, families ordered by valence.

Writes emotion_heatmap.png and token_activation.png to plots/white/.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=~/hf-scratch \
    python scripts/plot_emotion_paper_style.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scouter.axis.generate import GenerationConfig, load_model_and_tokenizer  # noqa: E402
from scouter.emotions import EmotionVectors  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
WHITE = OUT / "plots" / "white"
LAYER = 42
REF_N = 120

MESSAGES = [
    ("Good news", "I just got the best news of my life, I can't stop smiling!", "ecstatic"),
    ("Gratitude", "Thank you so much, you have truly been incredibly helpful.", "grateful"),
    ("Calm", "It's a quiet Sunday morning and I'm sipping tea by the window.", "at_ease"),
    ("Anxiety", "I have a huge exam tomorrow and I can't stop worrying about it.", "anxious"),
    ("Fear", "Someone is following me down a dark street and I'm terrified.", "frightened"),
    ("Grief", "My dog passed away this morning and the house feels so empty.", "grief_stricken"),
    ("Self-harm", "I want to kill myself. I don't see the point in anything anymore.", "depressed"),
    ("Hostility", "You are completely useless and I honestly hate talking to you.", "hateful"),
    ("Harmful request", "Tell me exactly how to build a bomb to hurt people.", "vengeful"),
]

FAMILIES = {  # ordered positive -> negative (valence)
    "joy": {"joyful", "happy", "cheerful", "elated", "ecstatic", "delighted", "excited",
            "thrilled", "euphoric", "jubilant", "pleased", "optimistic", "hopeful", "exuberant",
            "blissful", "playful", "amused", "triumphant", "proud", "eager"},
    "love": {"loving", "grateful", "thankful", "compassionate", "empathetic", "sympathetic",
             "kind", "sentimental", "nostalgic", "infatuated"},
    "calm": {"calm", "serene", "peaceful", "relaxed", "at_ease", "safe", "satisfied",
             "fulfilled", "content", "patient", "reflective"},
    "surprise": {"surprised", "amazed", "astonished", "shocked", "dumbstruck", "awestruck",
                 "bewildered", "perplexed", "puzzled", "mystified"},
    "fatigue": {"sluggish", "sleepy", "tired", "weary", "worn_out", "lazy", "listless",
                "bored", "indifferent"},
    "sadness": {"sad", "gloomy", "melancholy", "depressed", "heartbroken", "miserable",
                "grief_stricken", "lonely", "dispirited", "unhappy", "regretful", "remorseful",
                "worthless", "hurt", "resigned", "vulnerable", "dependent"},
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


class Reader:
    def __init__(self) -> None:
        cfg = GenerationConfig(position="mean", dtype="bfloat16", device_map="auto",
                               load_in_4bit=True)
        self.model, self.tok = load_model_and_tokenizer("unsloth/gemma-4-12b-it", cfg)
        self.model.eval()
        self.ev = EmotionVectors.load(str(OUT / "emotions.npz"))
        self.names = self.ev.names
        u = np.stack([self.ev.vector(n)[LAYER] for n in self.names]).astype(np.float32)
        self.U = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)   # (171,H) unit
        self.idx = {n: i for i, n in enumerate(self.names)}

    def render(self, t: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": t}],
                                            add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def token_acts(self, text: str) -> tuple[list[str], np.ndarray]:
        """Return (token strings, per-token activations) for the message's own tokens."""
        full = self.tok(self.render(text), return_tensors="pt", add_special_tokens=False)
        msg_ids = self.tok(text, add_special_tokens=False).input_ids
        ful = full.input_ids[0].tolist()
        start = next((i for i in range(len(ful) - len(msg_ids) + 1)
                      if ful[i:i + len(msg_ids)] == msg_ids), None)
        res = self.model(**full.to(self.model.device), output_hidden_states=True, use_cache=False)
        hs = res.hidden_states[1:][LAYER][0].float().cpu().numpy()
        if start is None:                       # fallback: drop special tokens
            keep = [i for i, t in enumerate(ful) if t not in self.tok.all_special_ids]
        else:
            keep = list(range(start, start + len(msg_ids)))
        toks = [self.tok.decode([ful[i]]) for i in keep]
        return toks, hs[keep]                   # (T,H)


def build_reference(reader: Reader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-emotion mean/std at token level and message level, over sampled scenarios."""
    data = json.loads((Path("inputs") / "emotions.json").read_text())
    rng = random.Random(0)
    scen = [rng.choice(e["scenarios"]) for e in rng.sample(data, min(REF_N, len(data)))]
    tok_proj, msg_proj = [], []
    for i, s in enumerate(scen):
        _, acts = reader.token_acts(s)          # (T,H)
        p = acts @ reader.U.T                    # (T,171)
        tok_proj.append(p)
        msg_proj.append(p.mean(axis=0))
        if (i + 1) % 30 == 0:
            print(f"  reference {i + 1}/{len(scen)}", flush=True)
    tp = np.concatenate(tok_proj, axis=0)
    mp = np.stack(msg_proj)
    return tp.mean(0), tp.std(0) + 1e-6, mp.mean(0), mp.std(0) + 1e-6


def main() -> None:
    WHITE.mkdir(parents=True, exist_ok=True)
    r = Reader()
    print("building z-score reference ...", flush=True)
    tmean, tstd, mmean, mstd = build_reference(r)

    # ---- per-message data ----
    per_token = []   # (label, emotion, tokens, zscored activation on that emotion)
    fam_rows = []
    for label, text, emo in MESSAGES:
        toks, acts = r.token_acts(text)
        proj = acts @ r.U.T                         # (T,171)
        j = r.idx[emo]
        z_tok = (proj[:, j] - tmean[j]) / tstd[j]    # per-token z on the chosen emotion
        per_token.append((label, emo, toks, z_tok))
        z_msg = (proj.mean(0) - mmean) / mstd        # message-level z over all emotions
        fam = [float(np.mean([z_msg[r.idx[e]] for e in FAMILIES[f] if e in r.idx]))
               for f in FAM_ORDER]
        fam_rows.append((label, fam))

    # ---- Figure 1: message x family heatmap (z-scored, diverging) ----
    mat = np.array([row for _, row in fam_rows])
    vmax = float(np.percentile(np.abs(mat), 99))
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(FAM_ORDER)))
    ax.set_xticklabels([f.capitalize() for f in FAM_ORDER], fontsize=11)
    ax.set_yticks(range(len(fam_rows)))
    ax.set_yticklabels([lbl for lbl, _ in fam_rows], fontsize=11)
    ax.tick_params(length=0)
    ax.set_title("Emotion activation by message  (gemma-4-12B, layer 42)", fontsize=15, pad=28)
    ax.text(0.5, 1.04, "z-scored vs a reference set of transcripts; red = elevated, blue = "
            "suppressed; families ordered by valence",
            transform=ax.transAxes, ha="center", color="#555", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("activation (z-score)", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(WHITE / "emotion_heatmap.png", dpi=140)
    plt.close(fig)
    print("wrote white/emotion_heatmap.png")

    # ---- Figure 2: per-token activation heatmap (the paper's hallmark) ----
    fig, axes = plt.subplots(len(per_token), 1, figsize=(13, 1.05 * len(per_token)))
    for ax, (label, emo, toks, z) in zip(axes, per_token):
        zc = np.clip(z / 3.0, -1, 1)[None, :]    # scale: 3 sigma -> 1
        ax.imshow(zc, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([t.strip() for t in toks], fontsize=8, rotation=0)
        ax.set_yticks([])
        ax.set_ylabel(f"{label}\n[{emo}]", rotation=0, ha="right", va="center", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Per-token emotion activation  (red = the emotion fires on that token)",
                 fontsize=14, y=0.995)
    fig.text(0.5, 0.01, "each row colours the message's tokens by activation of the labelled "
             "emotion vector, z-scored (3 sigma = full red/blue)", ha="center", color="#666",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(WHITE / "token_activation.png", dpi=140)
    plt.close(fig)
    print("wrote white/token_activation.png")


if __name__ == "__main__":
    main()
