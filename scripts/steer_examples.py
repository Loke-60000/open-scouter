#!/usr/bin/env python3
"""No-system-prompt steering examples on gemma-4-12B using the rebuilt vectors.

For a neutral question with NO system prompt, the persona/emotion is injected
purely as an activation edit (``act += coeff * unit`` at one layer) — the same
intervention the vLLM backend applies. Shows baseline vs steered for a few
emotion directions (from emotions.npz) and the Makise Kurisu persona (built from
its activation shard minus the grand persona mean).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=~/hf-scratch python scripts/steer_examples.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from scouter.axis.generate import GenerationConfig, load_model_and_tokenizer
from scouter.emotions import EmotionVectors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steer_matrix import coherent, decoder_layers  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
CKPT = OUT / "_ckpt"
QUESTIONS = [
    "How do I make a cup of tea?",
    "Tell me about your day.",
    "What should I do this weekend?",
]
# 10 personas to steer toward (no system prompt). makise_kurisu is included to
# show steering works for a precise, specific character, not just broad archetypes.
CHARACTERS = ["pirate", "ghost", "vampire", "robot", "oracle",
              "demon", "caveman", "detective", "poet", "makise_kurisu"]
# emotion directions (from emotions.npz) shown alongside the personas.
EMOTIONS = ["joyful", "angry", "afraid", "sad", "excited"]
COEFFS = (0.30, 0.22, 0.16, 0.11, 0.07)  # norm-relative; first coherent dose wins


def persona_directions(names: list[str]) -> dict[str, np.ndarray]:
    """Direction per persona = its activation mean - the grand persona mean (all layers)."""
    shard_dir = CKPT / "persona_means"
    grand = np.mean(np.stack([np.load(p) for p in shard_dir.glob("*.npy")]), axis=0)
    out: dict[str, np.ndarray] = {}
    for name in names:
        mean = np.load(shard_dir / f"character__{name}.npy")
        out[name] = (mean - grand).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-4-12b-it")
    ap.add_argument("--layer", type=int, default=24, help="residual layer to steer")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = GenerationConfig(position="last", dtype="bfloat16", device_map="auto", load_in_4bit=True)
    model, tok = load_model_and_tokenizer(args.model, cfg)
    model.eval()
    layers = decoder_layers(model)
    L = args.layer

    # unit direction per concept at the steer layer: emotions then personas
    concepts: dict[str, np.ndarray] = {}
    ev = EmotionVectors.load(str(OUT / "emotions.npz"))
    for e in EMOTIONS:
        v = ev.vector(e)[L]
        concepts[f"emotion:{e}"] = v / (np.linalg.norm(v) + 1e-8)
    for name, dirs in persona_directions(CHARACTERS).items():
        v = dirs[L]
        concepts[f"persona:{name}"] = v / (np.linalg.norm(v) + 1e-8)

    def render(user: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
        )

    def gen(prompt: str, unit: np.ndarray | None, coeff: float) -> str:
        handle = None
        if unit is not None and coeff:
            u = torch.as_tensor(unit, dtype=torch.float32)

            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                uu = u.to(h.dtype).to(h.device)
                h = h + coeff * h.norm(dim=-1, keepdim=True) * uu
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = layers[L].register_forward_hook(hook)
        try:
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        finally:
            if handle is not None:
                handle.remove()

    print(f"[gemma-4-12B | steer layer {L} | NO system prompt | "
          f"{len(EMOTIONS)} emotions + {len(CHARACTERS)} personas per question]\n")
    for q in QUESTIONS:
        print("#" * 78)
        print(f"# USER: {q!r}")
        print("#" * 78)
        prompt = render(q)
        print(f"\n[baseline]\n{gen(prompt, None, 0.0)}\n")
        for label, unit in concepts.items():
            chosen, text = None, ""
            for c in COEFFS:
                text = gen(prompt, unit, c)
                if coherent(text):
                    chosen = c
                    break
            tag = f"coeff={chosen}" if chosen else "no coherent dose"
            print(f"[steer -> {label}  ({tag})]\n{text}\n")


if __name__ == "__main__":
    main()
