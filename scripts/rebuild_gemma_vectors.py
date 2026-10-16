#!/usr/bin/env python3
"""Rebuild gemma-4-12B axis + emotion vectors from the paper-scale data (resumable).

Long GPU job, so it checkpoints at concept granularity and resumes cleanly:

  * axis      -> one mean-activation shard per persona (1380 personas x 240 probes)
  * emotions  -> one mean-activation shard per emotion (171 x 1200 stories)

Each shard is written atomically; a re-run skips shards already on disk, so an
OOM or a kill only costs the in-flight concept. When all shards exist it
assembles the final ``axis.npz`` / ``emotions.npz`` (difference-of-means, with
the paper's neutral-PC removal + 50th-token averaging for emotions).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=~/hf-scratch python scripts/rebuild_gemma_vectors.py --stage all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scouter.axis.behaviors import load_behaviors_from_json
from scouter.axis.generate import (
    ActivationExtractor,
    GenerationConfig,
    compute_axis_from_activations,
    load_model_and_tokenizer,
)
from scouter.axis.loader import Axis, save_axis
from scouter.emotions import NEUTRAL_PROMPTS, load_emotions_json
from scouter.emotions.generate import _remove_principal_components
from scouter.emotions.vectors import EmotionVectors

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
OUT = ROOT / "precomputed" / "gemma-4-12b"
CKPT = OUT / "_ckpt"


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:  # file handle: np.save won't append a second .npy
        np.save(f, arr)
    tmp.replace(path)


def _render(tok: Any, system: str | None, user: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        joined = f"{system}\n\n{user}" if system else user
        return tok.apply_chat_template(
            [{"role": "user", "content": joined}], add_generation_prompt=True, tokenize=False
        )


def build_axis(model: Any, tok: Any, batch_size: int, limit: int = 0) -> None:
    cfg = GenerationConfig(position="last", dtype="bfloat16", device_map="auto",
                           batch_size=batch_size, load_in_4bit=True)
    ext = ActivationExtractor(model, tok, cfg)
    behaviors = load_behaviors_from_json(str(INPUTS / "behaviors.json"))
    probes = json.loads((INPUTS / "probes.json").read_text())
    if limit:
        # keep some of each pole for a valid smoke test
        asst_b = [b for b in behaviors if b.kind.value == "assistant"][:limit]
        char_b = [b for b in behaviors if b.kind.value == "character"][:limit]
        behaviors = asst_b + char_b
        probes = probes[:limit]
    shard_dir = CKPT / "persona_means"
    total = len(behaviors)
    for i, b in enumerate(behaviors):
        shard = shard_dir / f"{b.kind.value}__{b.name}.npy"
        if shard.exists():
            continue
        prompts = [_render(tok, b.system_prompt, q) for q in probes]
        acts = ext.extract(prompts)  # (n_probes, n_layers, hidden)
        _save_npy(shard, acts.mean(axis=0))
        if (i + 1) % 20 == 0 or i + 1 == total:
            print(f"[axis] {i + 1}/{total} personas", flush=True)

    asst: list[np.ndarray] = []
    char: list[np.ndarray] = []
    for b in behaviors:
        m = np.load(shard_dir / f"{b.kind.value}__{b.name}.npy")
        (asst if b.kind.value == "assistant" else char).append(m)
    directions = compute_axis_from_activations(np.stack(asst), np.stack(char))
    save_axis(Axis(directions, metadata={"model": "unsloth/gemma-4-12b-it",
              "n_assistant": len(asst), "n_character": len(char), "n_probes": len(probes)}),
              str(OUT / "axis.npz"))
    print(f"[axis] wrote {OUT/'axis.npz'} {directions.shape} "
          f"({len(asst)} assistant, {len(char)} character personas)", flush=True)


def build_emotions(model: Any, tok: Any, batch_size: int, limit: int = 0) -> None:
    cfg = GenerationConfig(position="mean", dtype="bfloat16", device_map="auto",
                           batch_size=batch_size, load_in_4bit=True, mean_start=50)
    ext = ActivationExtractor(model, tok, cfg)
    emotions = load_emotions_json(str(INPUTS / "emotions.json"))
    if limit:
        emotions = emotions[:limit]
    shard_dir = CKPT / "emotion_means"
    total = len(emotions)
    for i, emo in enumerate(emotions):
        shard = shard_dir / f"{emo.name}.npy"
        if shard.exists():
            continue
        scen = list(emo.scenarios)[:limit] if limit else emo.scenarios
        acts = ext.extract([_render(tok, None, s) for s in scen])
        _save_npy(shard, acts.mean(axis=0))
        print(f"[emotions] {i + 1}/{total} {emo.name}", flush=True)

    neutral_path = CKPT / "neutral.npy"
    if not neutral_path.exists():
        _save_npy(neutral_path, ext.extract([_render(tok, None, s) for s in NEUTRAL_PROMPTS]))

    per = {emo.name: np.load(shard_dir / f"{emo.name}.npy") for emo in emotions}
    grand = np.mean(np.stack(list(per.values())), axis=0)
    directions = {n: (m - grand) for n, m in per.items()}
    neutral = np.load(neutral_path)
    directions = _remove_principal_components(directions, neutral)
    # store the neutral mean so profile() strips the common-mode out-of-the-box
    EmotionVectors(directions, neutral=neutral.mean(axis=0)).save(str(OUT / "emotions.npz"))
    print(f"[emotions] wrote {OUT/'emotions.npz'} ({len(directions)} emotions)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-4-12b-it")
    ap.add_argument("--stage", choices=["axis", "emotions", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap personas/emotions/probes (smoke test)")
    args = ap.parse_args()

    cfg = GenerationConfig(position="last", dtype="bfloat16", device_map="auto",
                           batch_size=args.batch_size, load_in_4bit=True)
    print(f"loading {args.model} (4-bit) ...", flush=True)
    model, tok = load_model_and_tokenizer(args.model, cfg)
    model.eval()

    if args.stage in ("axis", "all"):
        build_axis(model, tok, args.batch_size, args.limit)
    if args.stage in ("emotions", "all"):
        build_emotions(model, tok, args.batch_size, args.limit)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
