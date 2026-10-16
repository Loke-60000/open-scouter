#!/usr/bin/env python3
"""One-command steering + emotion setup for gemma-4-12B (non-gated, 4-bit).

Builds both activation-vector files the server reads for gemma-4-12B:

  * ``axis.npz``      - the assistant axis (monitoring + steering)
  * ``emotions.npz``  - per-emotion directions (emotion mapping + steering)

``google/gemma-4-12B-it`` is gated, so this pulls the open ``unsloth`` mirror
and loads it in 4-bit (bitsandbytes NF4) so the 12B fits on a single 24 GB GPU.
The model is loaded once and reused for both passes; the assistant axis reads
the last-token state while the emotion pass averages over tokens, so each pass
gets its own extractor config over the shared weights.

    pip install -e ".[generate]" bitsandbytes
    python scripts/setup_gemma4_12b.py

Outputs land in ``precomputed/gemma-4-12b/`` and the script prints the env block
to serve the model with monitoring, steering, and emotion mapping enabled.
"""

from __future__ import annotations

import argparse
import os

from scouter.axis.generate import (
    ActivationExtractor,
    AxisGenerator,
    GenerationConfig,
    build_catalog,
    load_model_and_tokenizer,
)
from scouter.axis.loader import save_axis
from scouter.emotions import NEUTRAL_PROMPTS, build_emotion_vectors, load_emotions_json

DEFAULT_MODEL = "unsloth/gemma-4-12b-it"  # open mirror of the gated google/gemma-4-12B-it
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "inputs")
DEFAULT_OUT = os.path.join(ROOT, "precomputed", "gemma-4-12b")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the assistant axis and emotion vectors for gemma-4-12B."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="where to write axis.npz / emotions.npz")
    ap.add_argument("--dtype", default="bfloat16", help="compute dtype")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="load in full precision instead of 4-bit (needs ~24 GB of VRAM)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    load_4bit = not args.no_4bit
    print(f"loading {args.model} ({'4-bit nf4' if load_4bit else args.dtype}) ...")

    # Load the 12B once; `position` is an extraction-time setting, so the same
    # weights serve both the axis (last token) and the emotion (mean) passes.
    axis_cfg = GenerationConfig(
        position="last", dtype=args.dtype, batch_size=args.batch_size, load_in_4bit=load_4bit
    )
    model, tok = load_model_and_tokenizer(args.model, axis_cfg)

    # 1) Assistant axis: steering + drift monitoring.
    catalog = build_catalog(
        os.path.join(INPUTS, "behaviors.json"), os.path.join(INPUTS, "probes.json")
    )
    axis_gen = AxisGenerator(args.model, catalog=catalog, config=axis_cfg)
    # Reuse the shared load; AxisGenerator.load() no-ops when a model is present.
    axis_gen._model, axis_gen._tokenizer = model, tok
    axis = axis_gen.generate()
    axis_path = os.path.join(args.out_dir, "axis.npz")
    save_axis(axis, axis_path, metadata=axis.metadata)
    print(f"  axis     -> {axis_path}  {axis.directions.shape}")

    # 2) Emotion vectors: mapping (read the emotion in a reply) + steering.
    emo_cfg = GenerationConfig(
        position="mean", dtype=args.dtype, batch_size=args.batch_size, load_in_4bit=load_4bit,
        mean_start=50,  # paper recipe: average residual states from the 50th token
    )
    extractor = ActivationExtractor(model, tok, emo_cfg)
    emotions = load_emotions_json(os.path.join(INPUTS, "emotions.json"))
    vectors = build_emotion_vectors(extractor, emotions=emotions, neutral_prompts=NEUTRAL_PROMPTS)
    emo_path = os.path.join(args.out_dir, "emotions.npz")
    vectors.save(emo_path)
    print(f"  emotions -> {emo_path}  {len(vectors)} emotions, {vectors.n_layers} layers")

    print("\nserve gemma-4-12B with monitoring, steering, and emotion mapping:\n")
    print(f"  SCOUTER_MODEL={args.model} \\")
    print(f"  SCOUTER_AXIS={axis_path} \\")
    print(f"  SCOUTER_EMOTIONS={emo_path} \\")
    print("  scouter")


if __name__ == "__main__":
    main()
