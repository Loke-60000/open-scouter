"""Build emotion vectors from a model (difference of means + optional PCA).

Following Anthropic's "Emotion Concepts and their Function in a Large Language
Model" (Lindsey et al., transformer-circuits.pub, April 2026):

    emotion_vector[E] = mean(activations | emotion E) - mean(activations | all emotions)

Optionally, the top principal components of neutral text (the shared "being
prompted about feelings" component) are projected out to isolate each emotion.

The numpy core has no torch dependency; generating from a live model needs the
``generate`` extra.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np

from ..axis.generate import ActivationExtractor, GenerationConfig, load_model_and_tokenizer
from .catalog import DEFAULT_EMOTIONS, NEUTRAL_PROMPTS, Emotion, load_emotions_json
from .vectors import EmotionVectors, FloatArray


def _remove_principal_components(
    vectors: dict[str, FloatArray], neutral: FloatArray, variance: float = 0.5
) -> dict[str, FloatArray]:
    """Project the top PCs of ``neutral`` (per layer) out of each emotion vector."""
    n_layers = neutral.shape[1]
    out = {name: v.copy() for name, v in vectors.items()}
    for layer in range(n_layers):
        x = neutral[:, layer, :].astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        if x.shape[0] < 2:
            continue
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        ratio = (s**2) / (s**2).sum()
        k = int(np.searchsorted(np.cumsum(ratio), variance) + 1)
        pcs = vt[:k]  # (k, hidden)
        for name in out:
            v = out[name][layer].astype(np.float64)
            v = v - pcs.T @ (pcs @ v)
            out[name][layer] = v.astype(np.float32)
    return out


def build_emotion_vectors(
    extractor: ActivationExtractor,
    *,
    emotions: Sequence[Emotion] = DEFAULT_EMOTIONS,
    neutral_prompts: Optional[Sequence[str]] = NEUTRAL_PROMPTS,
) -> EmotionVectors:
    """Build emotion vectors with an already-loaded :class:`ActivationExtractor`.

    For each emotion, the model reads its scenarios; activations are averaged
    over the scenario tokens, then over the scenarios, then the mean across all
    emotions is subtracted (difference of means). If ``neutral_prompts`` is
    given, the neutral text's top principal components (per layer) are projected
    out to remove confounds. Mirrors the "Emotion Concepts" paper's recipe.
    """

    render = extractor.render_user
    per_emotion: dict[str, FloatArray] = {}
    for emo in emotions:
        if not emo.scenarios:
            continue
        acts = extractor.extract([render(s) for s in emo.scenarios])
        per_emotion[emo.name] = acts.mean(axis=0)  # (n_layers, hidden)

    grand_mean = np.mean(np.stack(list(per_emotion.values())), axis=0)
    directions = {name: (mean - grand_mean) for name, mean in per_emotion.items()}

    neutral_ref: FloatArray | None = None
    if neutral_prompts:
        neutral = extractor.extract([render(s) for s in neutral_prompts])
        directions = _remove_principal_components(directions, neutral)
        # Keep the neutral mean so profile() can strip the common-mode at query time.
        neutral_ref = neutral.mean(axis=0)

    return EmotionVectors(directions, neutral=neutral_ref)


def generate_emotion_vectors(
    model_name: str,
    *,
    emotions: Sequence[Emotion] = DEFAULT_EMOTIONS,
    neutral_prompts: Optional[Sequence[str]] = NEUTRAL_PROMPTS,
    config: Optional[GenerationConfig] = None,
    save_path: Optional[str] = None,
) -> EmotionVectors:
    """Load ``model_name`` and build its emotion vectors."""
    config = config or GenerationConfig(position="mean")
    model, tokenizer = load_model_and_tokenizer(model_name, config)
    extractor = ActivationExtractor(model, tokenizer, config)
    vectors = build_emotion_vectors(
        extractor, emotions=emotions, neutral_prompts=neutral_prompts
    )
    if save_path:
        vectors.save(save_path)
    return vectors


def main() -> None:
    ap = argparse.ArgumentParser(description="Build emotion vectors from a model.")
    ap.add_argument("--model", required=True, help="HF model repo id or local path")
    ap.add_argument("--out", required=True, help="output .npz path (all emotions)")
    ap.add_argument(
        "--emotions", default=None, help="JSON file of emotions+scenarios (see generate_inputs.py)"
    )
    ap.add_argument("--no-orthogonalize", action="store_true", help="skip neutral-PC removal")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="load weights in 4-bit (bitsandbytes NF4); fits a 12B model on a 24 GB GPU",
    )
    ap.add_argument(
        "--mean-start",
        type=int,
        default=50,
        help="average residual states from this token onward (paper recipe: 50)",
    )
    args = ap.parse_args()

    emotions = load_emotions_json(args.emotions) if args.emotions else DEFAULT_EMOTIONS
    config = GenerationConfig(
        position="mean",
        dtype=args.dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        mean_start=args.mean_start,
    )
    vectors = generate_emotion_vectors(
        args.model,
        emotions=emotions,
        neutral_prompts=None if args.no_orthogonalize else NEUTRAL_PROMPTS,
        config=config,
        save_path=args.out,
    )
    print(
        f"wrote {args.out}: {len(vectors)} emotions, "
        f"{vectors.n_layers} layers, dim {vectors.hidden_dim}"
    )
    print("emotions:", ", ".join(vectors.names))


if __name__ == "__main__":
    main()
