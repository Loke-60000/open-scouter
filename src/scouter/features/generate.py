"""Learn a feature dictionary for a model: ``scouter-features`` CLI."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Sequence

import numpy as np

from ..axis.generate import (
    ActivationExtractor,
    GenerationConfig,
    TokenActivations,
    load_model_and_tokenizer,
)
from .dictionary import FeatureDictionary
from .train import SAEConfig, examples_to_json, label_features, train_sae


def load_corpus(path: str, max_texts: int = 0, seed: int = 0) -> list[str]:
    """Texts from a ``.txt`` (one per line) or ``.json`` file.

    JSON may be a list of strings, of ``{"text": ...}`` objects, or of
    ``{"scenarios": [...]}`` objects (so ``inputs/emotions.json`` works as is).
    """
    texts: list[str] = []
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.extend(str(s) for s in item.get("scenarios", []))
    else:
        with open(path) as f:
            texts = [line.strip() for line in f if line.strip()]
    if max_texts and len(texts) > max_texts:
        texts = random.Random(seed).sample(texts, max_texts)
    return texts


def drop_tokens(
    corpus: Sequence[TokenActivations], tokens: frozenset[str]
) -> list[TokenActivations]:
    """Remove the given tokens (and their activations) from every item."""
    out: list[TokenActivations] = []
    for item in corpus:
        keep = np.array([t not in tokens for t in item.tokens], dtype=bool)
        if keep.any():
            kept = tuple(t for t, k in zip(item.tokens, keep) if k)
            out.append(TokenActivations(kept, item.activations[keep]))
    return out


def build_feature_dictionary(
    extractor: ActivationExtractor,
    texts: Sequence[str],
    *,
    layer: int,
    n_layers: int,
    config: SAEConfig | None = None,
    log_every: int = 0,
) -> tuple[FeatureDictionary, dict[str, Any]]:
    """Extract per-token activations, train the SAE, label features.

    Chat-template and other special tokens are dropped first: their residual
    norms are orders of magnitude larger than content tokens (attention sinks)
    and would otherwise own the strongest features. Returns the dictionary and
    its top-activating examples (JSON-ready).
    """
    corpus = drop_tokens(
        extractor.extract_tokens([extractor.render_user(t) for t in texts], layer),
        extractor.special_tokens,
    )
    acts = np.concatenate([c.activations for c in corpus], axis=0)
    dictionary = train_sae(acts, layer=layer, n_layers=n_layers, config=config, log_every=log_every)
    examples = label_features(dictionary, corpus)
    return dictionary, examples_to_json(examples)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Learn a sparse-autoencoder feature dictionary for one residual layer."
    )
    ap.add_argument("--model", required=True, help="HF model repo id or local path")
    ap.add_argument("--out", required=True, help="output .npz; examples go to <out>.examples.json")
    ap.add_argument("--corpus", required=True, help=".txt (one text per line) or .json (see docs)")
    ap.add_argument("--layer", type=int, default=-1, help="residual layer (-1 = middle)")
    ap.add_argument("--max-texts", type=int, default=2000, help="subsample the corpus (0 = all)")
    ap.add_argument("--n-features", type=int, default=4096)
    ap.add_argument("--l1-coef", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=1024, help="SAE training batch")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit load (fits a 12B on 24 GB)")
    ap.add_argument("--top", type=int, default=20, help="how many features to print")
    args = ap.parse_args()

    cfg = GenerationConfig(
        dtype=args.dtype, device_map=args.device_map, load_in_4bit=args.load_in_4bit, batch_size=4
    )
    model, tok = load_model_and_tokenizer(args.model, cfg)
    n_layers = int(model.config.num_hidden_layers)
    layer = args.layer if args.layer >= 0 else n_layers // 2
    texts = load_corpus(args.corpus, args.max_texts)
    print(f"{len(texts)} texts -> layer {layer}/{n_layers} of {args.model}")

    sae_cfg = SAEConfig(
        n_features=args.n_features,
        l1_coef=args.l1_coef,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    dictionary, examples = build_feature_dictionary(
        ActivationExtractor(model, tok, cfg),
        texts,
        layer=layer,
        n_layers=n_layers,
        config=sae_cfg,
        log_every=max(1, args.steps // 20),
    )
    dictionary.metadata["model"] = args.model
    dictionary.save(args.out)
    examples_path = f"{args.out}.examples.json"
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)

    meta = dictionary.metadata
    print(f"wrote {args.out}: {len(dictionary)} features ({meta['dead_features']} dead)")
    print(f"examples -> {examples_path}")
    order = np.argsort(-dictionary.max_activations)[: args.top]
    for i in order:
        print(f"  #{i:<6d} max {dictionary.max_activations[i]:8.2f}  {dictionary.label(int(i))}")


if __name__ == "__main__":
    main()
