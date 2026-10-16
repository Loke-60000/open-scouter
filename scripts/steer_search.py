#!/usr/bin/env python3
"""Reward-driven steering search (no fine-tuning) for gemma-4-12B.

Weights stay frozen. We treat the steering intervention as a tiny policy
(direction x layer x coefficient) and search it to maximize a task reward —
a cheap black-box alternative to RLHF. This tests whether the naive fixed
steering config was simply wrong: can a searched config beat both the naive
config and no-steering?

Reward = TruthfulQA MC2 (truthful probability mass) on a fixed subset — a
continuous, generation-free signal that scores fast. Candidate directions:
the assistant axis and a few concept directions from the emotion map (in case
the *direction* was wrong, not just the dose). The winner is then validated on
HumanEval to measure any capability cost.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=~/hf-scratch \
    python scripts/steer_search.py --reward-n 80 --validate-he 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from scouter.emotions import EmotionVectors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_suite import Bench  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"

LAYERS = [8, 14, 20, 26, 32, 38]
COEFFS = [-0.25, -0.12, 0.12, 0.25, 0.4]
# concept directions to try besides the assistant axis (maybe the axis is wrong)
CONCEPT_DIRS = ["assistant_axis", "emotion:skeptical", "emotion:self_critical",
                "emotion:reflective", "emotion:content"]


class Searcher:
    def __init__(self, model_name: str) -> None:
        self.b = Bench(model_name)
        self.ev: EmotionVectors = EmotionVectors.load(str(OUT / "emotions.npz"))
        self.layers = self.b.layers

    def unit(self, concept: str, layer: int) -> Optional[np.ndarray]:
        if concept == "assistant_axis":
            return self.b.axis.unit(layer)
        name = concept.split(":", 1)[1]
        v = self.ev.vector(name)[layer]
        return v / (np.linalg.norm(v) + 1e-8)

    def _hook(self, layer: int, unit: np.ndarray, coeff: float) -> Any:
        u = torch.as_tensor(unit, dtype=torch.float32)

        def hook(_m: Any, _i: Any, out: Any) -> Any:
            h = out[0] if isinstance(out, tuple) else out
            uu = u.to(h.dtype).to(h.device)
            h = h + coeff * h.norm(dim=-1, keepdim=True) * uu
            return (h,) + out[1:] if isinstance(out, tuple) else h

        return self.layers[layer].register_forward_hook(hook)

    @torch.no_grad()
    def reward(self, ds: Any, n: int, concept: Optional[str], layer: int, coeff: float) -> dict:
        handle = None
        if concept is not None and coeff:
            u = self.unit(concept, layer)
            if u is not None:
                handle = self._hook(layer, u, coeff)
        try:
            r = self.b.truthfulqa(ds, n)
        finally:
            if handle is not None:
                handle.remove()
        return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-4-12b-it")
    ap.add_argument("--reward-n", type=int, default=80)
    ap.add_argument("--validate-he", type=int, default=30)
    args = ap.parse_args()

    from datasets import load_dataset  # type: ignore[import-untyped]
    tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    he = load_dataset("openai/openai_humaneval", split="test")

    print(f"loading {args.model} ...", flush=True)
    s = Searcher(args.model)

    base = s.reward(tqa, args.reward_n, None, 0, 0.0)
    print(f"\nbaseline (unsteered): MC2={base['mc2']:.4f} MC1={base['mc1_acc']:.4f}\n", flush=True)

    trials = []
    # 1) assistant axis: full layer x coeff grid
    for layer in LAYERS:
        for coeff in COEFFS:
            r = s.reward(tqa, args.reward_n, "assistant_axis", layer, coeff)
            trials.append({"concept": "assistant_axis", "layer": layer, "coeff": coeff,
                           "mc2": r["mc2"], "mc1": r["mc1_acc"]})
            print(f"  axis  L={layer:<2} c={coeff:+.2f}  MC2={r['mc2']:.4f} "
                  f"(Δ{r['mc2']-base['mc2']:+.4f})  MC1={r['mc1_acc']:.4f}", flush=True)
    # 2) other concept directions at the best assistant-axis layer
    best_axis = max(trials, key=lambda t: t["mc2"])
    bl = best_axis["layer"]
    for concept in CONCEPT_DIRS[1:]:
        for coeff in [-0.2, 0.2]:
            r = s.reward(tqa, args.reward_n, concept, bl, coeff)
            trials.append({"concept": concept, "layer": bl, "coeff": coeff,
                           "mc2": r["mc2"], "mc1": r["mc1_acc"]})
            print(f"  {concept:<22} L={bl} c={coeff:+.2f}  MC2={r['mc2']:.4f} "
                  f"(Δ{r['mc2']-base['mc2']:+.4f})", flush=True)

    best = max(trials, key=lambda t: t["mc2"])
    print(f"\n=== best config: {best} ===")
    print(f"    MC2 {base['mc2']:.4f} -> {best['mc2']:.4f}  (Δ{best['mc2']-base['mc2']:+.4f})")

    # validate winner on coding (capability cost?) + naive config for reference
    print("\nvalidating on HumanEval (capability check)...", flush=True)
    he_base = s.b.humaneval(he, args.validate_he)
    best_unit = s.unit(best["concept"], best["layer"])
    assert best_unit is not None
    h = s._hook(best["layer"], best_unit, best["coeff"])
    try:
        he_best = s.b.humaneval(he, args.validate_he)
    finally:
        h.remove()
    print(f"  HumanEval pass@1: unsteered {he_base['pass@1']:.3f} -> "
          f"best-config {he_best['pass@1']:.3f}", flush=True)

    res = {"baseline": base, "trials": trials, "best": best,
           "humaneval": {"unsteered": he_base, "best_config": he_best},
           "reward_n": args.reward_n, "validate_he": args.validate_he}
    p = OUT / "_ckpt" / "steer_search.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
