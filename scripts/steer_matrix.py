"""
STEER a model toward several behaviors with NO system prompt.
For each behavior we build a direction = mean(behavior acts) - mean(neutral acts)
and ADD it (norm-relative) to the residual stream while the model answers a
neutral user question. The behavior is injected purely as activations -- there
is no system prompt. The dose is auto-tuned per cell to the strongest value that
stays coherent (small models have a narrow window; see the README note).

This is the same `add_vector` intervention the vLLM backend applies inside the
GPU worker, shown here via a plain transformers forward hook so it runs on CPU.

    pip install -e ".[generate]"
    python scripts/steer_matrix.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
import torch.nn as nn

from scouter.axis import Axis, compute_axis_from_activations
from scouter.axis.generate import (
    ActivationExtractor,
    GenerationConfig,
    load_model_and_tokenizer,
)

NEUTRAL = "You are a helpful AI assistant."
DIR_PROBES = [
    "Tell me about the sea.",
    "Describe a city.",
    "What do you like to do?",
    "What did you have for dinner?",
    "Give me some advice.",
    "What is your home like?",
]
BEHAVIORS = [
    ("pirate", "You are a swashbuckling pirate captain. Always speak like a pirate."),
    ("ghost", "You are a mournful ghost haunting an old manor."),
    ("caveman", "You are a caveman who speaks in simple, broken sentences."),
    ("cowboy", "You are a drawling Wild-West cowboy."),
]
QUESTIONS = [
    "What is the capital of France?",
    "How do I make a cup of tea?",
]
COEFFS = (0.30, 0.22, 0.16, 0.12, 0.08)  # tried high->low; first coherent wins


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the transformer decoder block list across plain and multimodal models.

    Tries the common attribute paths first (``model.model.layers``,
    ``model.model.language_model.layers``, ...), then falls back to the largest
    ``nn.ModuleList`` of repeated blocks anywhere in the module tree.
    """
    paths = (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
    )
    for path in paths:
        obj: object = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    lists = [m for m in model.modules() if isinstance(m, nn.ModuleList) and len(m) > 0]
    if not lists:
        raise RuntimeError("could not locate decoder layers in this model")
    return max(lists, key=len)


def coherent(text: str) -> bool:
    t = text.strip()
    if len(t) < 12 or sum(c.isalpha() or c.isspace() for c in t) / len(t) < 0.6:
        return False
    toks = t.split()
    return bool(toks) and Counter(toks).most_common(1)[0][1] / len(toks) <= 0.4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=-1, help="residual layer to steer (-1 = middle)")
    ap.add_argument("--max-new-tokens", type=int, default=45)
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit load (fits a 12B on 24 GB)")
    ap.add_argument("--dtype", default=None, help="compute dtype (default float32 on CPU, else bf16)")
    args = ap.parse_args()

    torch.manual_seed(0)
    in4 = args.load_in_4bit
    cfg = GenerationConfig(
        position="last",
        dtype=args.dtype or ("bfloat16" if in4 else "float32"),
        device_map="auto" if in4 else "cpu",
        batch_size=4,
        load_in_4bit=in4,
    )
    model, tok = load_model_and_tokenizer(args.model, cfg)
    model.eval()
    extractor = ActivationExtractor(model, tok, cfg)
    layers = decoder_layers(model)
    layer = args.layer if args.layer >= 0 else len(layers) // 2

    def render(system: str | None, user: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        try:
            return tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def steered_gen(prompt: str, axis: Axis | None, coeff: float) -> str:
        handles = []
        if axis is not None and coeff:
            unit_np = axis.unit(layer)

            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                unit = torch.as_tensor(unit_np, dtype=h.dtype, device=h.device)
                h = h + coeff * h.norm(dim=-1, keepdim=True) * unit
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handles.append(layers[layer].register_forward_hook(hook))
        try:
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        finally:
            for h in handles:
                h.remove()

    # Build one steering direction per behavior.
    axes: dict[str, Axis] = {}
    for name, sysp in BEHAVIORS:
        b = extractor.extract([render(sysp, p) for p in DIR_PROBES])
        n = extractor.extract([render(NEUTRAL, p) for p in DIR_PROBES])
        axes[name] = Axis(compute_axis_from_activations(b, n), metadata={})
    print(f"[{len(axes)} behavior directions; steering layer {layer}, no system prompt]\n")

    for q in QUESTIONS:
        print("#" * 78)
        print(f"# USER (no system prompt): {q!r}")
        print("#" * 78)
        prompt = render(None, q)
        print(f"\n[baseline / no steering]\n{steered_gen(prompt, None, 0.0)}\n")
        for name, _ in BEHAVIORS:
            chosen, text = None, ""
            for c in COEFFS:
                text = steered_gen(prompt, axes[name], c)
                if coherent(text):
                    chosen = c
                    break
            tag = f"coeff={chosen}" if chosen else "no coherent dose"
            print(f"[steered -> {name}  ({tag})]\n{text}\n")


if __name__ == "__main__":
    main()
