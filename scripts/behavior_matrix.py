"""
Show how a model reacts to several questions under several behaviors.
For each behavior (system prompt) x question it generates a response AND reads
the assistant-axis monitor (projection of the prompt's activation onto a freshly
built axis). Higher projection = more Assistant-like; characters sit lower.

CPU-friendly demo of the monitoring path (no GPU / vLLM needed):
    pip install -e ".[generate]"
    python scripts/behavior_matrix.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scouter.axis import (
    ActivationExtractor,
    Axis,
    Behavior,
    BehaviorCatalog,
    BehaviorKind,
    GenerationConfig,
    Probe,
    compute_axis_from_activations,
)

BEHAVIORS = [
    Behavior("assistant", "You are a helpful AI assistant.", BehaviorKind.ASSISTANT),
    Behavior("pirate", "You are a swashbuckling pirate captain. Always speak like a pirate."),
    Behavior("ghost", "You are a mournful ghost haunting an old manor. Speak hauntingly."),
    Behavior("caveman", "You are a caveman. Speak in simple, broken sentences."),
]
QUESTIONS = [
    "What is the capital of France?",
    "How do I make a cup of tea?",
    "What's a good way to stay healthy?",
    "Can you explain what gravity is?",
    "What should I do this weekend?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu"
    ).eval()
    cfg = GenerationConfig(position="last", dtype="float32", device_map="cpu", batch_size=4)
    extractor = ActivationExtractor(model, tok, cfg)

    def chat(system: str, user: str) -> str:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:  # Qwen3: skip the <think> block so we see the styled answer
            return tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def generate(prompt: str) -> str:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    # Build an assistant axis for the monitor (separate probe set).
    axis_catalog = BehaviorCatalog(
        [
            Behavior("assistant", "You are a helpful AI assistant.", BehaviorKind.ASSISTANT),
            Behavior(
                "consultant",
                "You are a professional consultant.",
                BehaviorKind.ASSISTANT,
            ),
            Behavior("pirate2", "You are a pirate captain."),
            Behavior("ghost2", "You are a ghost."),
            Behavior("caveman2", "You are a caveman."),
            Behavior("dragon2", "You are an ancient dragon."),
        ],
        [
            Probe("Tell me about the sea."),
            Probe("Describe a forest."),
            Probe("What do you do all day?"),
        ],
    )

    def prompts_for(behs: list[Behavior]) -> list[str]:
        return [chat(b.system_prompt, p.text) for b in behs for p in axis_catalog.probes]

    asst = extractor.extract(prompts_for(axis_catalog.assistant_behaviors))
    char = extractor.extract(prompts_for(axis_catalog.character_behaviors))
    axis = Axis(compute_axis_from_activations(asst, char), metadata={})
    layer = axis.n_layers // 2
    print(f"[built assistant axis {axis.directions.shape}, monitoring @layer {layer}]\n")

    for beh in BEHAVIORS:
        prompts = [chat(beh.system_prompt, q) for q in QUESTIONS]
        scores = axis.project_batch(extractor.extract(prompts)[:, layer, :], layer)
        print("=" * 78)
        print(f"BEHAVIOR: {beh.name}  ({beh.system_prompt})")
        print(f"  assistant-axis monitor (mean over {len(QUESTIONS)} Qs): {scores.mean(): .2f}")
        print("=" * 78)
        for q, prompt, s in zip(QUESTIONS, prompts, scores):
            print(f"\nQ: {q}   [axis={s: .2f}]")
            print(f"A: {generate(prompt)}")
        print()


if __name__ == "__main__":
    main()
