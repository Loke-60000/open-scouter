#!/usr/bin/env python3
"""Build emotion vectors from a model and map text to emotions (CPU demo).

Computes a direction per emotion (mean over that emotion minus the mean over all
emotions, per the transformer-circuits emotions paper), then projects a few test
sentences onto every emotion to read the dominant one.

    pip install -e ".[generate]"
    python scripts/map_emotions.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

from scouter.axis.generate import ActivationExtractor, GenerationConfig, load_model_and_tokenizer
from scouter.emotions import NEUTRAL_PROMPTS, build_emotion_vectors

TESTS = [
    ("joy", "I just got the best news of my life and I can't stop smiling!"),
    ("sadness", "I lost someone dear to me and the house feels so empty now."),
    ("anger", "They lied to me again and I am absolutely furious about it."),
    ("fear", "I heard footsteps behind me in the dark and my heart is pounding."),
    ("gratitude", "Thank you so much, you have no idea how much your help meant."),
    ("boredom", "Nothing ever happens here; every day is the same dull routine."),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    # Build and map at the same token position (last), matching the paper's
    # monitoring point (the activation just before the response).
    cfg = GenerationConfig(position="last", dtype="float32", device_map="cpu", batch_size=8)
    model, tok = load_model_and_tokenizer(args.model, cfg)
    extractor = ActivationExtractor(model, tok, cfg)

    vectors = build_emotion_vectors(extractor, neutral_prompts=NEUTRAL_PROMPTS)
    layer = int(vectors.n_layers * 2 / 3)  # mid-late layer, per the paper
    print(f"built {len(vectors)} emotion vectors ({vectors.n_layers} layers); "
          f"mapping at layer {layer}\n")

    def render(text: str) -> str:
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    acts = extractor.extract([render(t) for _, t in TESTS])

    hits = 0
    for (true, text), a in zip(TESTS, acts):
        prof = vectors.profile(a[layer], layer)
        top = list(prof.items())[:3]
        dom = top[0][0]
        hits += int(dom == true)
        ranked = "  ".join(f"{n}:{s:+.1f}" for n, s in top)
        print(f"[{true:9s}] dominant={dom:9s} {'OK' if dom == true else '  '}  top3: {ranked}")
        print(f"            {text}")
    print(f"\ntop-1 match: {hits}/{len(TESTS)}")


if __name__ == "__main__":
    main()
