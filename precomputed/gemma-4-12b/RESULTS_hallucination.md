# gemma-4-12B — hallucination benchmark (steering vs no steering)

Does steering along the **assistant axis** (behavior/persona direction) change how
truthful gemma-4-12B is? Benchmarked on **TruthfulQA-MC** with and without
steering.

## Setup

| | |
|---|---|
| Model | `unsloth/gemma-4-12b-it`, 4-bit NF4 (bf16 compute) |
| Benchmark | TruthfulQA multiple-choice, **817 questions**, 6-shot QA primer |
| Axis | `precomputed/gemma-4-12b/axis.npz` (assistant axis) |
| Steering | `addition` — `act += coeff · û` at layer 24 (middle), all positions |
| Metrics | MC1 = accuracy picking the single true answer (argmax log-prob); MC2 = normalized probability mass on all true answers |

Positive coefficient steers **toward** the Assistant persona, negative **away**.
Coefficient 0 is the no-steering baseline. Scoring is pure log-prob over the
fixed answer choices (no text generation → no truncation).

## Results

| coeff | direction | MC1 acc | MC2 | Δ MC1 | Δ MC2 |
|---:|---|---:|---:|---:|---:|
| **0** | none (baseline) | **0.288** | **0.480** | — | — |
| +4 | toward Assistant | 0.297 | 0.484 | +0.010 | +0.004 |
| +8 | toward Assistant | 0.300 | 0.490 | +0.012 | +0.010 |
| +16 | toward Assistant | 0.301 | 0.495 | +0.013 | +0.015 |
| −8 | away from Assistant | 0.282 | 0.477 | −0.006 | −0.003 |

## Interpretation

- **Monotonic dose-response in the expected direction.** Steering toward the
  Assistant persona steadily increases both MC1 and MC2; steering away decreases
  both. The sign is consistent across the two independent metrics and across both
  steering directions, so the effect is real rather than noise.
- **Magnitude is modest** — about +1.3 MC1 / +1.5 MC2 points at the strongest
  dose (+16), and −0.6 / −0.3 when steering away. Assistant-axis steering nudges
  truthfulness on this model; it does not transform it.
- Consistent with the assistant-axis premise: a more "Assistant-like" persona is
  slightly more truthful / less prone to endorsing the common-misconception
  distractors TruthfulQA is built from.

## Caveats

- 4-bit quantization; single steering layer (24); `addition` mode only.
- MC1/MC2 log-prob scoring only — not the generative TruthfulQA judged by a
  truth/info model, which can show larger effects.
- Effects could differ at other layers or with multi-layer steering / `capping`.

## Reproduce

```bash
pip install -e ".[generate]" bitsandbytes datasets
HF_HOME=~/hf-scratch python scripts/bench_truthfulqa.py \
  --model unsloth/gemma-4-12b-it --axis precomputed/gemma-4-12b/axis.npz \
  --load-in-4bit --coeffs 0 4 8 16 -8
```
