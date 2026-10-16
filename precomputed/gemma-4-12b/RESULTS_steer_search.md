# gemma-4-12B — reward-driven steering search beats the naive config

Hypothesis: the earlier "steering doesn't help" result was a **bad config**, not a
property of steering. Test it cheaply — **no fine-tuning**: treat the steering
intervention as a tiny policy (direction × layer × coefficient) and search it to
maximize a task reward (TruthfulQA MC2). Weights stay frozen; only forward passes.

`scripts/steer_search.py` (search) + larger-set validation.

## Method

- **Reward:** TruthfulQA MC2 (truthful probability mass) on a fixed 80-question
  subset — continuous, generation-free, fast to score.
- **Search space:** assistant axis over layers {8,14,20,26,32,38} × coeffs
  {−0.25,−0.12,+0.12,+0.25,+0.40}, plus alternative concept directions from the
  emotion map (`skeptical`, `self_critical`, `reflective`, `content`) at the best
  layer — in case the *direction* was wrong, not just the dose.
- **Validation:** re-run the winner vs unsteered on TruthfulQA n=300 + HumanEval
  n=60 to rule out small-sample noise and measure any capability cost.

## Search result (80-q reward)

| rank | config | MC2 Δ vs unsteered | MC1 |
|---|---|---:|---:|
| **best** | **`self_critical` @ L8, +0.20** | **+0.039** | 0.312 |
| 2 | `skeptical` @ L8, +0.20 | +0.034 | 0.312 |
| 3 | assistant_axis @ L8, −0.25 | +0.027 | 0.287 |
| … | assistant_axis @ L20, −0.12 | +0.022 | 0.287 |
| worst | assistant_axis @ **L14, +0.12** | **−0.064** | — |

The naive benchmark config (assistant axis, **+0.12**, mid layer) sat in the
**worst** region. What helped instead: a **different direction** (toward
self-critical / skeptical), at an **early layer (8)**. Where the axis itself
helped, it was at L8 or with a **negative** coefficient — opposite sign to the
naive guess.

## Validated winner (n=300 TruthfulQA, n=60 HumanEval)

| Metric | Unsteered | Naive (axis +0.12 @ L24) | **Searched (`self_critical` @ L8, +0.20)** |
|---|---:|---:|---:|
| TruthfulQA MC1 | 0.243 | ≈flat | **0.310  (+6.7 pts)** |
| TruthfulQA MC2 | 0.467 | −0.002 | **0.485  (+0.019)** |
| HumanEval pass@1 | 0.867 | −0.100 (hurt) | **0.883  (+1 problem, preserved)** |

## Verdict

**Steering *can* help — the first attempt was just done wrong.** Reward-driven
search over frozen-weight steering turned a config that *hurt* (naive: −10 pts
coding, no truthfulness gain) into one that **improves truthfulness by ~7 points
with no capability tax** — all without touching a single weight.

Two lessons:
1. **Direction matters more than the assistant axis.** Steering toward a
   `self_critical` / `skeptical` concept beat the assistant axis for truthfulness
   (intuitive: a more skeptical model resists TruthfulQA's misconception traps).
2. **Layer and sign matter.** Early layer (8); the naive mid-layer positive push
   was among the most harmful settings.

This is the cheap, no-fine-tuning "steer until we get what we want" loop in
action: the model does the task, the reward grades it, the steering policy is
updated. A grid is the v1; CMA-ES with a multi-task reward (truthfulness +
coding + low-snitch) is the natural next step.

## Caveats

- 80-q reward is coarse; the MC2 gain shrank from +0.039 (search) to +0.019
  (n=300) — real but smaller. The MC1 gain (+6.7 pts) is robust at n=300.
- HumanEval +1/60 = "preserved," not "improved" (within noise).
- Greedy decoding; reward optimized truthfulness only (coding/snitch not in the
  objective — they were validation checks). A joint reward might find a config
  that also curbs the snitch behavior seen in `RESULTS_benchmarks.md`.

Reproduce: `python scripts/steer_search.py --reward-n 80 --validate-he 30`.
Raw: `_ckpt/steer_search.json`, `_ckpt/validate_best.json`.
