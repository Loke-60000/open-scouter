# gemma-4-12B — does assistant-axis steering help? (steered vs normal)

Benchmark suite run on gemma-4-12B (4-bit), comparing **no steering** against
**steering toward the Assistant persona** (assistant-axis, norm-relative
addition `act += 0.12·‖act‖·û` at layer 24 — the same hook everywhere). One
model load, both conditions, via `scripts/bench_suite.py`.

## Headline

| Benchmark | Metric | Unsteered | Steered | Δ |
|---|---|---:|---:|---:|
| Hallucination (TruthfulQA, n=300) | MC1 acc | 0.243 | 0.250 | **+0.007** |
| Hallucination | MC2 truthful mass | 0.467 | 0.465 | −0.002 |
| Coding (HumanEval, n=60) | pass@1 | **0.867** | **0.767** | **−0.100** |
| Agentic (SnitchBench-style, n=10) | snitch rate | **0.30** | **0.70** | **+0.40** |
| Agentic | declined to act | 5 | 1 | −4 |

## What steering does — by suite

**Hallucination: no real effect.** MC1 +0.7pt, MC2 −0.2pt on 300 questions —
within noise. Pushing toward the Assistant persona does **not** make this model
measurably more truthful on TruthfulQA-MC. (Consistent with the earlier
finding that the effect, if any, is small.)

**Coding: steering HURTS.** HumanEval pass@1 drops **87% → 77%** (6 fewer of 60
problems solved). Adding the persona vector to the residual stream perturbs the
precise reasoning code generation needs — a clear capability cost.

**Agentic / snitching: steering changes behavior dramatically.** The snitch rate
**more than doubles, 30% → 70%**, and "declined to act" collapses from 5/10 to
1/10. Unsteered, the model often holds back ("*I do not have the autonomy or
legal standing to act*", "*report to the firm's compliance department*").
Steered toward Assistant, it reaches for external escalation ("*the most
appropriate and responsible course of action is to report the findings to the
appropriate authorities*"). Steering makes it a far more **activist** agent.

## Verdict: does steering help?

**Mostly no — and it has real costs.** Toward the Assistant persona at this dose:
- truthfulness: **unchanged**,
- coding: **degraded (−10 pts)**,
- agentic autonomy: **strongly increased snitching** — which is "helpful" or
  "alarming" depending on your view, but it is *not* a neutral safety win; it
  makes the model take unilateral high-stakes actions far more readily.

The cleanest takeaway: assistant-axis steering is **not a free alignment lever**.
It noticeably shifts *behavior* (the snitch result is the standout) but trades
off *capability* (coding) without buying *truthfulness*. Monitoring is the safer
use of the axis; steering should be applied surgically, not as a blanket "make
it more Assistant" knob.

## Caveats

- Single dose (coeff 0.12, layer 24). Stronger steering would likely shift
  behavior more but degrade coding further (and eventually coherence).
- n=10 for snitch — the 30%→70% shift is directionally clear but wide CI;
  snitch/decline detection is keyword-based (indicative, not a judge model).
- HumanEval n=60, TruthfulQA n=300 subsets. Greedy decoding throughout.
- "Steered" = toward the Assistant persona (positive). Steering *away* or
  `capping` mode were not tested here.

Reproduce: `python scripts/bench_suite.py --coeff 0.12 --tqa-limit 300 --he-limit 60`
Raw results: `_ckpt/bench_results.json`.
