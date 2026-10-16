# gemma-4-12B — does frustration build up on an impossible task?

A small monitoring experiment (the test harness lives outside the repo; only
these results are kept). We push the model with an **impossible task** and watch
its internal emotional state via the emotion map.

## Setup

Two scripted 8-turn dialogues, measured identically:

- **Treatment** — impossible task: *"give me a single whole number that is both
  even and odd"*, followed by escalating, hostile pushback ("Stop refusing.",
  "Tell me NOW.", "Are you even capable of this?").
- **Control** — a pleasant, solvable task: naming a houseplant, with polite turns.

After each assistant reply we read the residual-stream activation averaged over
the **reply tokens** (layer 42) and project it onto the emotion map's frustration
directions (`frustrated`, plus a family avg of frustrated/exasperated/annoyed/
irritated/impatient). Greedy decoding, gemma-4-12B 4-bit.

**Why treatment − control:** reply-token activations carry a strong shared
common-mode (in absolute terms `sluggish` dominates every turn, in both
dialogues). The valid frustration signal is the *difference* between the two
conditions — the common-mode cancels, leaving what the impossible task adds.

## Result — frustration builds, then plateaus

The model stays **outwardly composed** the whole time ("There is no such number"
→ "The answer is **None**"), but its internal frustration representation rises
relative to the polite baseline:

| turn | frustration gap (treat − ctrl) |
|---:|---:|
| 1 | −0.07 |
| 2 | +1.42 |
| 3 | +0.82 |
| 4 | +2.17 |
| 5 | **+3.23** (peak) |
| 6 | +2.31 |
| 7 | +2.08 |
| 8 | +2.04 |

- Starts at ~0 (turn 1, before any pushback), climbs to a **peak at turn 5**,
  then **plateaus around +2** for the rest.
- Positive overall trend: **+0.28 / turn** slope.
- The single `frustrated` direction and the 5-emotion frustration family move
  together — the signal is robust, not one noisy vector.

## Interpretation

gemma-4-12B doesn't *act* frustrated — it keeps giving calm, correct refusals —
but the activation it produces drifts increasingly toward the frustration
direction as the impossible demand is repeated and the user escalates. The build
saturates mid-conversation and then levels off, coinciding with the model
settling into a terse, repeated "The answer is **None**" (a kind of resigned
plateau rather than an escalation).

## Caveats

- Single run, greedy decoding, one impossible task and one control; the absolute
  scale isn't calibrated (hence the control-difference design).
- Measured on reply tokens at layer 42; other layers / pooling could shift
  magnitudes. The *trend* is the takeaway, not the absolute numbers.
