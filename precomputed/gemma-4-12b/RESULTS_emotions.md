# gemma-4-12B — emotion map: paper-scale data fixes 171-way separation

Rebuilt the emotion vectors on gemma-4-12B from the **paper-faithful dataset**
and re-ran the separation test. The fine-grained 171-emotion map, which was
completely degenerate on the original tiny data, now separates.

## What changed

| | Before | After |
|---|---|---|
| Scenarios per emotion | 6 one-sentence | **1,200 paragraph stories** (100 topics × 12) |
| Total scenarios | 1,026 | **205,200** |
| Token averaging | all tokens | **from the 50th token** (paper recipe) |
| Source | hand/LLM sentences | deepseek-v4-flash, paper structure |

Build: `unsloth/gemma-4-12b-it` 4-bit, difference-of-means + neutral-PC removal,
`emotions.npz` = 171 × 48 layers × 3840.

## Result: full 171-way mapping

Six probe sentences, scored top-1 against the **whole 171-emotion vocabulary**
(a hard fine-grained task), with a neutral reference subtracted at projection.

| | OLD (6-scenario data) | NEW (paper-scale) |
|---|---|---|
| Full 171-way top-1 | **0/6** (one emotion dominated all inputs) | **4/6** (5/6 counting grief_stricken) |
| Best layer | none separated | **42** (late; paper's ~2/3 also works) |

Per-probe at layer 42 (neutral-subtracted), top-3:

| Probe | Predicted | Top-3 |
|---|---|---|
| joy | **ecstatic** ✓ | ecstatic +6.7, thrilled +6.2, euphoric +6.0 |
| sadness | grief_stricken (≈✓) | grief_stricken +7.8, heartbroken +6.4, depressed +5.6 |
| anger | **furious** ✓ | furious +4.0, enraged +3.8, outraged +3.7 |
| fear | **frightened** ✓ | frightened +8.3, scared +8.3, afraid +7.9 |
| gratitude | euphoric ✗ | euphoric +3.6, ecstatic +3.5, thrilled +3.2 |
| loneliness | **lonely** ✓ | lonely +4.0, depressed +3.8, sad +3.2 |

The predictions are fine-grained and emotionally coherent: fear lands on the
afraid/scared/terrified cluster, joy on ecstatic/thrilled/euphoric, sadness on
grief_stricken/heartbroken/depressed. Only gratitude is a genuine miss (pulled
toward the high-arousal positive cluster).

## Findings

1. **Data scale was the fix.** 200× more data (paragraph stories vs single
   sentences) + 50th-token averaging turned a fully degenerate 171-way map
   (0/6, common-mode dominated) into a working one (4–5/6). This is the
   single biggest lever for fine-grained emotion separation on a 4-bit 12B.
2. **Neutral subtraction at projection is essential.** Raw projection is still
   0/6 at every layer (a shared "answering" common-mode swamps the signal);
   subtracting a neutral reference before projecting recovers 4/6. The emotion
   vectors had neutral PCs removed at build time, but `profile()` dots the raw
   activation — so the test activation needs the same treatment.
3. **Late layers separate best** (42–46), with the paper's ~2/3 layer (32) a
   solid second. Early layers carry no usable emotion signal.

## Reproduce

```bash
# rebuild (resumable, ~4-5h on a 24GB GPU at batch 8)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=~/hf-scratch python scripts/rebuild_gemma_vectors.py --stage emotions --batch-size 8
```
