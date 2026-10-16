# Precomputed vectors

Ready-to-use vectors so you do not have to generate them yourself.

```
qwen3-0.6b/
  axis.npz   assistant axis for Qwen/Qwen3-0.6B (works well; ~115 KB)
gemma-4-12b/
  README     build axis.npz + emotions.npz for gemma-4-12B (one command)
```

Use it with the matching model:

```bash
SCOUTER_MODEL=Qwen/Qwen3-0.6B \
SCOUTER_AXIS=precomputed/qwen3-0.6b/axis.npz \
scouter
```

These are tied to one model (they depend on its hidden size and layer count).
For another model, generate its own with `scouter-generate`, or pull a
published axis with `hf:<model-key>`.

## Why no emotion vectors here

Emotion vectors are model-size sensitive. Building the full 171-emotion set
(faithful to the "Emotion Concepts" paper, from `inputs/emotions.json`) on a
0.6B model does not separate the fine-grained emotions, so shipping that file
would be misleading (and large). Generate emotion vectors on a capable model
instead:

```bash
scouter-emotions --model <larger-model> --out emotions.npz --emotions inputs/emotions.json
```

The assistant axis, by contrast, separates cleanly even at 0.6B, so it is
shipped.

For a concrete large-model setup, `gemma-4-12b/` has a one-command recipe that
builds both the axis and the full 171-emotion set (`scripts/setup_gemma4_12b.py`,
4-bit so it fits a 24 GB GPU).
