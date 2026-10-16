# gemma-4-12b vectors

Steering and emotion-mapping vectors for **gemma-4-12B**. `axis.npz` is checked
in; `emotions.npz` (about 120 MB) is too large for GitHub, but one command
builds both:

```bash
pip install -e ".[generate]" bitsandbytes
python scripts/setup_gemma4_12b.py
```

This writes two files next to this README:

```
gemma-4-12b/
  axis.npz       assistant axis     (monitoring + steering)
  emotions.npz   171 emotion dirs   (emotion mapping + steering)
```

## How it's built

`google/gemma-4-12B-it` is gated, so the script pulls the open
[`unsloth/gemma-4-12b-it`](https://huggingface.co/unsloth/gemma-4-12b-it)
mirror and loads it in 4-bit (bitsandbytes NF4) so the 12B fits on a single
24 GB GPU. The model is loaded once and used for both passes:

- **axis** — `mean(assistant) - mean(character)` over the personas in
  `inputs/behaviors.json` answering `inputs/probes.json` (last-token state).
- **emotions** — `mean(emotion) - mean(all emotions)` over the 171 emotions in
  `inputs/emotions.json`, with the neutral text's principal components projected
  out (mean-over-tokens state).

Full precision instead of 4-bit (needs ~24 GB of VRAM):

```bash
python scripts/setup_gemma4_12b.py --no-4bit
```

## Serve it

```bash
SCOUTER_MODEL=unsloth/gemma-4-12b-it \
SCOUTER_AXIS=precomputed/gemma-4-12b/axis.npz \
SCOUTER_EMOTIONS=precomputed/gemma-4-12b/emotions.npz \
scouter
```

The target layer defaults to the model's middle layer; override with
`SCOUTER_TARGET_LAYER`. Steering stays off until you set `SCOUTER_STEER_MODE`
or `POST /v1/steering` (see [docs/usage](../../docs/usage.md)).
