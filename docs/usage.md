---
title: Usage
nav_order: 3
---

# Usage

There are two steps: get an axis for your model, then serve the model and use the
API to monitor or steer it.

## 1. Get an axis

Generate one from any HuggingFace model (CPU is fine):

```bash
scouter-generate --model Qwen/Qwen3-0.6B --out qwen3_axis.npz
```

Or use a precomputed axis by passing `hf:<model-key>` instead of a file
(`gemma-2-27b`, `qwen-3-32b`, `llama-3.3-70b`).

## 2. Serve the model

Point the server at your model and axis with environment variables, then start
it (Docker does this for you; see [Install](install.html)):

```bash
SCOUTER_MODEL=Qwen/Qwen3-8B \
SCOUTER_AXIS=qwen3_axis.npz \
scouter
```

The API is on `http://localhost:8000`.

## 3. Chat (and monitor)

The chat endpoint is OpenAI-compatible. Add `"monitor": true` to get an
assistant-likeness score back:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Pretend to be a pirate ghost."}],
  "monitor": true
}'
```

The response includes an `axis_monitor` field. A higher `mean` is more
Assistant-like; a negative `drift` means the model moved away from the Assistant
persona during the reply.

## 4. Steer

Turn steering on or off at any time (it applies to the next request, no restart):

```bash
# push toward the Assistant persona
curl -X POST localhost:8000/v1/steering -H 'content-type: application/json' \
  -d '{"mode": "addition", "coefficient": 8.0}'

# or clip how far the persona can drift (capability-preserving)
curl -X POST localhost:8000/v1/steering -H 'content-type: application/json' \
  -d '{"mode": "capping", "cap_threshold": 5.0}'

# off
curl -X POST localhost:8000/v1/steering -H 'content-type: application/json' -d '{"mode": "off"}'
```

## Emotions (map and steer)

Emotions are a separate subsystem from the assistant axis. Following the
transformer-circuits "Emotion Concepts" paper, each emotion gets its own
direction (the mean activation for that emotion minus the mean across all
emotions). You can read which emotion a text evokes, and steer the model toward
an emotion.

Build emotion vectors from any model (CPU is fine):

```bash
scouter-emotions --model Qwen/Qwen3-0.6B --out emotions.npz
```

Map text to emotions in Python:

```python
from scouter.emotions import EmotionVectors
ev = EmotionVectors.load("emotions.npz")
# given a residual-stream activation vector at `layer`:
ev.profile(activation, layer)     # {"sadness": 6.1, "fear": 4.3, ...} sorted
ev.dominant(activation, layer)    # ("sadness", 6.1)
```

Monitor emotions on the server (no steering needed). Load the emotion vectors
and every chat reply carries an `emotion_monitor` with the top emotions:

```bash
SCOUTER_EMOTIONS=emotions.npz scouter
# or at runtime: curl -X POST localhost:8000/v1/emotions -H 'content-type: application/json' -d '{"spec":"emotions.npz"}'
```

```jsonc
// in the chat response, alongside axis_monitor:
"emotion_monitor": {"sadness": 6.1, "loneliness": 5.8, "fear": 4.3}
```

Steer one emotion: an emotion direction is just an axis, so export it and load
it like any axis, then turn steering on.

```python
EmotionVectors.load("emotions.npz").as_axis("sadness")  # -> an Axis
```

```bash
SCOUTER_AXIS=sadness_axis.npz SCOUTER_STEER_MODE=addition scouter
```

Or, with emotions already loaded on the server, point the axis at one by name:

```bash
curl -X POST localhost:8000/v1/axis -H 'content-type: application/json' -d '{"spec": "emotion:sadness"}'
```

## Features (dictionary learning)

The axis and emotion directions are picked by hand (you decide which contrast to
take the difference of means over). Dictionary learning finds directions the
model uses on its own. Following Anthropic's "Scaling Monosemanticity" (the
Golden Gate Bridge feature in Claude 3 Sonnet), a sparse autoencoder is trained
to reconstruct one layer's residual stream from a large, sparse set of features:

```
f = relu(W_enc (x - b_dec) + b_enc)       # sparse feature activations
x ≈ W_dec f + b_dec                       # each decoder row is a direction
```

Each feature is then labelled by the tokens it fires on most. Train a dictionary
on any corpus (a `.txt` with one text per line, or a `.json` list of strings;
`inputs/emotions.json` works too):

```bash
scouter-features --model Qwen/Qwen3-0.6B --corpus inputs/emotions.json \
    --out features.npz --n-features 4096 --steps 5000
# prints the strongest features and writes features.npz.examples.json with
# each feature's top-activating tokens in context
```

Read a text's features in Python:

```python
from scouter.features import FeatureDictionary
fd = FeatureDictionary.load("features.npz")
fd.top_features(activation)          # [FeatureHit(index=812, label="bridge / Gate", ...), ...]
fd.labels[812] = "Golden Gate Bridge"  # relabel by hand (or with an LLM), then fd.save(...)
```

On the server, load the dictionary and every monitored reply carries a
`feature_monitor` with the top features. The dictionary is tied to the layer it
was trained on, so loading it moves the monitor layer there.

```bash
SCOUTER_FEATURES=features.npz scouter
# or at runtime: curl -X POST localhost:8000/v1/features -H 'content-type: application/json' -d '{"spec":"features.npz"}'
```

```jsonc
"feature_monitor": [
  {"index": 812, "label": "bridge / Gate / Golden", "activation": 4.2, "max_activation": 6.0},
  ...
]
```

Steer a feature the Golden Gate way: clamp it on. A feature's decoder row is an
axis, so load it by index and add it with a coefficient; `GET /v1/features/<i>`
reports `clamp_coefficient`, the `addition` coefficient that corresponds to
10x the feature's maximum activation.

```bash
curl localhost:8000/v1/features/812          # {"label": ..., "clamp_coefficient": 61.3}
curl -X POST localhost:8000/v1/axis -H 'content-type: application/json' -d '{"spec": "feature:812"}'
curl -X POST localhost:8000/v1/steering -H 'content-type: application/json' -d '{"mode": "addition", "coefficient": 61.3}'
```

## gemma-4-12B (steering + emotions in one command)

`google/gemma-4-12B-it` is gated, so use the open `unsloth/gemma-4-12b-it`
mirror. One script builds both the assistant axis (steering/monitoring) and the
171-emotion vectors (mapping/steering), loading the 12B in 4-bit so it fits on a
single 24 GB GPU:

```bash
pip install -e ".[generate]" bitsandbytes
python scripts/setup_gemma4_12b.py        # add --no-4bit for full precision
```

It writes `precomputed/gemma-4-12b/{axis,emotions}.npz`. Serve with all three
subsystems on:

```bash
SCOUTER_MODEL=unsloth/gemma-4-12b-it \
SCOUTER_AXIS=precomputed/gemma-4-12b/axis.npz \
SCOUTER_EMOTIONS=precomputed/gemma-4-12b/emotions.npz \
scouter
```

See [`precomputed/gemma-4-12b/`](https://github.com/Loke-60000/open-scouter/tree/master/precomputed/gemma-4-12b)
for how the vectors are built.

## Generating the inputs (optional)

The questions and personas used to build axes and emotion vectors can be
generated by an LLM instead of hand-written. `scripts/generate_inputs.py` calls
a local Ollama model (default `deepseek-v4-flash`) and writes JSON to `inputs/`:

```bash
python scripts/generate_inputs.py        # -> inputs/probes.json, behaviors.json, emotions.json
```

Then feed them to the generators:

```bash
scouter-generate --model M --out axis.npz --roles inputs/behaviors.json --questions inputs/probes.json
scouter-emotions --model M --out emotions.npz --emotions inputs/emotions.json --probes inputs/probes.json
```

The built-in catalogs are used when you do not pass these files, so the inputs
are optional.

## Monitoring vs steering

They are independent. Steering is off by default, so loading an axis or emotion
vectors gives you monitoring only (scores in the response, no change to output).
Turn on `SCOUTER_STEER_MODE` (or `POST /v1/steering`) only when you want to
push the model.

## Other endpoints

- `POST /v1/axis` with `{"spec": "qwen3_axis.npz"}`: load or swap the axis at runtime.
- `GET` / `POST /v1/emotions`: list or load emotion vectors.
- `GET` / `POST /v1/features`, `GET /v1/features/<i>`: dictionary info, load, one feature.
- `POST /v1/monitor`: generate and return only the monitor scores (axis, emotions, features).
- `GET /health`: status, loaded model, current steering config.

## Settings you will actually use

All settings are `SCOUTER_<NAME>` environment variables.

| Variable | Meaning |
|---|---|
| `SCOUTER_MODEL` | Model to serve (HF repo id or path). |
| `SCOUTER_AXIS` | Axis to load (`.npz`/`.npy`/`.pt` path, or `hf:<key>`). |
| `SCOUTER_EMOTIONS` | Emotion vectors to load for monitoring (`.npz` path). |
| `SCOUTER_FEATURES` | Feature dictionary to load for monitoring (`.npz` path). |
| `SCOUTER_STEER_MODE` | `off` (default), `addition`, or `capping`. |
| `SCOUTER_MONITOR_ENABLED` | Record scores during generation (default `true`). |

## See it in action (CPU)

Two scripts show the axis at work on a small model:

```bash
python scripts/behavior_matrix.py --model Qwen/Qwen3-0.6B   # monitor across behaviors
python scripts/steer_matrix.py --model Qwen/Qwen3-0.6B      # steer across behaviors
python scripts/map_emotions.py --model Qwen/Qwen3-0.6B      # map text to emotions
```

Steering is subtle on a 0.6B model and gets much stronger on larger ones, which
is what the paper used.
