<p align="center">
  <img src="docs/assets/github-header-banner.png" alt="open-scouter" width="100%">
</p>

<p align="center"><em>Read the persona and emotion hidden in a language model's activations, and steer them.</em></p>

Serve a language model while watching, and steering, where it sits on the
assistant axis.

The assistant axis is a direction in a model's residual stream that captures how
"Assistant-like" its current persona is, from Anthropic's research on the
[assistant axis](https://www.anthropic.com/research/assistant-axis)
(paper: [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)).
It is computed as the difference of mean activations between the Assistant
persona and other character personas:

```
axis = mean(assistant activations) - mean(character activations)
```

On every request, `open-scouter` can:

1. Monitor: project the residual stream onto the axis and report an
   assistant-likeness score (to catch persona drift).
2. Steer: add the axis direction to the residual stream (addition), or clip the
   projection at a threshold (capping), to keep the model anchored.

It can also map and steer emotions, using the same activation-vector approach
on per-emotion directions (see the transformer-circuits "Emotion Concepts"
paper, https://transformer-circuits.pub/2026/emotions/index.html).

Beyond hand-picked directions, it learns a dictionary of the model's own
features with a sparse autoencoder, the method behind Anthropic's
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
and its Golden Gate Bridge feature: label features by the tokens they fire on,
see which fire on each reply, and clamp one on to steer.

It generates the axis, emotion vectors and feature dictionary itself from any
HuggingFace model, so no precomputed files are required. Everything is behind one OpenAI-compatible
HTTP API, in Docker, on a GPU (vLLM with the IBM vLLM-Hook plugin).

## Documentation

Full docs are a GitHub Pages site built from the [`docs/`](docs/) folder:
https://loke-60000.github.io/open-scouter/

- [Install](docs/install.md)
- [Usage](docs/usage.md)

## Quickstart

```bash
# pull the prebuilt vLLM image
docker pull ghcr.io/loke-60000/open-scouter:latest

# or run from source (GPU)
git clone --recurse-submodules https://github.com/Loke-60000/open-scouter open-scouter
cd open-scouter
docker compose -f docker/docker-compose.yml up vllm
```

For the default model (Qwen3-0.6B) the assistant axis ships precomputed in
[`precomputed/`](precomputed/), so monitoring works with no generation step:

```bash
SCOUTER_MODEL=Qwen/Qwen3-0.6B \
SCOUTER_AXIS=precomputed/qwen3-0.6b/axis.npz \
scouter
```

Build vectors for another model, or try the CPU demos:

```bash
pip install -e ".[generate]"
scouter-generate --model <model> --out axis.npz
# emotion vectors (171 emotions; use a capable model for clean separation)
scouter-emotions --model <model> --out emotions.npz --emotions inputs/emotions.json
# sparse-autoencoder feature dictionary at the middle layer
scouter-features --model <model> --corpus inputs/emotions.json --out features.npz
python scripts/behavior_matrix.py --model Qwen/Qwen3-0.6B   # monitor across behaviors
```

## License

MIT for this repository. The submodules keep their own licenses (assistant-axis:
MIT, vLLM-Hook: Apache-2.0). See [LICENSE](LICENSE).
