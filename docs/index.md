---
title: Home
nav_order: 1
---

# open-scouter

Serve a language model while watching, and steering, where it sits on the
assistant axis.

The assistant axis is a direction in a model's residual stream that captures how
"Assistant-like" its current persona is, from Anthropic's research on the
[assistant axis](https://www.anthropic.com/research/assistant-axis)
(paper: [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)).

On every request, `open-scouter` can:

- **Monitor**: report an assistant-likeness score, to catch persona drift.
- **Steer**: nudge the model back toward (or away from) the Assistant persona.
- **Emotions**: map text to emotions and steer the model's emotion, using the
  same activation-vector approach (see the transformer-circuits "Emotion
  Concepts" paper).

It also builds the axis (and emotion vectors) itself from any HuggingFace model,
so you do not need precomputed ones. Everything is behind an OpenAI-compatible
HTTP API, in Docker.

## Documentation

- [Install](install.html): get it running.
- [Usage](usage.html): make an axis, chat, monitor, and steer.

## License

MIT for this repository. The submodules keep their own licenses (assistant-axis:
MIT, vLLM-Hook: Apache-2.0).
