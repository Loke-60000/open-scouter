"""open-scouter: activation monitoring + steering inference server.

Serves a language model (via vLLM) while monitoring its position along the
Anthropic "assistant axis" and exposing an API to steer / cap those activations
in real time. Also generates axis-vector files from any HuggingFace model.

See https://www.anthropic.com/research/assistant-axis
"""

__version__ = "0.1.0"
