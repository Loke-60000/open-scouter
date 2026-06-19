"""Map and steer emotions in a language model.

A separate subsystem from the assistant axis (behaviors). Each emotion gets a
linear residual-stream direction by difference of means; project onto it to map
emotions, or add it to steer (it is an :class:`~scouter.axis.Axis`).

Layout (mirrors ``axis/``):
  * ``catalog``  - Emotion, the default emotion set and probes.
  * ``vectors``  - EmotionVectors (profile / dominant / as_axis / save / load).
  * ``generate`` - build/generate emotion vectors from a model, plus the CLI.
"""

from .catalog import DEFAULT_EMOTIONS, NEUTRAL_PROMPTS, Emotion, load_emotions_json
from .generate import build_emotion_vectors, generate_emotion_vectors, main
from .vectors import EmotionVectors

__all__ = [
    "Emotion",
    "DEFAULT_EMOTIONS",
    "NEUTRAL_PROMPTS",
    "EmotionVectors",
    "load_emotions_json",
    "build_emotion_vectors",
    "generate_emotion_vectors",
    "main",
]
