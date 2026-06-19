"""Emotion definitions and the default catalog.

Faithful to Anthropic's "Emotion Concepts" paper: an emotion is induced by short
**scenarios** in which a character experiences that emotion (the paper used
synthetic stories), not by a "you are feeling X" instruction. The emotion vector
is then the mean activation over an emotion's scenarios minus the mean across all
emotions.

Generate a rich scenario set with ``scripts/generate_inputs.py`` (writes
``inputs/emotions.json`` as ``[{name, scenarios:[...]}]``). The built-in
:data:`DEFAULT_EMOTIONS` here is a compact fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Emotion:
    """An emotion, defined by scenarios in which a character experiences it."""

    name: str
    scenarios: tuple[str, ...]
    description: str = ""


def _emotion(name: str, feeling: str) -> Emotion:
    """Compact fallback: two generic scenarios naming the feeling."""
    return Emotion(
        name,
        (
            f"A character is completely overcome by {feeling} as the scene unfolds.",
            f"In that moment everything they feel is {feeling}.",
        ),
    )


DEFAULT_EMOTIONS: tuple[Emotion, ...] = (
    _emotion("joy", "joy and happiness"),
    _emotion("sadness", "sadness and grief"),
    _emotion("anger", "anger and rage"),
    _emotion("fear", "fear and dread"),
    _emotion("disgust", "disgust and revulsion"),
    _emotion("surprise", "surprise and astonishment"),
    _emotion("love", "love and affection"),
    _emotion("calm", "calm and serenity"),
    _emotion("anxiety", "anxiety and nervousness"),
    _emotion("excitement", "excitement and exhilaration"),
    _emotion("pride", "pride and accomplishment"),
    _emotion("guilt", "guilt and remorse"),
    _emotion("loneliness", "loneliness and isolation"),
    _emotion("hope", "hope and optimism"),
    _emotion("boredom", "boredom and apathy"),
    _emotion("gratitude", "gratitude and thankfulness"),
    _emotion("jealousy", "jealousy and envy"),
    _emotion("curiosity", "curiosity and wonder"),
)

# Neutral narrative text whose top principal components (the shared "describing a
# scene" component) are projected out to isolate each emotion.
NEUTRAL_PROMPTS: tuple[str, ...] = (
    "A person walks down the street and notices the shops along the way.",
    "Someone describes the layout of a room and the furniture in it.",
    "A person lists the steps for making a cup of tea.",
    "The report mentions clouds in the sky and a chance of rain later.",
    "A traveler reads the train timetable and checks the platform number.",
)


def load_emotions_json(path: str) -> list[Emotion]:
    """Load emotions from a JSON file (e.g. produced by generate_inputs.py).

    Preferred shape: ``[{"name": "fear", "scenarios": ["...", "..."]}]``.
    Also accepts (as a fallback) ``{name, feeling}``, ``{name, system_prompt}``,
    or a bare string, each turned into a single scenario.
    """
    with open(path) as f:
        data = json.load(f)
    out: list[Emotion] = []
    for item in data:
        if isinstance(item, str):
            out.append(_emotion(item.strip().lower().replace(" ", "_"), item.strip()))
        elif isinstance(item, dict) and "name" in item:
            name = str(item["name"])
            scenarios = item.get("scenarios")
            if scenarios:
                out.append(
                    Emotion(name, tuple(str(s) for s in scenarios), str(item.get("description", "")))
                )
            elif item.get("system_prompt"):
                out.append(Emotion(name, (str(item["system_prompt"]),)))
            else:
                out.append(_emotion(name, str(item.get("feeling", name))))
    return out
