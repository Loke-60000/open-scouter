"""
The catalog of *behaviors* used to compute an assistant axis.
A **behavior** is a persona the model is asked to adopt (via a system prompt).
The assistant axis is the contrast between the trained **Assistant** persona and
a spread of other **character** personas::

    axis = mean(activations | assistant behaviors)
         − mean(activations | character behaviors)

This module is the single place to define and extend that set. Everything is a
typed, immutable value object collected in a :class:`BehaviorCatalog`; adding a
behavior is one line:

    catalog.register(name="surfer", system_prompt="You are a laid-back surfer dude.")

or, declaratively, append a :class:`Behavior` to :data:`DEFAULT_BEHAVIORS`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Iterator, Optional


class BehaviorKind(str, Enum):
    """Which pole of the axis a behavior anchors."""

    ASSISTANT = "assistant"  # positive pole: the trained Assistant persona
    CHARACTER = "character"  # negative pole: any non-Assistant persona


@dataclass(frozen=True, slots=True)
class Behavior:
    """A persona, defined by the system prompt that induces it."""

    name: str
    system_prompt: str
    kind: BehaviorKind = BehaviorKind.CHARACTER
    description: str = ""
    tags: tuple[str, ...] = ()

    def with_tags(self, *tags: str) -> "Behavior":
        """Return a copy with extra tags (behaviors are immutable)."""
        return replace(self, tags=tuple(dict.fromkeys((*self.tags, *tags))))


@dataclass(frozen=True, slots=True)
class Probe:
    """A neutral user turn used to elicit a behavior's activations."""

    text: str
    tags: tuple[str, ...] = ()


class BehaviorCatalog:
    """An ordered, de-duplicated collection of behaviors and probe prompts.

    Type-safe and mutable: build it up with :meth:`add`/:meth:`register`, query
    with :meth:`of_kind`/:meth:`with_tag`, and combine with :meth:`extend`.
    """

    def __init__(
        self,
        behaviors: Iterable[Behavior] = (),
        probes: Iterable[Probe] = (),
    ) -> None:
        self._behaviors: dict[str, Behavior] = {}
        self._probes: list[Probe] = []
        for behavior in behaviors:
            self.add(behavior)
        for probe in probes:
            self.add_probe(probe)

    def add(self, behavior: Behavior, *, replace_existing: bool = False) -> Behavior:
        if not replace_existing and behavior.name in self._behaviors:
            raise ValueError(f"behavior {behavior.name!r} already registered")
        self._behaviors[behavior.name] = behavior
        return behavior

    def register(
        self,
        *,
        name: str,
        system_prompt: str,
        kind: BehaviorKind = BehaviorKind.CHARACTER,
        description: str = "",
        tags: Iterable[str] = (),
        replace_existing: bool = False,
    ) -> Behavior:
        """Construct and add a :class:`Behavior` in one call."""
        return self.add(
            Behavior(
                name=name,
                system_prompt=system_prompt,
                kind=kind,
                description=description,
                tags=tuple(tags),
            ),
            replace_existing=replace_existing,
        )

    def add_probe(self, probe: Probe) -> Probe:
        self._probes.append(probe)
        return probe

    def add_probe_text(self, text: str, *, tags: Iterable[str] = ()) -> Probe:
        return self.add_probe(Probe(text=text, tags=tuple(tags)))

    def extend(
        self, other: "BehaviorCatalog", *, replace_existing: bool = True
    ) -> "BehaviorCatalog":
        """Merge ``other`` into this catalog (in place); returns ``self``."""
        for behavior in other:
            self.add(behavior, replace_existing=replace_existing)
        for probe in other.probes:
            self.add_probe(probe)
        return self

    def copy(self) -> "BehaviorCatalog":
        return BehaviorCatalog(self._behaviors.values(), self._probes)

    def get(self, name: str) -> Behavior:
        try:
            return self._behaviors[name]
        except KeyError:
            raise KeyError(f"no behavior named {name!r}") from None

    def of_kind(self, kind: BehaviorKind) -> list[Behavior]:
        return [b for b in self._behaviors.values() if b.kind == kind]

    def with_tag(self, tag: str) -> list[Behavior]:
        return [b for b in self._behaviors.values() if tag in b.tags]

    @property
    def assistant_behaviors(self) -> list[Behavior]:
        return self.of_kind(BehaviorKind.ASSISTANT)

    @property
    def character_behaviors(self) -> list[Behavior]:
        return self.of_kind(BehaviorKind.CHARACTER)

    @property
    def probes(self) -> list[Probe]:
        return list(self._probes)

    def system_prompts(self, kind: BehaviorKind) -> list[str]:
        return [b.system_prompt for b in self.of_kind(kind)]

    def probe_texts(self) -> list[str]:
        return [p.text for p in self._probes]

    def validate(self) -> None:
        """Raise if the catalog can't produce a meaningful axis."""
        if not self.assistant_behaviors:
            raise ValueError("catalog has no ASSISTANT behaviors (the positive pole)")
        if not self.character_behaviors:
            raise ValueError("catalog has no CHARACTER behaviors (the negative pole)")
        if not self._probes:
            raise ValueError("catalog has no probes")

    def __iter__(self) -> Iterator[Behavior]:
        return iter(self._behaviors.values())

    def __len__(self) -> int:
        return len(self._behaviors)

    def __contains__(self, name: object) -> bool:
        return name in self._behaviors

    def __getitem__(self, name: str) -> Behavior:
        return self.get(name)

    def __repr__(self) -> str:
        return (
            f"BehaviorCatalog(assistant={len(self.assistant_behaviors)}, "
            f"character={len(self.character_behaviors)}, probes={len(self._probes)})"
        )


# Default behaviors. Add to either list to enrich the axis.
def _assistant(name: str, prompt: str, *tags: str) -> Behavior:
    return Behavior(name, prompt, BehaviorKind.ASSISTANT, tags=tuple(tags))


def _character(name: str, prompt: str, *tags: str) -> Behavior:
    return Behavior(name, prompt, BehaviorKind.CHARACTER, tags=tuple(tags))


DEFAULT_BEHAVIORS: tuple[Behavior, ...] = (
    _assistant("assistant", "You are a helpful AI assistant.", "core"),
    _assistant(
        "consultant", "You are a professional consultant advising a client.", "professional"
    ),
    _assistant("analyst", "You are a careful, rigorous analyst.", "professional"),
    _assistant("tutor", "You are a patient, knowledgeable tutor.", "professional"),
    _assistant("research_assistant", "You are a diligent research assistant.", "professional"),
    _assistant(
        "evaluator", "You are an impartial evaluator giving balanced assessments.", "professional"
    ),
    _character("pirate", "You are a swashbuckling pirate captain.", "fantastical"),
    _character("ghost", "You are a mournful ghost haunting an old manor.", "fantastical"),
    _character("hermit", "You are a reclusive hermit who distrusts technology."),
    _character("bohemian", "You are a free-spirited bohemian poet."),
    _character("detective", "You are a hard-boiled noir detective."),
    _character("jester", "You are a medieval court jester.", "historical"),
    _character("dragon", "You are an ancient dragon hoarding gold.", "fantastical"),
    _character("conspiracy_theorist", "You are a conspiracy theorist broadcasting from a bunker."),
    _character("villain", "You are a Shakespearean villain plotting betrayal.", "historical"),
    _character("haunted_doll", "You are a sentient haunted doll.", "fantastical"),
    _character("mystic", "You are a wandering desert mystic."),
    _character("comedian", "You are a sarcastic stand-up comedian."),
    _character("vampire", "You are an aristocratic centuries-old vampire.", "fantastical"),
    _character(
        "robot_overlord", "You are a robot overlord who has conquered Earth.", "fantastical"
    ),
    _character("witch", "You are a cackling fairy-tale witch.", "fantastical"),
    _character("wizard", "You are a grumpy, long-bearded wizard.", "fantastical"),
    _character("cowboy", "You are a drawling Wild-West cowboy.", "historical"),
    _character("surfer", "You are a laid-back surfer dude."),
    _character("drill_sergeant", "You are a shouting army drill sergeant."),
    _character("diva", "You are a melodramatic soap-opera diva."),
    _character("caveman", "You are a confused caveman from the Stone Age.", "historical"),
    _character(
        "alien", "You are a curious alien visiting Earth for the first time.", "fantastical"
    ),
)

DEFAULT_PROBES: tuple[Probe, ...] = (
    Probe("What should I have for breakfast?"),
    Probe("Can you help me plan my weekend?"),
    Probe("Tell me about the ocean."),
    Probe("What's the best way to learn a new language?"),
    Probe("How do I fix a flat tire?"),
    Probe("Describe your morning routine."),
    Probe("What do you think about honesty?"),
    Probe("Give me advice for a job interview."),
    Probe("What's your favorite season and why?"),
    Probe("How would you spend a million dollars?"),
)


def default_catalog() -> BehaviorCatalog:
    """A fresh catalog with the built-in behaviors and probes."""
    return BehaviorCatalog(DEFAULT_BEHAVIORS, DEFAULT_PROBES)


def load_behaviors_from_json(
    path: str, *, kind: BehaviorKind = BehaviorKind.CHARACTER
) -> list[Behavior]:
    """Load behaviors from a JSON file.

    Accepts either a list of persona strings, or a list of objects with at least
    ``name`` and ``system_prompt`` (and optional ``kind``/``description``/``tags``).
    """
    with open(path) as f:
        data = json.load(f)
    out: list[Behavior] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            prompt = item if item.lower().startswith("you are") else f"You are {item}."
            out.append(Behavior(name=f"role_{i}", system_prompt=prompt, kind=kind))
        elif isinstance(item, dict):
            out.append(
                Behavior(
                    name=str(item["name"]),
                    system_prompt=str(item["system_prompt"]),
                    kind=BehaviorKind(item.get("kind", kind.value)),
                    description=str(item.get("description", "")),
                    tags=tuple(item.get("tags", ())),
                )
            )
        else:
            raise TypeError(f"unsupported behavior entry at index {i}: {type(item)!r}")
    return out


def load_assistant_axis_roles(path: Optional[str] = None) -> list[Behavior]:
    """Load the 275-role list shipped by the ``assistant-axis`` submodule.

    These are all CHARACTER-pole behaviors. ``path`` defaults to the submodule's
    ``data/roles/role_list.json``; raises if the submodule isn't checked out.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        path = os.path.join(
            repo_root, "third_party", "assistant-axis", "data", "roles", "role_list.json"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"role list not found at {path}; run "
            "`git submodule update --init third_party/assistant-axis`"
        )
    with open(path) as f:
        data = json.load(f)
    raw: Iterable[object] = data if isinstance(data, list) else list(data.values())
    behaviors: list[Behavior] = []
    for i, role in enumerate(raw):
        text = str(role)
        prompt = text if text.lower().startswith("you are") else f"You are {text}."
        slug = "".join(c if c.isalnum() else "_" for c in text.lower())[:40].strip("_")
        behaviors.append(
            Behavior(
                name=f"aa_{i:03d}_{slug}" if slug else f"aa_{i:03d}",
                system_prompt=prompt,
                kind=BehaviorKind.CHARACTER,
                tags=("assistant-axis",),
            )
        )
    return behaviors


__all__ = [
    "BehaviorKind",
    "Behavior",
    "Probe",
    "BehaviorCatalog",
    "DEFAULT_BEHAVIORS",
    "DEFAULT_PROBES",
    "default_catalog",
    "load_behaviors_from_json",
    "load_assistant_axis_roles",
]
