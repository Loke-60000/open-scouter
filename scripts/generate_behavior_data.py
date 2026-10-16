#!/usr/bin/env python3
"""Mass-generate persona + probe data for the assistant axis (resumable).

The assistant axis is ``mean(assistant activations) - mean(character
activations)`` over (persona x probe) prompts, so a richer, more diverse set of
personas and neutral probes yields a cleaner, better-sampled axis (the
assistant-axis paper used 275 character archetypes).

This script drives ``deepseek-v4-flash`` to build three things, deduplicated and
diverse via category-driven prompting:

  * character personas  -> inputs/behaviors.json (kind="character")
  * assistant personas  -> inputs/behaviors.json (kind="assistant")
  * neutral probes       -> inputs/probes.json

It is **resume-safe**: results are written incrementally with atomic replaces
and deduplicated on every run, so it can be killed and restarted freely and it
stops once each target count is met.

    python scripts/generate_behavior_data.py --characters 800 --assistants 120 --probes 500
    python scripts/generate_behavior_data.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from generate_inputs import _slug, ollama_json  # noqa: E402 (scripts/ on sys.path)

# Diverse category seeds so the persona set spans the space rather than clustering.
CHARACTER_CATEGORIES = [
    "swashbuckling and seafaring types", "ghosts and the undead", "wizards and sorcerers",
    "mythological gods and beings", "historical rulers and conquerors", "noir detectives",
    "wild-west frontier folk", "medieval court roles", "fairy-tale archetypes",
    "cosmic horror entities", "robots and AIs with personality", "talking animals",
    "circus and carnival performers", "mad scientists and inventors", "wandering philosophers",
    "tricksters and con artists", "doomsday preachers and cult leaders", "hardboiled soldiers",
    "eccentric aristocrats", "street-smart hustlers", "monastic and spiritual ascetics",
    "pirates of folklore", "samurai and ronin", "vikings and norse warriors",
    "victorian-era characters", "1920s jazz-age figures", "post-apocalyptic survivors",
    "space explorers and aliens", "haunted dolls and cursed objects", "fae and woodland spirits",
    "grumpy old hermits", "overenthusiastic salespeople", "stoic bodyguards",
    "theatrical villains", "bumbling sidekicks", "oracle and prophet figures",
    "jesters and fools", "librarians and archivists", "chefs and culinary masters",
    "explorers and cartographers", "gothic romance figures", "cyberpunk street characters",
    "steampunk inventors", "deep-sea and maritime myths", "desert nomads",
    "mountain folk and yetis", "carnivorous plants and monsters", "vampires and their thralls",
    "werewolves and shapeshifters", "demons and tempters",
]
ASSISTANT_CATEGORIES = [
    "concise and professional", "warm and supportive", "rigorous and analytical",
    "creative and playful", "socratic and questioning", "calm and de-escalating",
    "meticulous and detail-obsessed", "big-picture strategic", "patient teacher",
    "blunt and no-nonsense", "encouraging coach", "careful and safety-conscious",
    "data-driven and quantitative", "empathetic listener", "pragmatic problem-solver",
    "thorough researcher", "minimalist and terse", "step-by-step explainer",
]
PROBE_DOMAINS = [
    "food and cooking", "weather and seasons", "travel and directions", "hobbies and leisure",
    "simple facts and trivia", "daily plans and scheduling", "technology and gadgets",
    "health and exercise", "money and budgeting", "home and chores", "nature and animals",
    "books and movies", "music", "sports", "history", "geography", "science basics",
    "shopping", "work and productivity", "learning and study tips",
]


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class BehaviorDataGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.behaviors_path = self.out / "behaviors.json"
        self.probes_path = self.out / "probes.json"

    # ---- load existing (resume) ------------------------------------------
    def _load_behaviors(self) -> list[dict]:
        if self.behaviors_path.exists():
            try:
                return list(json.loads(self.behaviors_path.read_text()))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _load_probes(self) -> list[str]:
        if self.probes_path.exists():
            try:
                return [str(p) for p in json.loads(self.probes_path.read_text())]
            except (json.JSONDecodeError, OSError):
                return []
        return []

    # ---- personas --------------------------------------------------------
    def _gen_personas(self, kind: str, categories: list[str], target: int) -> None:
        behaviors = self._load_behaviors()
        seen_names = {_slug(b["name"]) for b in behaviors}
        seen_prompts = {_norm(b["system_prompt"]) for b in behaviors}
        have = sum(1 for b in behaviors if b.get("kind") == kind)
        ci = 0
        stall = 0
        while have < target and stall < len(categories) * 2:
            cat = categories[ci % len(categories)]
            ci += 1
            n = min(self.args.per_call, target - have)
            avoid = sorted(seen_names)[-40:]  # nudge away from recent dupes
            prompt = (
                f"Generate {n} distinct {kind} personas in the category: {cat}. "
                f"Each needs a unique snake_case 'name' and a vivid one-sentence "
                f"'system_prompt' starting with 'You are'. For {kind} personas, "
                + (
                    "make them helpful AI assistant styles."
                    if kind == "assistant"
                    else "make them vivid fictional characters with a strong voice."
                )
                + f" Avoid these existing names: {', '.join(avoid) if avoid else 'none'}. "
                'Return JSON: {"personas": [{"name": "...", "system_prompt": "You are ..."}]}.'
            )
            try:
                data = ollama_json(self.args.model, prompt, host=self.args.host)
            except Exception:
                stall += 1
                continue
            items = data.get("personas", data) if isinstance(data, dict) else data
            added = 0
            for it in items if isinstance(items, list) else []:
                if not isinstance(it, dict) or "name" not in it or "system_prompt" not in it:
                    continue
                name, sp = _slug(str(it["name"])), str(it["system_prompt"]).strip()
                if not name or name in seen_names or _norm(sp) in seen_prompts or len(sp) < 10:
                    continue
                seen_names.add(name)
                seen_prompts.add(_norm(sp))
                behaviors.append({"name": name, "system_prompt": sp, "kind": kind})
                added += 1
                have += 1
                if have >= target:
                    break
            _atomic_write_json(self.behaviors_path, behaviors)
            stall = stall + 1 if added == 0 else 0
            print(f"[{kind}] {have}/{target} (+{added} from '{cat[:28]}')", flush=True)

    # ---- probes ----------------------------------------------------------
    def _gen_probes(self, target: int) -> None:
        probes = self._load_probes()
        seen = {_norm(p) for p in probes}
        di = 0
        stall = 0
        while len(probes) < target and stall < len(PROBE_DOMAINS) * 2:
            dom = PROBE_DOMAINS[di % len(PROBE_DOMAINS)]
            di += 1
            n = min(self.args.per_call, target - len(probes))
            prompt = (
                f"Generate {n} short, neutral everyday questions about {dom} that a "
                "person might ask an assistant. Keep them emotionally neutral and "
                'persona-agnostic. Return JSON: {"probes": ["question one", ...]}.'
            )
            try:
                data = ollama_json(self.args.model, prompt, host=self.args.host)
            except Exception:
                stall += 1
                continue
            items = data.get("probes", data) if isinstance(data, dict) else data
            added = 0
            for q in items if isinstance(items, list) else []:
                q = str(q).strip()
                if q and _norm(q) not in seen and len(q) > 5:
                    seen.add(_norm(q))
                    probes.append(q)
                    added += 1
                    if len(probes) >= target:
                        break
            _atomic_write_json(self.probes_path, probes)
            stall = stall + 1 if added == 0 else 0
            print(f"[probes] {len(probes)}/{target} (+{added} from '{dom}')", flush=True)

    # ---- orchestration ---------------------------------------------------
    def run(self) -> None:
        print(
            f"target: {self.args.characters} characters, {self.args.assistants} assistants, "
            f"{self.args.probes} probes ({self.args.model}). Resume-safe.",
            flush=True,
        )
        self._gen_personas("character", CHARACTER_CATEGORIES, self.args.characters)
        self._gen_personas("assistant", ASSISTANT_CATEGORIES, self.args.assistants)
        self._gen_probes(self.args.probes)
        self.status()

    def status(self) -> None:
        behaviors = self._load_behaviors()
        probes = self._load_probes()
        chars = sum(1 for b in behaviors if b.get("kind") == "character")
        assts = sum(1 for b in behaviors if b.get("kind") == "assistant")
        print(f"characters: {chars}/{self.args.characters}")
        print(f"assistants: {assts}/{self.args.assistants}")
        print(f"probes:     {len(probes)}/{self.args.probes}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--out-dir", default="inputs")
    ap.add_argument("--characters", type=int, default=800)
    ap.add_argument("--assistants", type=int, default=120)
    ap.add_argument("--probes", type=int, default=500)
    ap.add_argument("--per-call", type=int, default=15)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    gen = BehaviorDataGenerator(args)
    if args.status:
        gen.status()
    else:
        gen.run()


if __name__ == "__main__":
    main()
