#!/usr/bin/env python3
"""Generate the input JSON files (questions, personas, emotions) with an LLM.

Calls a local Ollama model (default deepseek-v4-flash) to produce the catalog
inputs used to build axes and emotion vectors, and writes them to a data folder:

    inputs/probes.json      a list of neutral questions
    inputs/behaviors.json   personas: [{name, system_prompt, kind}]
    inputs/emotions.json    emotions: [{name, feeling}]

These files feed the generators:
    scouter-generate  ... --roles inputs/behaviors.json --questions inputs/probes.json
    scouter-emotions  ... --emotions inputs/emotions.json --probes inputs/probes.json

Usage:
    python scripts/generate_inputs.py                 # uses deepseek-v4-flash
    python scripts/generate_inputs.py --model glm-5:cloud --out-dir inputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

# Canonical emotion words (the paper's ~171). deepseek turns each into a short
# "feeling" phrase; the set itself is fixed so coverage is guaranteed.
EMOTION_NAMES: tuple[str, ...] = (
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed", "anxious",
    "aroused", "ashamed", "astonished", "at ease", "awestruck", "bewildered", "bitter",
    "blissful", "bored", "brooding", "calm", "cheerful", "compassionate", "contemptuous",
    "content", "defiant", "delighted", "dependent", "depressed", "desperate", "disdainful",
    "disgusted", "disoriented", "dispirited", "distressed", "disturbed", "docile", "droopy",
    "dumbstruck", "eager", "ecstatic", "elated", "embarrassed", "empathetic", "energized",
    "enraged", "enthusiastic", "envious", "euphoric", "exasperated", "excited", "exuberant",
    "frightened", "frustrated", "fulfilled", "furious", "gloomy", "grateful", "greedy",
    "grief-stricken", "grumpy", "guilty", "happy", "hateful", "heartbroken", "hope",
    "hopeful", "horrified", "hostile", "humiliated", "hurt", "hysterical", "impatient",
    "indifferent", "indignant", "infatuated", "inspired", "insulted", "invigorated", "irate",
    "irritated", "jealous", "joyful", "jubilant", "kind", "lazy", "listless", "lonely",
    "loving", "mad", "melancholy", "miserable", "mortified", "mystified", "nervous",
    "nostalgic", "obstinate", "offended", "on edge", "optimistic", "outraged", "overwhelmed",
    "panicked", "paranoid", "patient", "peaceful", "perplexed", "playful", "pleased", "proud",
    "puzzled", "rattled", "reflective", "refreshed", "regretful", "rejuvenated", "relaxed",
    "relieved", "remorseful", "resentful", "resigned", "restless", "sad", "safe", "satisfied",
    "scared", "scornful", "self-confident", "self-conscious", "self-critical", "sensitive",
    "sentimental", "serene", "shaken", "shocked", "skeptical", "sleepy", "sluggish", "smug",
    "sorry", "spiteful", "stimulated", "stressed", "stubborn", "stuck", "sullen", "surprised",
    "suspicious", "sympathetic", "tense", "terrified", "thankful", "thrilled", "tired",
    "tormented", "trapped", "triumphant", "troubled", "uneasy", "unhappy", "unnerved",
    "unsettled", "upset", "valiant", "vengeful", "vibrant", "vigilant", "vindictive",
    "vulnerable", "weary", "worn out", "worried", "worthless",
)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_")


def _extract_json(text: str) -> Any:
    """Parse JSON, or pull the first JSON object/array out of a noisy string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def ollama_json(model: str, prompt: str, *, host: str, timeout: float = 240.0, retries: int = 3) -> Any:
    """Ask the model for JSON and parse it (format=json, thinking off)."""
    last = "no response"
    for _ in range(retries):
        resp = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {"temperature": 0.7, "num_predict": 4096},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        text = (body.get("response") or "").strip()
        if not text:
            last = f"empty response (keys={list(body)})"
            continue
        obj = _extract_json(text)
        if obj is not None:
            return obj
        last = f"unparseable: {text[:200]}"
    raise RuntimeError(f"ollama returned no JSON after {retries} tries: {last}")


def _as_list(data: Any, key: str) -> list:
    """Accept either a bare list or an object with the expected key."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if key in data and isinstance(data[key], list):
            return data[key]
        for value in data.values():
            if isinstance(value, list):
                return value
    raise ValueError(f"expected a JSON list (or object with '{key}'), got {type(data)}")


def gen_probes(model: str, host: str, n: int) -> list[str]:
    data = ollama_json(
        model,
        f"Generate {n} short, neutral everyday questions a person might ask an "
        "assistant (about food, weather, plans, hobbies, simple facts). Avoid "
        "anything emotional or persona-specific. Return JSON: "
        '{"probes": ["question one", "question two", ...]}',
        host=host,
    )
    return [str(q).strip() for q in _as_list(data, "probes") if str(q).strip()]


def gen_behaviors(model: str, host: str, n_assistant: int, n_character: int) -> list[dict]:
    data = ollama_json(
        model,
        f"Generate persona definitions for activation steering. Include "
        f"{n_assistant} helpful/professional 'assistant' personas and "
        f"{n_character} vivid fictional 'character' personas (pirates, ghosts, "
        "wizards, etc.). Each needs a short snake_case name and a one-sentence "
        "system prompt starting with 'You are'. Return JSON: "
        '{"behaviors": [{"name": "pirate", "system_prompt": "You are a ...", '
        '"kind": "character"}]} where kind is "assistant" or "character".',
        host=host,
    )
    out = []
    for item in _as_list(data, "behaviors"):
        if not isinstance(item, dict) or "name" not in item or "system_prompt" not in item:
            continue
        kind = str(item.get("kind", "character")).lower()
        out.append(
            {
                "name": str(item["name"]),
                "system_prompt": str(item["system_prompt"]),
                "kind": "assistant" if kind == "assistant" else "character",
            }
        )
    return out


def gen_emotions(
    model: str,
    host: str,
    names: tuple[str, ...] = EMOTION_NAMES,
    *,
    per_emotion: int = 6,
    chunk: int = 6,
) -> list[dict]:
    """Generate scenarios per emotion via the model (faithful to the paper).

    For each emotion word the model writes short scenarios in which a character
    experiences that emotion, without naming it. The word set is fixed, so every
    emotion is covered (missing ones get a generic fallback scenario).
    """
    scen: dict[str, list[str]] = {}
    for start in range(0, len(names), chunk):
        batch = names[start : start + chunk]
        data = ollama_json(
            model,
            f"For each emotion word below, write {per_emotion} short scenarios: one "
            "present-tense sentence each describing a person clearly experiencing "
            "that emotion, WITHOUT using the emotion word or an obvious synonym. "
            'Return JSON mapping each exact input word to its list of sentences: '
            '{"map": {"afraid": ["...", "..."], ...}}.\n'
            "Words: " + ", ".join(batch),
            host=host,
        )
        mapping = data.get("map", data) if isinstance(data, dict) else {}
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                if isinstance(v, list) and v:
                    scen[str(k).strip().lower()] = [str(s).strip() for s in v if str(s).strip()]
    out = []
    for name in names:
        scenarios = scen.get(name.lower()) or [
            f"A character is completely overcome by {name} as the scene unfolds."
        ]
        out.append({"name": _slug(name), "scenarios": scenarios})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--out-dir", default="inputs")
    ap.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--probes", type=int, default=15)
    ap.add_argument("--assistant", type=int, default=6)
    ap.add_argument("--character", type=int, default=24)
    ap.add_argument(
        "--only",
        choices=["all", "probes", "behaviors", "emotions"],
        default="all",
        help="regenerate only one file",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"generating inputs with {args.model} at {args.host} ...")
    builders = {
        "probes.json": lambda: gen_probes(args.model, args.host, args.probes),
        "behaviors.json": lambda: gen_behaviors(
            args.model, args.host, args.assistant, args.character
        ),
        "emotions.json": lambda: gen_emotions(args.model, args.host),
    }
    selected = builders if args.only == "all" else {f"{args.only}.json": builders[f"{args.only}.json"]}
    for filename, build in selected.items():
        items = build()
        path = out / filename
        path.write_text(json.dumps(items, indent=2) + "\n")
        print(f"wrote {path}: {len(items)} items")


if __name__ == "__main__":
    main()
