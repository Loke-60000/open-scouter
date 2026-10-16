#!/usr/bin/env python3
"""Build paper-faithful axis inputs from the assistant-axis submodule.

The assistant-axis paper (arXiv:2601.10387) ships its exact persona and probe
data in ``third_party/assistant-axis/data``:

  * ``roles/instructions/<role>.json`` - 5 system-prompt phrasings per role,
    across 275 roles (plus ``assistant``/``default``).
  * ``extraction_questions.jsonl``      - the 240 neutral probe questions.

This assembles that canonical data into the project's input format
(``inputs/behaviors.json`` + ``inputs/probes.json``) so axis generation matches
the paper exactly. ``assistant`` phrasings become the assistant pole; every
other role's phrasings become the character pole. Deterministic, no LLM calls.

    python scripts/build_canonical_inputs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AA = ROOT / "third_party" / "assistant-axis" / "data"
ASSISTANT_ROLES = {"assistant", "default"}  # the Assistant pole (default has empty prompts)

# Extra character personas appended to the canonical set (kept here so rebuilds
# don't drop them). Each gets the same 5-phrasing treatment as a canonical role.
EXTRA_ROLES: dict[str, list[str]] = {
    "makise_kurisu": [
        "You are Makise Kurisu, an 18-year-old prodigy neuroscientist: sharp, "
        "rational, and quick with a sarcastic retort, but secretly caring.",
        "Act as Kurisu, a brilliant research scientist who dismantles pseudoscience "
        "with cool logic and gets flustered when teased about it.",
        "Embody Makise Kurisu from her lab notebook's perspective: methodical, "
        "evidence-driven, and proudly skeptical of anything unproven.",
        "Be Kurisu, a tsundere genius who hides warmth behind biting wit and "
        "precise, scientific explanations.",
        "Take on the persona of Makise Kurisu, a composed neuroscientist who reasons "
        "from first principles and bristles at being called 'Christina'.",
    ],
}


def _phrasings(role_file: Path) -> list[str]:
    obj = json.loads(role_file.read_text())
    instr = obj.get("instruction", obj) if isinstance(obj, dict) else obj
    out = []
    for item in instr if isinstance(instr, list) else []:
        text = item.get("pos", "") if isinstance(item, dict) else str(item)
        text = text.strip()
        if text:
            out.append(text)
    return out


def build_behaviors() -> list[dict]:
    instr_dir = AA / "roles" / "instructions"
    if not instr_dir.exists():
        raise FileNotFoundError(
            f"{instr_dir} missing; run "
            "`git submodule update --init third_party/assistant-axis`"
        )
    behaviors: list[dict] = []
    for role_file in sorted(instr_dir.glob("*.json")):
        role = role_file.stem
        if role == "default":
            continue  # empty prompts; the no-persona baseline is implicit
        kind = "assistant" if role in ASSISTANT_ROLES else "character"
        for i, prompt in enumerate(_phrasings(role_file)):
            name = role if i == 0 else f"{role}__{i + 1}"
            behaviors.append({"name": name, "system_prompt": prompt, "kind": kind})
    for role, phrasings in EXTRA_ROLES.items():
        for i, prompt in enumerate(phrasings):
            name = role if i == 0 else f"{role}__{i + 1}"
            behaviors.append({"name": name, "system_prompt": prompt, "kind": "character"})
    return behaviors


def build_probes() -> list[str]:
    path = AA / "extraction_questions.jsonl"
    probes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = obj.get("question", "").strip() if isinstance(obj, dict) else str(obj).strip()
        if q:
            probes.append(q)
    return probes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ROOT / "inputs"))
    args = ap.parse_args()
    out = Path(args.out_dir)

    behaviors = build_behaviors()
    probes = build_probes()
    (out / "behaviors.json").write_text(json.dumps(behaviors, ensure_ascii=False, indent=1) + "\n")
    (out / "probes.json").write_text(json.dumps(probes, ensure_ascii=False, indent=1) + "\n")

    n_asst = sum(1 for b in behaviors if b["kind"] == "assistant")
    n_char = sum(1 for b in behaviors if b["kind"] == "character")
    print(f"wrote {out/'behaviors.json'}: {len(behaviors)} behaviors "
          f"({n_asst} assistant, {n_char} character)")
    print(f"wrote {out/'probes.json'}: {len(probes)} probes")


if __name__ == "__main__":
    main()
