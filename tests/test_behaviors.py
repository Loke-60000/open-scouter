import dataclasses
import json

import pytest

from scouter.axis import (
    Behavior,
    BehaviorCatalog,
    BehaviorKind,
    Probe,
    default_catalog,
)
from scouter.axis.behaviors import load_behaviors_from_json


def test_behavior_is_frozen_and_with_tags():
    b = Behavior("pirate", "You are a pirate.", tags=("fantastical",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(b, "name", "x")  # frozen - setattr avoids a static type error
    b2 = b.with_tags("nautical", "fantastical")
    assert b2.tags == ("fantastical", "nautical")  # de-duped, original kept
    assert b.tags == ("fantastical",)  # unchanged


def test_catalog_add_and_duplicate_guard():
    cat = BehaviorCatalog()
    cat.register(name="a", system_prompt="You are A.", kind=BehaviorKind.ASSISTANT)
    assert "a" in cat
    assert len(cat) == 1
    with pytest.raises(ValueError):
        cat.register(name="a", system_prompt="dup")
    # replace_existing allows override
    cat.register(name="a", system_prompt="You are A2.", replace_existing=True)
    assert cat["a"].system_prompt == "You are A2."


def test_catalog_filtering():
    cat = BehaviorCatalog()
    cat.register(name="asst", system_prompt="s", kind=BehaviorKind.ASSISTANT)
    cat.register(name="pirate", system_prompt="s", tags=("sea",))
    cat.register(name="ghost", system_prompt="s", tags=("spooky",))
    assert [b.name for b in cat.assistant_behaviors] == ["asst"]
    assert {b.name for b in cat.character_behaviors} == {"pirate", "ghost"}
    assert [b.name for b in cat.with_tag("sea")] == ["pirate"]


def test_catalog_extend_merges():
    a = BehaviorCatalog([Behavior("x", "s", BehaviorKind.ASSISTANT)], [Probe("q1")])
    b = BehaviorCatalog([Behavior("y", "s")], [Probe("q2")])
    a.extend(b)
    assert {bh.name for bh in a} == {"x", "y"}
    assert a.probe_texts() == ["q1", "q2"]


def test_default_catalog_has_both_poles_and_probes():
    cat = default_catalog()
    assert len(cat.assistant_behaviors) >= 2
    assert len(cat.character_behaviors) >= 10
    assert len(cat.probes) >= 5
    cat.validate()  # should not raise


def test_validate_requires_both_poles():
    cat = BehaviorCatalog([Behavior("only_char", "s")], [Probe("q")])
    with pytest.raises(ValueError):
        cat.validate()


def test_load_behaviors_from_json_strings_and_objects(tmp_path):
    p = tmp_path / "roles.json"
    p.write_text(json.dumps(["a wizard", {"name": "bob", "system_prompt": "You are Bob."}]))
    behaviors = load_behaviors_from_json(str(p))
    assert behaviors[0].system_prompt == "You are a wizard."
    assert behaviors[0].kind == BehaviorKind.CHARACTER
    assert behaviors[1].name == "bob"


def test_extending_default_catalog_is_one_call():
    cat = default_catalog()
    before = len(cat)
    cat.register(name="surfer_x", system_prompt="You are a surfer.")
    assert len(cat) == before + 1
    assert "surfer_x" in cat
