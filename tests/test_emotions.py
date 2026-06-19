import json

import numpy as np
import pytest

from scouter.axis import Axis
from scouter.emotions import DEFAULT_EMOTIONS, EmotionVectors
from scouter.emotions.catalog import load_emotions_json


def make_vectors(n_layers=4, hidden=8):
    # Each emotion points along a distinct basis direction at every layer.
    dirs = {}
    for i, name in enumerate(["joy", "sadness", "anger"]):
        v = np.zeros((n_layers, hidden), dtype=np.float32)
        v[:, i] = 1.0  # unit along axis i
        dirs[name] = v
    return EmotionVectors(dirs)


def test_shapes_and_names():
    ev = make_vectors()
    assert ev.n_layers == 4
    assert ev.hidden_dim == 8
    assert set(ev.names) == {"joy", "sadness", "anger"}
    assert "joy" in ev and "fear" not in ev


def test_profile_and_dominant_pick_the_aligned_emotion():
    ev = make_vectors(hidden=8)
    layer = 1
    act = np.zeros(8, dtype=np.float32)
    act[1] = 5.0  # aligned with "sadness" (basis index 1)
    prof = ev.profile(act, layer)
    assert next(iter(prof)) == "sadness"        # sorted, highest first
    assert ev.dominant(act, layer) == ("sadness", pytest.approx(5.0))
    assert prof["joy"] == pytest.approx(0.0)


def test_as_axis_bridges_to_steering():
    ev = make_vectors()
    ax = ev.as_axis("anger")
    assert isinstance(ax, Axis)
    assert ax.directions.shape == (4, 8)
    assert ax.metadata["emotion"] == "anger"


def test_save_and_load_roundtrip(tmp_path):
    ev = make_vectors()
    p = str(tmp_path / "emotions.npz")
    ev.save(p)
    reloaded = EmotionVectors.load(p)
    assert set(reloaded.names) == set(ev.names)
    assert np.allclose(reloaded.vector("joy"), ev.vector("joy"))


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        EmotionVectors({"a": np.zeros((4, 8), np.float32), "b": np.zeros((4, 6), np.float32)})


def test_empty_rejected():
    with pytest.raises(ValueError):
        EmotionVectors({})


def test_default_catalog_is_nonempty_and_distinct():
    names = [e.name for e in DEFAULT_EMOTIONS]
    assert len(names) >= 12
    assert len(names) == len(set(names))   # unique
    assert "joy" in names and "anger" in names


def test_load_emotions_json_scenarios_and_fallbacks(tmp_path):
    p = tmp_path / "emotions.json"
    p.write_text(json.dumps([
        {"name": "fear", "scenarios": ["The footsteps grew louder.", "Her hands shook."]},
        {"name": "legacy", "feeling": "old joy"},  # fallback -> a scenario
        "awe",                                       # bare string fallback
    ]))
    emos = load_emotions_json(str(p))
    assert emos[0].name == "fear" and emos[0].scenarios == (
        "The footsteps grew louder.",
        "Her hands shook.",
    )
    assert emos[1].scenarios          # fallback produced at least one scenario
    assert emos[2].name == "awe" and emos[2].scenarios
