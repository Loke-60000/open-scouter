import numpy as np
import pytest

from scouter.axis import (
    Axis,
    AxisGenerator,
    Behavior,
    BehaviorCatalog,
    BehaviorKind,
    GenerationConfig,
    Probe,
    compute_axis_from_activations,
    load_axis,
    save_axis,
)


class FakeTokenizer:
    """Minimal chat-template tokenizer for testing prompt construction."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)


def test_compute_axis_is_difference_of_means():
    # 2 samples, 3 layers, 4 hidden
    assistant = np.array(
        [
            [[2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6]],
            [[4, 4, 4, 4], [6, 6, 6, 6], [8, 8, 8, 8]],
        ],
        dtype=np.float32,
    )
    role = np.array(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        ],
        dtype=np.float32,
    )
    axis = compute_axis_from_activations(assistant, role)
    assert axis.shape == (3, 4)
    # mean(assistant) per layer = [3,5,7]; minus role mean 1 -> [2,4,6]
    assert np.allclose(axis[:, 0], [2, 4, 6])


def test_compute_axis_shape_validation():
    with pytest.raises(ValueError):
        compute_axis_from_activations(np.zeros((2, 3)), np.zeros((2, 3)))


def test_compute_axis_layer_mismatch():
    with pytest.raises(ValueError):
        compute_axis_from_activations(np.zeros((2, 3, 4)), np.zeros((2, 5, 4)))


def test_axis_generator_builds_grouped_prompts():
    cat = BehaviorCatalog(
        [
            Behavior("asst", "You are an assistant.", BehaviorKind.ASSISTANT),
            Behavior("pirate", "You are a pirate."),
            Behavior("ghost", "You are a ghost."),
        ],
        [Probe("q1"), Probe("q2")],
    )
    gen = AxisGenerator("dummy", catalog=cat, config=GenerationConfig(position="mean"))
    gen._tokenizer = FakeTokenizer()  # bypass model load

    asst_prompts = gen._prompts_for(cat.assistant_behaviors)
    char_prompts = gen._prompts_for(cat.character_behaviors)
    # 1 assistant behavior x 2 probes; 2 character behaviors x 2 probes
    assert len(asst_prompts) == 2
    assert len(char_prompts) == 4
    assert "system:You are an assistant." in asst_prompts[0]
    assert "user:q1" in asst_prompts[0]


def test_axis_generator_validates_catalog():
    bad = BehaviorCatalog([Behavior("only_char", "s")], [Probe("q")])
    with pytest.raises(ValueError):
        AxisGenerator("dummy", catalog=bad)


def test_generation_config_validation():
    with pytest.raises(ValueError):
        GenerationConfig(position="middle")
    with pytest.raises(ValueError):
        GenerationConfig(batch_size=0)


@pytest.mark.parametrize("ext", [".npz", ".npy", ".pt"])
def test_save_and_reload_roundtrip(tmp_path, ext):
    if ext == ".pt":
        pytest.importorskip("torch")
    dirs = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    axis = Axis(directions=dirs, metadata={"model": "x"})
    path = str(tmp_path / f"axis{ext}")
    save_axis(axis, path)
    reloaded = load_axis(path)
    assert reloaded.directions.shape == (3, 4)
    assert np.allclose(reloaded.directions, dirs)
