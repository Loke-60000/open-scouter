import numpy as np
import pytest

from scouter.axis import Axis


def make_axis(n_layers=4, hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_layers, hidden)).astype(np.float32)
    return Axis(directions=dirs, metadata={"k": "v"})


def test_shape_and_props():
    ax = make_axis(4, 8)
    assert ax.n_layers == 4
    assert ax.hidden_dim == 8
    assert ax.metadata["k"] == "v"


def test_unit_is_normalized():
    ax = make_axis()
    for layer in range(ax.n_layers):
        assert np.isclose(np.linalg.norm(ax.unit(layer)), 1.0, atol=1e-5)


def test_projection_matches_dot_with_unit():
    ax = make_axis()
    layer = 2
    act = np.arange(ax.hidden_dim, dtype=np.float32)
    expected = float(act @ (ax.direction(layer) / np.linalg.norm(ax.direction(layer))))
    assert np.isclose(ax.project(act, layer), expected, atol=1e-5)


def test_projection_of_axis_itself_is_its_norm():
    # Projecting the raw direction onto its unit vector returns ||direction||.
    ax = make_axis()
    layer = 1
    proj = ax.project(ax.direction(layer), layer)
    assert np.isclose(proj, np.linalg.norm(ax.direction(layer)), atol=1e-4)


def test_project_batch():
    ax = make_axis()
    layer = 0
    batch = np.stack([np.ones(ax.hidden_dim), np.zeros(ax.hidden_dim)]).astype(np.float32)
    scores = ax.project_batch(batch, layer)
    assert scores.shape == (2,)
    assert np.isclose(scores[1], 0.0, atol=1e-6)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        Axis(directions=np.zeros(8, dtype=np.float32), metadata={})


def test_projection_dim_mismatch_raises():
    ax = make_axis(hidden=8)
    with pytest.raises(ValueError):
        ax.project(np.zeros(4), 0)
