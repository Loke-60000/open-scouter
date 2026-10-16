import json

import numpy as np
import pytest

from scouter.axis import Axis, TokenActivations
from scouter.features import FeatureDictionary, SAEConfig, drop_tokens, label_features, train_sae


def make_dictionary(hidden=4, n_features=3, n_layers=6, layer=2):
    # feature i reads and writes basis direction i
    w_enc = np.zeros((hidden, n_features), np.float32)
    w_dec = np.zeros((n_features, hidden), np.float32)
    for i in range(n_features):
        w_enc[i, i] = 1.0
        w_dec[i, i] = 2.0
    return FeatureDictionary(
        w_enc,
        np.zeros(n_features, np.float32),
        w_dec,
        np.zeros(hidden, np.float32),
        layer=layer,
        n_layers=n_layers,
        labels={0: "bridge"},
        max_activations=np.array([5.0, 1.0, 1.0], np.float32),
    )


def test_encode_is_relu_and_top_features_sorted():
    fd = make_dictionary()
    act = np.array([3.0, -1.0, 0.5, 9.0], np.float32)  # dim 3 has no feature
    f = fd.encode(act)
    assert f.tolist() == [3.0, 0.0, 0.5]
    hits = fd.top_features(act, k=5)
    assert [h.index for h in hits] == [0, 2]  # zero activations are dropped
    assert hits[0].label == "bridge" and hits[1].label == "feature_2"
    assert hits[0].max_activation == 5.0


def test_top_features_takes_max_over_steps():
    fd = make_dictionary()
    steps = np.array([[0.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], np.float32)
    hits = fd.top_features(steps, k=1)
    assert hits[0].index == 1 and hits[0].activation == 4.0


def test_as_axis_and_clamp_coefficient():
    fd = make_dictionary()
    ax = fd.as_axis(0)
    assert isinstance(ax, Axis)
    assert ax.directions.shape == (6, 4)
    assert np.allclose(ax.direction(2), [2, 0, 0, 0])
    assert not ax.directions[[0, 1, 3, 4, 5]].any()
    assert ax.metadata == {"feature": 0, "label": "bridge"}
    # 10 x max activation (5) x decoder norm (2)
    assert fd.clamp_coefficient(0) == pytest.approx(100.0)


def test_save_load_roundtrip(tmp_path):
    fd = make_dictionary()
    fd.metadata["model"] = "m"
    p = str(tmp_path / "features.npz")
    fd.save(p)
    back = FeatureDictionary.load(p)
    assert back.layer == 2 and back.n_layers == 6 and len(back) == 3
    assert back.labels == {0: "bridge"} and back.metadata == {"model": "m"}
    act = np.array([3.0, -1.0, 0.5, 9.0], np.float32)
    assert np.allclose(back.encode(act), fd.encode(act))
    assert np.allclose(back.decode(fd.encode(act)), fd.decode(fd.encode(act)))


def test_shape_validation():
    with pytest.raises(ValueError):
        FeatureDictionary(
            np.zeros((4, 3)), np.zeros(3), np.zeros((2, 4)), np.zeros(4), layer=0, n_layers=1
        )
    with pytest.raises(ValueError):
        FeatureDictionary(
            np.zeros((4, 3)), np.zeros(3), np.zeros((3, 4)), np.zeros(4), layer=5, n_layers=2
        )
    with pytest.raises(ValueError):
        make_dictionary().encode(np.zeros(7))


def test_train_sae_recovers_sparse_ground_truth():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    hidden, n_true, n = 16, 6, 4000
    truth = rng.normal(size=(n_true, hidden)).astype(np.float32)
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    codes = rng.exponential(2.0, size=(n, n_true)) * (rng.random((n, n_true)) < 0.15)
    acts = (codes @ truth).astype(np.float32)

    cfg = SAEConfig(n_features=12, l1_coef=0.2, lr=5e-3, steps=1500, batch_size=256, device="cpu")
    fd = train_sae(acts, layer=3, n_layers=8, config=cfg)

    recon = fd.decode(fd.encode(acts))
    rel_err = np.linalg.norm(recon - acts) / np.linalg.norm(acts)
    assert rel_err < 0.1
    feats = fd.encode(acts)
    assert (feats > 0).sum(axis=1).mean() < 4  # sparse: far fewer active than 12 features
    # every true direction has a decoder row closely aligned with it
    cos = np.abs(truth @ fd.w_dec.T) / (np.linalg.norm(fd.w_dec, axis=1) + 1e-8)
    assert cos.max(axis=1).min() > 0.9
    assert fd.layer == 3 and fd.metadata["n_samples"] == n
    assert np.all(fd.max_activations >= feats.max(axis=0) - 1e-4)


def test_label_features_uses_top_tokens():
    fd = make_dictionary()
    a = np.zeros((3, 4), np.float32)
    a[1, 0] = 9.0  # feature 0 fires on the 2nd token
    b = np.zeros((2, 4), np.float32)
    b[0, 0] = 4.0
    b[1, 2] = 1.0
    corpus = [
        TokenActivations(("The", " Golden", " Gate"), a),
        TokenActivations(("Golden", " Bridge"), b),
    ]
    examples = label_features(fd, corpus, n_examples=2, context=1)
    assert fd.labels[0] == "Golden"
    assert examples[0][0].activation == 9.0 and examples[0][0].context == "The[[ Golden]] Gate"
    assert fd.labels[2] == "Bridge" and 1 not in examples  # feature 1 never fired
    json.dumps({k: [e.context for e in v] for k, v in examples.items()})  # serialisable


def test_drop_tokens_removes_special_tokens_and_their_rows():
    acts = np.arange(6, dtype=np.float32).reshape(3, 2)
    corpus = [
        TokenActivations(("<bos>", "hi", "<eos>"), acts),
        TokenActivations(("<bos>",), acts[:1]),  # nothing left -> item dropped
    ]
    out = drop_tokens(corpus, frozenset({"<bos>", "<eos>"}))
    assert len(out) == 1
    assert out[0].tokens == ("hi",) and out[0].activations.tolist() == [[2.0, 3.0]]
