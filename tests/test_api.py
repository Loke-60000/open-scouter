import numpy as np
import pytest
from fastapi.testclient import TestClient

from scouter.api.app import create_app
from scouter.axis import Axis
from scouter.backends.base import Backend, GenerationResult
from scouter.config import Settings
from scouter.emotions import EmotionVectors
from scouter.features import FeatureDictionary


class FakeBackend(Backend):
    """In-memory backend: echoes a reply and captures deterministic activations."""

    name = "fake"
    supported_steer_modes = ("off", "addition", "capping")

    def __init__(self, hidden=8, layers=4):
        super().__init__()
        self._hidden = hidden
        self._layers = layers
        self.last_steering = None

    def load(self):
        pass

    @property
    def num_layers(self):
        return self._layers

    @property
    def hidden_size(self):
        return self._hidden

    def generate(self, messages, params, steering, *, monitor, monitor_layer):
        self.last_steering = steering
        text = f"echo: {messages[-1].content}"
        result = GenerationResult(text=text, prompt_tokens=3, completion_tokens=2)
        if monitor:
            # capture raw activations; the server does the projection
            result.monitor_activations = np.stack(
                [np.full(self._hidden, float(i), dtype=np.float32) for i in range(3)]
            )
        return result


@pytest.fixture
def client():
    app = create_app(Settings(backend="vllm", model="dummy-model", monitor_enabled=True))
    st = app.state.scouter
    st.backend = FakeBackend(hidden=8, layers=4)
    st._model_loaded = True  # skip real model load
    with TestClient(app) as c:
        c.vllm_state = st
        yield c


def _load_axis(client):
    st = client.vllm_state
    ax = Axis(directions=np.ones((4, 8), dtype=np.float32), metadata={}, source="test")
    st.backend.set_axis(ax)
    st._axis = ax


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "vllm"
    assert body["axis_loaded"] is False


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "scouter"


def test_chat_without_axis_has_no_monitor(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "echo: hi"
    assert body["axis_monitor"] is None


def test_chat_with_axis_returns_monitor(client):
    _load_axis(client)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "monitor": True},
    )
    body = r.json()
    mon = body["axis_monitor"]
    assert mon is not None
    assert mon["count"] == 3
    assert mon["layer"] == 2  # default target layer = num_layers // 2
    # activations 0,1,2 projected onto unit(all-ones/sqrt8) -> increasing
    assert mon["drift"] > 0


def test_chat_streaming_sse(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert "chat.completion.chunk" in body
    assert "echo: hi" in body
    assert "[DONE]" in body


def test_steering_requires_axis(client):
    r = client.post("/v1/steering", json={"mode": "addition", "coefficient": 5.0})
    assert r.status_code == 400  # no axis loaded


def test_steering_update_and_applied(client):
    _load_axis(client)
    r = client.post("/v1/steering", json={"mode": "addition", "coefficient": 5.0, "layers": [1]})
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["mode"] == "addition"
    assert cfg["active"] is True
    # generation should see the steering config
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert client.vllm_state.backend.last_steering.coefficient == 5.0
    assert client.vllm_state.backend.last_steering.layers == (1,)


def test_capping_rejected_when_backend_unsupported(client):
    _load_axis(client)
    client.vllm_state.backend.supported_steer_modes = ("off", "addition")
    r = client.post("/v1/steering", json={"mode": "capping", "cap_threshold": 1.0})
    assert r.status_code == 400


def test_axis_info_endpoint(client):
    _load_axis(client)
    r = client.get("/v1/axis")
    body = r.json()
    assert body["loaded"] is True
    assert body["n_layers"] == 4
    assert body["hidden_dim"] == 8


def test_monitor_endpoint(client):
    _load_axis(client)
    r = client.post("/v1/monitor", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 200
    assert r.json()["axis_monitor"]["count"] == 3


def _load_emotions(client):
    # joy along basis 0, sadness along basis 1, over 4 layers x 8 dims
    dirs = {}
    for i, name in enumerate(["joy", "sadness"]):
        v = np.zeros((4, 8), dtype=np.float32)
        v[:, i] = 1.0
        dirs[name] = v
    client.vllm_state._emotions = EmotionVectors(dirs)


def test_monitoring_works_without_steering_or_axis(client):
    # Emotions only: no axis loaded, steering off. Monitoring still runs.
    _load_emotions(client)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "monitor": True},
    )
    body = r.json()
    assert body["axis_monitor"] is None  # no axis loaded
    assert body["emotion_monitor"] is not None  # emotions still monitored
    assert set(body["emotion_monitor"]) == {"joy", "sadness"}


def test_emotions_endpoints(client):
    assert client.get("/v1/emotions").json()["loaded"] is False
    _load_emotions(client)
    info = client.get("/v1/emotions").json()
    assert info["loaded"] is True
    assert info["count"] == 2
    assert set(info["names"]) == {"joy", "sadness"}


def _load_features(client):
    # feature 0 reads basis 0, feature 1 reads basis 1, over 8 dims; trained at layer 2
    w_enc = np.zeros((8, 2), dtype=np.float32)
    w_enc[0, 0] = w_enc[1, 1] = 1.0
    client.vllm_state._features = FeatureDictionary(
        w_enc,
        np.zeros(2),
        w_enc.T * 3.0,
        np.zeros(8),
        layer=2,
        n_layers=4,
        labels={0: "bridge"},
        max_activations=np.array([2.0, 2.0], np.float32),
    )


def test_feature_monitor_and_endpoints(client):
    assert client.get("/v1/features").json()["loaded"] is False
    assert client.get("/v1/features/0").status_code == 400
    _load_features(client)
    info = client.get("/v1/features").json()
    assert info == {"loaded": True, "count": 2, "layer": 2, "labeled": 1}

    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "monitor": True},
    )
    hits = r.json()["feature_monitor"]
    assert r.json()["axis_monitor"] is None
    # activations are all-equal rows 0,1,2 -> both features peak at 2.0
    assert {h["index"] for h in hits} == {0, 1}
    assert hits[0]["activation"] == pytest.approx(2.0)
    assert next(h["label"] for h in hits if h["index"] == 0) == "bridge"

    feat = client.get("/v1/features/0").json()
    assert feat["label"] == "bridge"
    assert feat["clamp_coefficient"] == pytest.approx(10 * 2.0 * 3.0)
    assert client.get("/v1/features/7").status_code == 404


def test_feature_as_steering_axis(client):
    _load_features(client)
    r = client.post("/v1/axis", json={"spec": "feature:0"})
    assert r.status_code == 200
    assert r.json()["source"] == "feature:0"
    assert client.vllm_state.axis.metadata["label"] == "bridge"
    assert client.vllm_state.backend.axis is client.vllm_state.axis
    r = client.post("/v1/steering", json={"mode": "addition", "coefficient": 60.0})
    assert r.json()["active"] is True


def test_emotion_as_steering_axis(client):
    assert client.post("/v1/axis", json={"spec": "emotion:joy"}).status_code == 400
    _load_emotions(client)
    r = client.post("/v1/axis", json={"spec": "emotion:joy"})
    assert r.status_code == 200
    assert client.vllm_state.axis.metadata == {"emotion": "joy"}
