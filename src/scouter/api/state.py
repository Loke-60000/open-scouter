from __future__ import annotations

import logging
import threading
from typing import Iterator, Optional

from ..axis import Axis, load_axis
from ..backends import Backend, build_backend
from ..backends.base import ChatMessage, GenerationResult, SamplingParams, StreamChunk
from ..config import Settings, get_settings
from ..emotions import EmotionVectors
from ..monitoring import MonitorRecorder
from ..steering import SteeringConfig, SteeringController

logger = logging.getLogger("scouter")

EMOTION_TOP_K = 5


class AppState:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.backend: Backend = build_backend(self.settings)
        self.steering = SteeringController(
            SteeringConfig(
                mode=self.settings.steer_mode,
                coefficient=self.settings.steer_coefficient,
                cap_threshold=self.settings.steer_cap_threshold,
            )
        )
        self._model_loaded = False
        self._axis: Optional[Axis] = None
        self._emotions: Optional[EmotionVectors] = None
        self._gen_lock = threading.Lock()  # backends are not thread-safe

    def ensure_model(self) -> None:
        if not self._model_loaded:
            logger.info(
                "loading model %s on backend %s", self.settings.model, self.settings.backend
            )
            self.backend.load()
            self._model_loaded = True
            # Resolve a concrete target layer now that we know the model.
            if self.settings.target_layer < 0:
                self.settings.target_layer = self.settings.resolve_target_layer(
                    self.backend.num_layers
                )
            if self.settings.axis and self._axis is None:
                try:
                    self.load_axis(self.settings.axis)
                except Exception:  # pragma: no cover - startup best effort
                    logger.exception("failed to load configured axis %s", self.settings.axis)
            if self.settings.emotions and self._emotions is None:
                try:
                    self.load_emotions(self.settings.emotions)
                except Exception:  # pragma: no cover - startup best effort
                    logger.exception("failed to load emotions %s", self.settings.emotions)

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def axis(self) -> Optional[Axis]:
        return self._axis

    @property
    def emotions(self) -> Optional[EmotionVectors]:
        return self._emotions

    def load_axis(self, spec: str, target_layer: Optional[int] = None) -> Axis:
        axis = load_axis(
            spec,
            hf_repo=self.settings.axis_hf_repo,
            cache_dir=self.settings.cache_dir,
        )
        self.backend.set_axis(axis)
        self._axis = axis
        if target_layer is not None:
            self.settings.target_layer = target_layer
        logger.info(
            "axis loaded from %s (%d layers, dim %d)", axis.source, axis.n_layers, axis.hidden_dim
        )
        return axis

    def load_emotions(self, spec: str) -> EmotionVectors:
        ev = EmotionVectors.load(spec)
        if self._model_loaded and ev.hidden_dim != self.backend.hidden_size:
            raise ValueError(
                f"emotion hidden_dim {ev.hidden_dim} != model hidden_size "
                f"{self.backend.hidden_size}"
            )
        self._emotions = ev
        logger.info("emotions loaded from %s (%d emotions)", spec, len(ev))
        return ev

    @property
    def monitor_layer(self) -> int:
        return self.settings.resolve_target_layer(self.backend.num_layers or 1)

    def _should_monitor(self, override: Optional[bool]) -> bool:
        want = self.settings.monitor_enabled if override is None else override
        # Monitoring needs no steering; it just needs something to project onto.
        return bool(want) and (self._axis is not None or self._emotions is not None)

    def _enrich_monitor(self, result: GenerationResult) -> None:
        """Project the captured activations onto the axis and/or emotion vectors."""
        acts = result.monitor_activations
        if acts is None or len(acts) == 0:
            return
        layer = self.monitor_layer
        if self._axis is not None:
            recorder = MonitorRecorder(self._axis, layer)
            for vec in acts:
                recorder.record_activation(vec)
            result.monitor = recorder.trace
        if self._emotions is not None:
            elayer = min(layer, self._emotions.n_layers - 1)
            profile = self._emotions.profile(acts.mean(axis=0), elayer)
            result.emotion_profile = dict(list(profile.items())[:EMOTION_TOP_K])

    def generate(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        *,
        monitor: Optional[bool] = None,
    ) -> GenerationResult:
        self.ensure_model()
        do_monitor = self._should_monitor(monitor)
        steering = self.steering.get()
        with self._gen_lock:
            result = self.backend.generate(
                messages,
                params,
                steering,
                monitor=do_monitor,
                monitor_layer=self.monitor_layer,
            )
        self._enrich_monitor(result)
        return result

    def stream(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        *,
        monitor: Optional[bool] = None,
    ) -> Iterator[StreamChunk]:
        self.ensure_model()
        steering = self.steering.get()
        # Streaming returns text only; use the non-streaming path for the monitor.
        with self._gen_lock:
            yield from self.backend.stream(
                messages,
                params,
                steering,
                monitor=False,
                monitor_layer=self.monitor_layer,
            )
