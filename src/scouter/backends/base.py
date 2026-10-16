"""Backend abstraction for the serving engine (currently vLLM).

A backend owns the loaded model. It applies steering (optional) and, when
monitoring is requested, captures the residual-stream activations at the monitor
layer. The server projects those activations onto whatever is loaded (the
assistant axis and/or emotion vectors), so monitoring works with or without
steering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

from ..axis import Axis
from ..monitoring import MonitorTrace
from ..steering import SteeringConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Raw residual-stream activations at the monitor layer, (n_steps, hidden),
    # captured by the backend when monitoring is on. The server projects these.
    monitor_activations: Optional[np.ndarray] = None
    # Filled in by the server from monitor_activations:
    monitor: Optional[MonitorTrace] = None              # assistant-axis trace
    emotion_profile: Optional[dict[str, float]] = None  # top emotions
    feature_profile: Optional[list[dict]] = None        # top dictionary features


@dataclass
class StreamChunk:
    """One streamed step: a text delta and (optionally) its axis projection."""

    text: str = ""
    score: Optional[float] = None
    finish_reason: Optional[str] = None


class Backend(ABC):
    """Common interface for serving backends."""

    name: str = "base"

    def __init__(self) -> None:
        self._axis: Optional[Axis] = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Idempotent."""

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    def axis(self) -> Optional[Axis]:
        return self._axis

    def set_axis(self, axis: Optional[Axis]) -> None:
        if axis is not None:
            if axis.hidden_dim != self.hidden_size:
                raise ValueError(
                    f"axis hidden_dim {axis.hidden_dim} != model hidden_size {self.hidden_size}"
                )
        self._axis = axis
        self._on_axis_changed()

    def _on_axis_changed(self) -> None:
        """Hook for backends that need to react to a new axis (no-op default)."""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        steering: SteeringConfig,
        *,
        monitor: bool,
        monitor_layer: int,
    ) -> GenerationResult:
        """Generate a completion (non-streaming)."""

    def stream(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        steering: SteeringConfig,
        *,
        monitor: bool,
        monitor_layer: int,
    ) -> Iterator[StreamChunk]:
        """Stream a completion. Default: generate fully then emit one chunk.

        Backends that support incremental decoding override this to yield token
        deltas (and per-token projection scores) as they are produced.
        """
        result = self.generate(
            messages, params, steering, monitor=monitor, monitor_layer=monitor_layer
        )
        yield StreamChunk(text=result.text, finish_reason=result.finish_reason)


def build_backend(settings: "Settings") -> Backend:
    """Construct the backend named by ``settings.backend``."""
    if settings.backend == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(settings)
    raise ValueError(f"unknown backend: {settings.backend!r}")
