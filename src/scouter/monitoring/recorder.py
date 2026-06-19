"""Accumulate assistant-axis projection scores during a generation.

A :class:`MonitorRecorder` collects the per-token (or per-step) projection of
the residual stream onto the axis, and summarizes the trajectory - useful for
spotting *persona drift*: a sustained drop in assistant-likeness over a
conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..axis import Axis


@dataclass
class MonitorTrace:
    """Summary of one monitored generation."""

    layer: int
    scores: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> Optional[float]:
        return float(np.mean(self.scores)) if self.scores else None

    @property
    def min(self) -> Optional[float]:
        return float(np.min(self.scores)) if self.scores else None

    @property
    def max(self) -> Optional[float]:
        return float(np.max(self.scores)) if self.scores else None

    @property
    def last(self) -> Optional[float]:
        return self.scores[-1] if self.scores else None

    def drift(self) -> Optional[float]:
        """Signed slope of the projection over the generation (least squares).

        Negative means the model is drifting *away* from the Assistant persona
        as it generates.
        """
        if len(self.scores) < 2:
            return None
        x = np.arange(len(self.scores), dtype=np.float64)
        y = np.asarray(self.scores, dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "last": self.last,
            "drift": self.drift(),
            "scores": self.scores,
        }


class MonitorRecorder:
    """Records projections for a single generation against a fixed axis/layer."""

    def __init__(self, axis: "Axis", layer: int, *, keep_scores: bool = True) -> None:
        self._axis = axis
        self._layer = layer
        self._keep_scores = keep_scores
        self._trace = MonitorTrace(layer=layer)

    @property
    def trace(self) -> MonitorTrace:
        return self._trace

    def record_activation(self, activation: np.ndarray) -> float:
        """Project a ``(hidden_dim,)`` residual vector and record the score."""
        score = self._axis.project(activation, self._layer)
        if self._keep_scores:
            self._trace.scores.append(score)
        return score

    def record_score(self, score: float) -> None:
        if self._keep_scores:
            self._trace.scores.append(float(score))
