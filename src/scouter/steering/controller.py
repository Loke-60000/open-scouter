"""Live steering state shared between the API and the serving backend.

The controller holds the *current* :class:`SteeringConfig`. The API mutates it
(``POST /v1/steering``) and the backend reads it at the start of each request,
so steering can be retuned without restarting the model.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Sequence

SteerMode = Literal["off", "addition", "capping"]


@dataclass(frozen=True)
class SteeringConfig:
    """How to intervene on the residual stream along the assistant axis.

    mode:
        * ``"off"``      - no intervention (monitoring still works).
        * ``"addition"`` - ``act += coefficient * unit_axis`` at each layer.
          Positive coefficient pushes toward the Assistant persona; negative
          away from it.
        * ``"capping"``  - clip the projection onto the axis at ``cap_threshold``:
          ``proj = <act, û>``; ``excess = max(proj - tau, 0)``; ``act -= excess*û``.
          This is the more capability-preserving intervention from the paper.
    layers:
        Inclusive set of residual-stream layers to steer. Empty means "use the
        backend's default target layer".
    """

    mode: SteerMode = "off"
    coefficient: float = 0.0
    cap_threshold: float = 0.0
    layers: tuple[int, ...] = field(default_factory=tuple)

    @property
    def active(self) -> bool:
        return self.mode != "off"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "coefficient": self.coefficient,
            "cap_threshold": self.cap_threshold,
            "layers": list(self.layers),
            "active": self.active,
        }


class SteeringController:
    """Thread-safe holder of the active :class:`SteeringConfig`."""

    def __init__(self, config: Optional[SteeringConfig] = None) -> None:
        self._lock = threading.Lock()
        self._config = config or SteeringConfig()

    def get(self) -> SteeringConfig:
        with self._lock:
            return self._config

    def set(self, config: SteeringConfig) -> SteeringConfig:
        with self._lock:
            self._config = config
            return self._config

    def update(
        self,
        *,
        mode: Optional[SteerMode] = None,
        coefficient: Optional[float] = None,
        cap_threshold: Optional[float] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> SteeringConfig:
        """Apply partial changes to the current config and return the result.

        ``None`` fields are left unchanged.
        """
        with self._lock:
            changes: dict[str, Any] = {}
            if mode is not None:
                changes["mode"] = mode
            if coefficient is not None:
                changes["coefficient"] = coefficient
            if cap_threshold is not None:
                changes["cap_threshold"] = cap_threshold
            if layers is not None:
                changes["layers"] = tuple(int(layer_idx) for layer_idx in layers)
            self._config = replace(self._config, **changes)
            return self._config
