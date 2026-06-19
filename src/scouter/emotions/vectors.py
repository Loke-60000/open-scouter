"""The :class:`EmotionVectors` container: named per-layer emotion directions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..axis.loader import Axis

FloatArray = NDArray[np.float32]


class EmotionVectors:
    """A set of named emotion directions over a model's layers.

    ``directions[name]`` is an ``(n_layers, hidden_dim)`` float32 array. Project
    an activation onto an emotion's (unit) direction to score that emotion;
    profile against all of them to read the dominant emotion. An emotion is also
    an :class:`Axis` (:meth:`as_axis`), so it plugs into the steering path.
    """

    def __init__(self, directions: dict[str, FloatArray]) -> None:
        if not directions:
            raise ValueError("no emotion directions")
        self._dirs: dict[str, FloatArray] = {
            name: np.ascontiguousarray(v, dtype=np.float32) for name, v in directions.items()
        }
        shapes = {v.shape for v in self._dirs.values()}
        if len(shapes) != 1:
            raise ValueError(f"all emotion directions must share a shape; got {shapes}")
        self._units: dict[str, FloatArray] = {
            name: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            for name, v in self._dirs.items()
        }

    @property
    def names(self) -> list[str]:
        return list(self._dirs)

    @property
    def n_layers(self) -> int:
        return next(iter(self._dirs.values())).shape[0]

    @property
    def hidden_dim(self) -> int:
        return next(iter(self._dirs.values())).shape[1]

    def __len__(self) -> int:
        return len(self._dirs)

    def __contains__(self, name: object) -> bool:
        return name in self._dirs

    def vector(self, name: str) -> FloatArray:
        return self._dirs[name]

    def as_axis(self, name: str) -> Axis:
        """Return one emotion direction as an :class:`Axis` (for steering)."""
        return Axis(self._dirs[name], metadata={"emotion": name})

    def profile(self, activation: ArrayLike, layer: int) -> dict[str, float]:
        """Project ``activation`` onto every emotion at ``layer`` (sorted high to low)."""
        act = np.asarray(activation, dtype=np.float32).reshape(-1)
        scores = {name: float(act @ self._units[name][layer]) for name in self._dirs}
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def dominant(self, activation: ArrayLike, layer: int) -> tuple[str, float]:
        prof = self.profile(activation, layer)
        name = next(iter(prof))
        return name, prof[name]

    def save(self, path: str) -> str:
        arrays: dict[str, Any] = dict(self._dirs)  # Any: np.savez kwds typing
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "EmotionVectors":
        data = np.load(path)
        return cls({name: data[name] for name in data.files})
