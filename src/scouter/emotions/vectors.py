"""The :class:`EmotionVectors` container: named per-layer emotion directions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..axis.loader import Axis

FloatArray = NDArray[np.float32]
NEUTRAL_KEY = "__neutral__"  # reserved npz key for the neutral reference


class EmotionVectors:
    """A set of named emotion directions over a model's layers.

    ``directions[name]`` is an ``(n_layers, hidden_dim)`` float32 array. Project
    an activation onto an emotion's (unit) direction to score that emotion;
    profile against all of them to read the dominant emotion. An emotion is also
    an :class:`Axis` (:meth:`as_axis`), so it plugs into the steering path.

    ``neutral`` is an optional ``(n_layers, hidden_dim)`` reference activation
    (the mean over neutral prompts). When present, :meth:`profile` subtracts it
    from the query activation before projecting, removing the shared "being
    prompted" common-mode that otherwise swamps the per-emotion signal. The
    emotion vectors already had neutral PCs projected out at build time; this
    applies the matching correction to the activation being read.
    """

    def __init__(
        self, directions: dict[str, FloatArray], neutral: FloatArray | None = None
    ) -> None:
        if not directions:
            raise ValueError("no emotion directions")
        self._dirs: dict[str, FloatArray] = {
            name: np.ascontiguousarray(v, dtype=np.float32) for name, v in directions.items()
        }
        shapes = {v.shape for v in self._dirs.values()}
        if len(shapes) != 1:
            raise ValueError(f"all emotion directions must share a shape; got {shapes}")
        self._neutral: FloatArray | None = None
        if neutral is not None:
            neutral = np.ascontiguousarray(neutral, dtype=np.float32)
            if neutral.shape != next(iter(self._dirs.values())).shape:
                raise ValueError(
                    f"neutral shape {neutral.shape} must match a direction "
                    f"{next(iter(self._dirs.values())).shape}"
                )
            self._neutral = neutral
        self._units: dict[str, FloatArray] = {
            name: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            for name, v in self._dirs.items()
        }

    @property
    def has_neutral(self) -> bool:
        return self._neutral is not None

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
        """Project ``activation`` onto every emotion at ``layer`` (sorted high to low).

        If a neutral reference is loaded, it is subtracted from ``activation``
        first to strip the shared common-mode (greatly improves separation).
        """
        act = np.asarray(activation, dtype=np.float32).reshape(-1)
        if self._neutral is not None:
            act = act - self._neutral[layer]
        scores = {name: float(act @ self._units[name][layer]) for name in self._dirs}
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def dominant(self, activation: ArrayLike, layer: int) -> tuple[str, float]:
        prof = self.profile(activation, layer)
        name = next(iter(prof))
        return name, prof[name]

    def save(self, path: str) -> str:
        arrays: dict[str, Any] = dict(self._dirs)  # Any: np.savez kwds typing
        if self._neutral is not None:
            arrays[NEUTRAL_KEY] = self._neutral
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "EmotionVectors":
        data = np.load(path)
        neutral = data[NEUTRAL_KEY] if NEUTRAL_KEY in data.files else None
        dirs = {name: data[name] for name in data.files if name != NEUTRAL_KEY}
        return cls(dirs, neutral=neutral)
