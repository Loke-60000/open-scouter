"""The :class:`FeatureDictionary`: a sparse-autoencoder dictionary over one layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..axis.loader import Axis

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FeatureHit:
    """One feature firing on an activation."""

    index: int
    label: str
    activation: float
    max_activation: float  # largest activation seen on the training corpus

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "activation": self.activation,
            "max_activation": self.max_activation,
        }


class FeatureDictionary:
    """Dictionary features learned by a sparse autoencoder on one residual layer.

    Following Anthropic's "Scaling Monosemanticity" (the Golden Gate Bridge
    feature): the encoder reads an activation ``x`` into sparse feature
    activations ``f = relu(W_enc (x - b_dec) + b_enc)``, and the decoder
    reconstructs ``x ≈ W_dec f + b_dec``. Each decoder row is a direction in
    the residual stream, so a feature is also an :class:`Axis` (:meth:`as_axis`)
    and can be clamped on via the normal steering path.

    Weights live in the model's raw activation units (any training-time
    normalisation is folded in), so :meth:`encode` takes activations straight
    from the backend.
    """

    def __init__(
        self,
        w_enc: ArrayLike,
        b_enc: ArrayLike,
        w_dec: ArrayLike,
        b_dec: ArrayLike,
        *,
        layer: int,
        n_layers: int,
        labels: dict[int, str] | None = None,
        max_activations: ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.w_enc = np.ascontiguousarray(w_enc, dtype=np.float32)  # (hidden, n_features)
        self.b_enc = np.ascontiguousarray(b_enc, dtype=np.float32)  # (n_features,)
        self.w_dec = np.ascontiguousarray(w_dec, dtype=np.float32)  # (n_features, hidden)
        self.b_dec = np.ascontiguousarray(b_dec, dtype=np.float32)  # (hidden,)
        hidden, n = self.w_enc.shape
        if (
            self.w_dec.shape != (n, hidden)
            or self.b_enc.shape != (n,)
            or self.b_dec.shape != (hidden,)
        ):
            raise ValueError(
                f"inconsistent shapes: w_enc {self.w_enc.shape}, b_enc {self.b_enc.shape}, "
                f"w_dec {self.w_dec.shape}, b_dec {self.b_dec.shape}"
            )
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} outside 0..{n_layers - 1}")
        self.layer = int(layer)
        self.n_layers = int(n_layers)
        self.labels: dict[int, str] = {int(k): str(v) for k, v in (labels or {}).items()}
        self.max_activations = (
            np.ascontiguousarray(max_activations, dtype=np.float32)
            if max_activations is not None
            else np.ones(n, dtype=np.float32)
        )
        if self.max_activations.shape != (n,):
            raise ValueError(f"max_activations must be ({n},)")
        self.metadata = dict(metadata or {})

    @property
    def n_features(self) -> int:
        return self.w_enc.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.w_enc.shape[0]

    def __len__(self) -> int:
        return self.n_features

    def label(self, index: int) -> str:
        return self.labels.get(index, f"feature_{index}")

    def encode(self, activations: ArrayLike) -> FloatArray:
        """``(..., hidden) -> (..., n_features)`` sparse feature activations."""
        x = np.asarray(activations, dtype=np.float32)
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"activation dim {x.shape[-1]} != hidden_dim {self.hidden_dim}")
        pre = (x - self.b_dec) @ self.w_enc + self.b_enc
        return np.maximum(pre, 0.0, dtype=np.float32)

    def decode(self, features: ArrayLike) -> FloatArray:
        f = np.asarray(features, dtype=np.float32)
        return (f @ self.w_dec + self.b_dec).astype(np.float32)

    def top_features(self, activations: ArrayLike, k: int = 10) -> list[FeatureHit]:
        """Strongest features over ``(hidden,)`` or ``(steps, hidden)`` activations.

        Multi-step input is reduced with a max per feature: dictionary features
        are sparse and fire on specific tokens, so a mean would wash them out.
        """
        acts = self.encode(np.atleast_2d(np.asarray(activations, dtype=np.float32)))
        peak = acts.max(axis=0)
        order = np.argsort(-peak)[:k]
        return [
            FeatureHit(int(i), self.label(int(i)), float(peak[i]), float(self.max_activations[i]))
            for i in order
            if peak[i] > 0
        ]

    def direction(self, index: int) -> FloatArray:
        """Unit decoder direction of feature ``index`` in the residual stream."""
        v = self.w_dec[index]
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)

    def as_axis(self, index: int) -> Axis:
        """Feature ``index`` as an :class:`Axis` (non-zero only at this layer)."""
        dirs = np.zeros((self.n_layers, self.hidden_dim), dtype=np.float32)
        dirs[self.layer] = self.w_dec[index]
        return Axis(dirs, metadata={"feature": index, "label": self.label(index)})

    def clamp_coefficient(self, index: int, scale: float = 10.0) -> float:
        """Steering coefficient that adds ``scale`` x the feature's max activation.

        Golden Gate Claude clamped the feature to 10x its maximum observed value;
        with ``addition`` steering (``act += coefficient * unit``) this is the
        matching coefficient for :meth:`as_axis`.
        """
        return float(scale * self.max_activations[index] * np.linalg.norm(self.w_dec[index]))

    def save(self, path: str) -> str:
        np.savez(
            path,
            w_enc=self.w_enc,
            b_enc=self.b_enc,
            w_dec=self.w_dec,
            b_dec=self.b_dec,
            max_activations=self.max_activations,
            layer=self.layer,
            n_layers=self.n_layers,
            labels=json.dumps(self.labels),
            metadata=json.dumps(self.metadata),
        )
        return path

    @classmethod
    def load(cls, path: str) -> "FeatureDictionary":
        d = np.load(path)
        return cls(
            d["w_enc"],
            d["b_enc"],
            d["w_dec"],
            d["b_dec"],
            layer=int(d["layer"]),
            n_layers=int(d["n_layers"]),
            labels={int(k): v for k, v in json.loads(str(d["labels"])).items()},
            max_activations=d["max_activations"],
            metadata=json.loads(str(d["metadata"])),
        )
