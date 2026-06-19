"""Load and reason about the *assistant axis*.

The assistant axis (Anthropic, 2026 - https://www.anthropic.com/research/assistant-axis)
is a direction in a model's residual stream that captures how "Assistant-like"
the model's current persona is. The ``safety-research/assistant-axis`` project
computes it as::

    axis = mean(default_activations) - mean(role_activations)

and stores it as a torch tensor of shape ``(n_layers, hidden_dim)`` - one
direction per layer. Projecting a layer's activations onto the (L2-normalized)
axis gives a scalar "assistant-likeness" score; higher means more Assistant-like.

This module wraps loading that tensor (from disk or the precomputed HuggingFace
dataset) and exposes the projection used for monitoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np


@dataclass
class Axis:
    """An assistant axis: one direction per residual-stream layer.

    Attributes
    ----------
    directions:
        ``(n_layers, hidden_dim)`` float32 array. Raw (un-normalized) axis as
        produced by assistant-axis.
    metadata:
        Optional dict carried alongside the tensor (model name, layer, etc.).
    source:
        Human-readable description of where the axis came from.
    """

    directions: np.ndarray
    metadata: dict
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.directions.ndim != 2:
            raise ValueError(
                f"axis must be 2-D (n_layers, hidden_dim); got shape {self.directions.shape}"
            )
        self.directions = np.ascontiguousarray(self.directions, dtype=np.float32)
        # Pre-compute the L2-normalized directions used for projection.
        norms = np.linalg.norm(self.directions, axis=1, keepdims=True)
        self._unit = self.directions / (norms + 1e-8)

    @property
    def n_layers(self) -> int:
        return self.directions.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.directions.shape[1]

    def unit(self, layer: int) -> np.ndarray:
        """L2-normalized direction at ``layer``."""
        return self._unit[layer]

    def direction(self, layer: int) -> np.ndarray:
        """Raw direction at ``layer``."""
        return self.directions[layer]

    def project(self, activation: np.ndarray, layer: int) -> float:
        """Project an activation vector onto the unit axis at ``layer``.

        ``activation`` is a ``(hidden_dim,)`` residual-stream vector. Returns the
        signed assistant-likeness score (dot product with the normalized axis),
        matching ``assistant_axis.axis.project``.
        """
        act = np.asarray(activation, dtype=np.float32).reshape(-1)
        if act.shape[0] != self.hidden_dim:
            raise ValueError(f"activation dim {act.shape[0]} != axis hidden_dim {self.hidden_dim}")
        return float(act @ self._unit[layer])

    def project_batch(self, activations: np.ndarray, layer: int) -> np.ndarray:
        """Project a ``(n, hidden_dim)`` batch; returns ``(n,)`` scores."""
        acts = np.asarray(activations, dtype=np.float32)
        return acts @ self._unit[layer]


def _to_numpy(obj: object) -> np.ndarray:
    """Convert a torch tensor / numpy array / list to a float32 numpy array."""
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    # torch tensor (duck-typed to avoid a hard torch import where unnecessary)
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        tensor = cast(Any, obj)
        return tensor.detach().cpu().float().numpy()
    return np.asarray(obj, dtype=np.float32)


def _load_pt(path: str) -> tuple[np.ndarray, dict]:
    """Load an assistant-axis ``.pt`` file.

    Accepts both on-disk shapes produced by the upstream project:
      * a raw tensor ``(n_layers, hidden_dim)`` (``pipeline/5_axis.py``), or
      * a dict ``{"axis": tensor, "metadata": {...}}`` (``save_axis``).
    """
    import torch

    # weights_only=False because the dict form may carry python metadata. Only
    # load axes you trust (same caveat as upstream load_axis).
    obj = torch.load(path, map_location="cpu", weights_only=False)
    metadata: dict = {}
    if isinstance(obj, dict) and "axis" in obj:
        metadata = dict(obj.get("metadata") or {})
        tensor = obj["axis"]
    else:
        tensor = obj
    arr = _to_numpy(tensor)
    return arr, metadata


def _resolve_hf(spec: str, hf_repo: str, cache_dir: Optional[str]) -> str:
    """Resolve an ``hf:<model-key>`` spec to a local ``.pt`` path.

    Mirrors the layout of ``lu-christina/assistant-axis-vectors``:
    ``<model-key>/assistant_axis.pt``.
    """
    from huggingface_hub import hf_hub_download

    key = spec[len("hf:") :].strip("/")
    if key.endswith(".pt"):
        filename = key
    else:
        filename = f"{key}/assistant_axis.pt"
    return hf_hub_download(
        repo_id=hf_repo,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def load_axis(
    spec: str,
    *,
    hf_repo: str = "lu-christina/assistant-axis-vectors",
    cache_dir: Optional[str] = None,
) -> Axis:
    """Load an assistant axis from ``spec``.

    ``spec`` is one of:
      * ``""`` -> raises (no axis); callers should treat empty as "no axis".
      * a local path to a ``.pt`` file.
      * ``hf:<model-key>`` -> download from the precomputed HF dataset.
    """
    if not spec:
        raise ValueError("empty axis spec")

    if spec.startswith("hf:"):
        path = _resolve_hf(spec, hf_repo, cache_dir)
        source = f"{hf_repo}:{spec[3:]}"
    else:
        path = os.path.expanduser(spec)
        if not os.path.exists(path):
            raise FileNotFoundError(f"axis file not found: {path}")
        source = path

    if path.endswith(".npy"):
        directions, metadata = np.load(path).astype(np.float32), {}
    elif path.endswith(".npz"):
        npz = np.load(path)
        directions = npz["axis"].astype(np.float32)
        metadata = {k: npz[k].tolist() for k in npz.files if k != "axis"}
    else:  # .pt / .bin (torch)
        directions, metadata = _load_pt(path)
    return Axis(directions=directions, metadata=metadata, source=source)


def save_axis(axis: Axis, path: str, metadata: Optional[dict] = None) -> str:
    """Persist an :class:`Axis`.

    The extension selects the format:
      * ``.npz`` / ``.npy`` -> numpy (no torch needed to reload).
      * anything else (``.pt``) -> torch dict ``{"axis", "metadata"}``, matching
        the upstream ``assistant_axis.axis.save_axis`` layout.
    """
    meta = metadata if metadata is not None else axis.metadata
    if path.endswith(".npy"):
        np.save(path, axis.directions)
    elif path.endswith(".npz"):
        np.savez(path, axis=axis.directions)
    else:
        import torch

        torch.save({"axis": torch.from_numpy(axis.directions), "metadata": meta}, path)
    return path
