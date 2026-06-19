"""Runtime configuration, driven by environment variables (``SCOUTER_*``).

A single :class:`Settings` instance is created lazily via :func:`get_settings`
and shared across the app.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BackendName = Literal["vllm"]
SteerMode = Literal["off", "addition", "capping"]


class Settings(BaseSettings):
    """Server configuration.

    All values can be overridden with ``SCOUTER_<FIELD>`` environment
    variables (case-insensitive), e.g. ``SCOUTER_MODEL=Qwen/Qwen3-8B``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SCOUTER_",
        env_file=".env",
        extra="ignore",
    )

    backend: BackendName = "vllm"

    # The model to serve: a HuggingFace repo id or local path.
    model: str = "Qwen/Qwen3-0.6B"
    served_model_name: str = "scouter"

    # Where to load the axis tensor from. One of:
    #   - local path to a .pt file produced by assistant-axis
    #   - "hf:<model-key>" to pull from the precomputed HF dataset, e.g.
    #     "hf:qwen-3-32b" -> lu-christina/assistant-axis-vectors/qwen-3-32b
    #   - "" (empty) to start with no axis loaded (monitoring/steering off)
    axis: str = ""
    axis_hf_repo: str = "lu-christina/assistant-axis-vectors"

    # Emotion vectors to load (a .npz produced by `scouter-emotions`). Empty
    # means no emotion monitoring. Monitoring needs no steering.
    emotions: str = ""

    # The residual-stream layer used for monitoring/steering. If <0 the axis
    # subsystem picks ``num_layers // 2`` once the model is known.
    target_layer: int = -1

    steer_mode: SteerMode = "off"
    # Steering coefficient (for "addition"). Positive pushes toward the
    # Assistant end of the axis, negative away from it.
    steer_coefficient: float = 0.0
    # Capping threshold tau (for "capping"): projections above tau are clipped.
    steer_cap_threshold: float = 0.0
    # Inclusive layer range for steering. If unset, only `target_layer`.
    steer_layer_start: Optional[int] = None
    steer_layer_end: Optional[int] = None

    # Capture/record projection scores during generation.
    monitor_enabled: bool = True

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    # Which vLLM-Hook worker to attach. "steer_hook_act" enables steering,
    # "probe_hidden_states" enables monitoring only.
    vllm_hook_worker: str = "steer_hook_act"

    host: str = "0.0.0.0"
    port: int = 8000

    # Cache directory for downloaded axes/configs.
    cache_dir: str = "/data/axes"

    def resolve_steer_layers(self, num_layers: int) -> list[int]:
        """Return the list of layer indices steering should touch."""
        if self.steer_layer_start is not None and self.steer_layer_end is not None:
            return list(range(self.steer_layer_start, self.steer_layer_end + 1))
        return [self.resolve_target_layer(num_layers)]

    def resolve_target_layer(self, num_layers: int) -> int:
        if self.target_layer is not None and self.target_layer >= 0:
            return self.target_layer
        return num_layers // 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
