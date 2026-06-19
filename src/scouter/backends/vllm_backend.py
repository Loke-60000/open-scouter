"""vLLM backend (via IBM ``vLLM-Hook``).

``vLLM-Hook`` is a vLLM *plugin* that attaches PyTorch forward hooks inside the
GPU worker so activations can be read (``probe_hidden_states``) or rewritten
(``steer_hook_act``) per request, driven by ``SamplingParams.extra_args``.

This backend bridges the assistant axis to that machinery:

* **Steering** - the axis row for the target layer is written as a vLLM-Hook
  steering-vector file (``{"dir": û, "avg_proj": τ}``) plus a model-config JSON,
  and applied per request via ``extra_args={"steer": {...}}``:
    - ``addition`` -> method ``add_vector`` (``act += coefficient * û``)
    - ``capping``  -> method ``adjust_rs``  (force the projection toward ``τ``)
* **Monitoring** - requests ``extra_args={"output_hidden_states": [layer]}`` and
  projects the returned hidden states onto the axis.

A single ``HookLLM`` instance attaches exactly one worker (``vllm_hook_worker``
in settings). ``steer_hook_act`` enables steering; ``probe_hidden_states``
enables layer-resolved monitoring. Requires a GPU and the ``vllm`` +
``vllm_hook_plugins`` packages (see ``docker/Dockerfile.vllm``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..steering import SteeringConfig
from .base import Backend, ChatMessage, GenerationResult, SamplingParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings

# Map our steering modes to vLLM-Hook worker methods.
_METHOD_FOR_MODE = {"addition": "add_vector", "capping": "adjust_rs"}


class VLLMBackend(Backend):
    name = "vllm"
    supported_steer_modes = ("off", "addition", "capping")

    def __init__(self, settings: "Settings") -> None:
        super().__init__()
        self._settings = settings
        self._llm: Any = None
        self._tokenizer: Any = None
        self._num_layers = 0
        self._hidden = 0
        self._worker = settings.vllm_hook_worker
        self._steer_dir = os.path.join(settings.cache_dir, "vllm_steer")
        self._vector_path: Optional[str] = None

    def load(self) -> None:
        if self._llm is not None:
            return
        # vLLM v1 + spawn, per vLLM-Hook requirements.
        os.environ.setdefault("VLLM_USE_V1", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm_hook_plugins import HookLLM

        self._llm = HookLLM(
            model=self._settings.model,
            worker_name=self._worker,
            enforce_eager=True,
            enable_hook=True,
            tensor_parallel_size=self._settings.tensor_parallel_size,
            gpu_memory_utilization=self._settings.gpu_memory_utilization,
            dtype=self._settings.dtype,
            trust_remote_code=True,
        )
        self._tokenizer = self._llm.tokenizer
        cfg = self._llm.llm_engine.model_config.hf_config
        self._num_layers = int(getattr(cfg, "num_hidden_layers", 0))
        self._hidden = int(getattr(cfg, "hidden_size", 0))

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def _write_steer_files(self, layer: int, cap_threshold: float) -> str:
        """Persist the axis row at ``layer`` in vLLM-Hook's expected format.

        Returns the path to the steering-vector ``.pt`` file.
        """
        import torch

        assert self._axis is not None  # guarded by callers (steering active)
        os.makedirs(self._steer_dir, exist_ok=True)
        unit = self._axis.unit(layer)
        path = os.path.join(self._steer_dir, f"axis_layer{layer}.pt")
        torch.save(
            {
                "dir": torch.tensor(np.ascontiguousarray(unit), dtype=torch.float32),
                # avg_proj is the projection target used by adjust_rs ("capping").
                "avg_proj": torch.tensor(float(cap_threshold), dtype=torch.float32),
            },
            path,
        )
        self._vector_path = path
        return path

    def _steer_extra_args(self, steering: SteeringConfig, layer: int) -> Optional[dict]:
        if not steering.active or self._axis is None:
            return None
        method = _METHOD_FOR_MODE.get(steering.mode)
        if method is None:
            raise ValueError(f"unsupported steering mode for vLLM: {steering.mode!r}")
        path = self._write_steer_files(layer, steering.cap_threshold)
        return {
            "steer": {
                "vector_path": path,
                "optimal_layer": layer,
                "method": method,
                "coefficient": float(steering.coefficient),
                "apply_at_all_positions": True,
            }
        }

    def _render_prompt(self, messages: list[ChatMessage]) -> str:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return self._tokenizer.apply_chat_template(
            msg_dicts, add_generation_prompt=True, tokenize=False
        )

    def generate(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
        steering: SteeringConfig,
        *,
        monitor: bool,
        monitor_layer: int,
    ) -> GenerationResult:
        self.load()
        from vllm import SamplingParams as VLLMSamplingParams

        steer_layer = (
            list(steering.layers)[0]
            if steering.layers
            else self._settings.resolve_target_layer(self._num_layers)
        )

        extra_args: dict = {}
        steer_args = self._steer_extra_args(steering, steer_layer)
        if steer_args:
            extra_args.update(steer_args)
        if monitor:
            extra_args["output_hidden_states"] = [monitor_layer]

        sp = VLLMSamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k and params.top_k > 0 else -1,
            max_tokens=params.max_tokens,
            stop=params.stop or None,
            seed=params.seed,
            extra_args=extra_args or None,
        )

        prompt = self._render_prompt(messages)
        outputs = self._llm.generate(prompt, sp)
        out = outputs[0]
        completion = out.outputs[0]
        text = completion.text
        result = GenerationResult(
            text=text,
            finish_reason=completion.finish_reason or "stop",
            prompt_tokens=len(out.prompt_token_ids or []),
            completion_tokens=len(completion.token_ids or []),
        )

        if monitor:
            result.monitor_activations = self._capture(out, monitor_layer)
        return result

    def _capture(self, out: Any, layer: int) -> Optional[np.ndarray]:
        """Return the captured hidden states at ``layer`` as ``(n_steps, hidden)``.

        The server projects these onto the axis and/or emotion vectors; the
        backend itself does not interpret them.
        """
        try:
            probes = getattr(out, "probes", None)
            if probes is None:
                return None
            hidden = self._select_layer_hidden(probes, layer)
            if hidden is None:
                return None
            arr = np.asarray(hidden, dtype=np.float32)
            return arr[None, :] if arr.ndim == 1 else arr
        except Exception:
            return None

    @staticmethod
    def _select_layer_hidden(probes: Any, layer: int) -> Any:
        """Best-effort extraction of ``(tokens, hidden)`` for ``layer`` from probes.

        ``vLLM-Hook`` returns probes as a dict keyed by layer or a structure with
        a ``hidden_states`` field; we handle the common shapes defensively.
        """
        if isinstance(probes, dict):
            if layer in probes:
                return probes[layer]
            if str(layer) in probes:
                return probes[str(layer)]
            hs = probes.get("hidden_states")
            if isinstance(hs, dict):
                return hs.get(layer, hs.get(str(layer)))
            return hs
        hs = getattr(probes, "hidden_states", None)
        if isinstance(hs, dict):
            return hs.get(layer, hs.get(str(layer)))
        return hs
