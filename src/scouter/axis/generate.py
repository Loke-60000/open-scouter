"""
Generate an assistant-axis vector file from a HuggingFace model.
Implements Anthropic's recipe (https://www.anthropic.com/research/assistant-axis):
run the model under the default **Assistant** persona and under a spread of other
**character** personas, capture the residual stream at each layer, and take the
difference of means::

    axis = mean(assistant_activations) - mean(role_activations)

The personas come from a :class:`~scouter.axis.behaviors.BehaviorCatalog`
(see ``behaviors.py`` - that's where you add more). The result is an
``(n_layers, hidden_dim)`` tensor in the format the server loads.

Object model:
  * :class:`ActivationExtractor` - runs prompts, returns per-layer activations.
  * :class:`AxisGenerator` - orchestrates catalog -> prompts -> extract -> axis.
  * :func:`generate_axis` - a thin functional wrapper over :class:`AxisGenerator`.
  * :func:`compute_axis_from_activations` - the pure-numpy core (unit-tested).

Requires the ``generate`` extra (``pip install -e ".[generate]"``: transformers +
accelerate + torch). The numpy core has no such dependency.
"""

# torch/transformers are an optional extra ("[generate]"); don't flag them in the
# editor when that extra isn't installed.
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .behaviors import (
    Behavior,
    BehaviorCatalog,
    BehaviorKind,
    default_catalog,
    load_assistant_axis_roles,
    load_behaviors_from_json,
)
from .loader import Axis, save_axis

FloatArray = NDArray[np.float32]
Position = str  # Literal-ish: "last" | "mean"


def compute_axis_from_activations(
    assistant_activations: ArrayLike,
    role_activations: ArrayLike,
) -> FloatArray:
    """``axis = mean(assistant) - mean(role)`` over the sample axis.

    Both inputs are array-likes shaped ``(n_samples, n_layers, hidden_dim)`` (any
    dtype - coerced to float32); returns ``(n_layers, hidden_dim)`` float32.
    Mirrors ``assistant_axis.axis.compute_axis``.
    """
    a = np.asarray(assistant_activations, dtype=np.float32)
    r = np.asarray(role_activations, dtype=np.float32)
    if a.ndim != 3 or r.ndim != 3:
        raise ValueError("activations must be (n_samples, n_layers, hidden_dim)")
    if a.shape[1:] != r.shape[1:]:
        raise ValueError(f"layer/hidden mismatch: assistant {a.shape[1:]} vs role {r.shape[1:]}")
    return (a.mean(axis=0) - r.mean(axis=0)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Tunables for :class:`AxisGenerator` (all have sensible defaults)."""

    position: Position = "last"  # "last" = generation-token state; "mean" = avg tokens
    dtype: str = "bfloat16"
    device_map: str = "auto"
    batch_size: int = 8
    max_length: int = 2048

    def __post_init__(self) -> None:
        if self.position not in ("last", "mean"):
            raise ValueError(f"position must be 'last' or 'mean', got {self.position!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


class ActivationExtractor:
    """Captures per-layer residual-stream activations for a list of prompts.

    Wraps a loaded HF model + tokenizer. Uses ``output_hidden_states`` and drops
    the embedding layer so row ``i`` is the output of decoder block ``i``.
    """

    def __init__(self, model: Any, tokenizer: Any, config: GenerationConfig) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

    def extract(self, prompts: Sequence[str]) -> "FloatArray":
        """Return ``(n_prompts, n_layers, hidden_dim)`` float32 activations."""
        import torch

        cfg = self._config
        device = next(self._model.parameters()).device
        chunks: list["FloatArray"] = []
        self._model.eval()

        for start in range(0, len(prompts), cfg.batch_size):
            batch = list(prompts[start : start + cfg.batch_size])
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                add_special_tokens=False,  # chat template already added them
            ).to(device)

            with torch.no_grad():
                res = self._model(**enc, output_hidden_states=True, use_cache=False)

            hs = res.hidden_states[1:]  # drop embedding -> n_layers entries
            stacked = torch.stack(hs, dim=0).float()  # (n_layers, batch, seq, hidden)
            mask = enc["attention_mask"].bool()

            if cfg.position == "last":
                last_idx = enc["attention_mask"].sum(dim=1) - 1  # (batch,)
                rows = torch.arange(stacked.shape[1], device=device)
                vecs = stacked[:, rows, last_idx, :]  # (n_layers, batch, hidden)
            else:  # "mean"
                m = mask[None, :, :, None]
                summed = (stacked * m).sum(dim=2)
                counts = mask.sum(dim=1).clamp(min=1)[None, :, None]
                vecs = summed / counts

            chunks.append(vecs.permute(1, 0, 2).cpu().numpy().astype(np.float32))

        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def load_model_and_tokenizer(model_name: str, config: GenerationConfig) -> tuple[Any, Any]:
    """Load an HF causal LM and tokenizer for activation work (CPU or GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    torch_dtype = getattr(torch, config.dtype, torch.float32)
    # transformers >=5 renamed `torch_dtype` -> `dtype`; older versions only
    # accept `torch_dtype`. Try the new name, fall back to the old one.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, device_map=config.device_map
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=config.device_map
        )
    return model, tokenizer


class AxisGenerator:
    """Generate an assistant axis for a model from a :class:`BehaviorCatalog`."""

    def __init__(
        self,
        model_name: str,
        *,
        catalog: Optional[BehaviorCatalog] = None,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.catalog: BehaviorCatalog = catalog or default_catalog()
        self.config: GenerationConfig = config or GenerationConfig()
        self.catalog.validate()
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model, self._tokenizer = load_model_and_tokenizer(self.model_name, self.config)

    def _render(self, behavior: Behavior, probe_text: str) -> str:
        messages = [
            {"role": "system", "content": behavior.system_prompt},
            {"role": "user", "content": probe_text},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # Models without a system role: fold the persona into the user turn.
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{behavior.system_prompt}\n\n{probe_text}"}],
                add_generation_prompt=True,
                tokenize=False,
            )

    def _prompts_for(self, behaviors: Sequence[Behavior]) -> list[str]:
        probes = self.catalog.probe_texts()
        return [self._render(b, text) for b in behaviors for text in probes]

    def generate(self) -> Axis:
        self.load()
        extractor = ActivationExtractor(self._model, self._tokenizer, self.config)

        assistant = self.catalog.assistant_behaviors
        characters = self.catalog.character_behaviors
        assistant_acts = extractor.extract(self._prompts_for(assistant))
        role_acts = extractor.extract(self._prompts_for(characters))

        directions = compute_axis_from_activations(assistant_acts, role_acts)
        metadata: dict[str, Any] = {
            "model": self.model_name,
            "method": "mean(assistant) - mean(role)",
            "position": self.config.position,
            "n_assistant_behaviors": len(assistant),
            "n_character_behaviors": len(characters),
            "n_probes": len(self.catalog.probes),
        }
        return Axis(directions=directions, metadata=metadata, source=f"generated:{self.model_name}")

    def generate_and_save(self, save_path: str) -> Axis:
        axis = self.generate()
        save_axis(axis, save_path, metadata=axis.metadata)
        return axis


def generate_axis(
    model_name: str,
    *,
    catalog: Optional[BehaviorCatalog] = None,
    position: Position = "last",
    device_map: str = "auto",
    dtype: str = "bfloat16",
    batch_size: int = 8,
    max_length: int = 2048,
    save_path: Optional[str] = None,
) -> Axis:
    """Functional convenience wrapper around :class:`AxisGenerator`."""
    config = GenerationConfig(
        position=position,
        dtype=dtype,
        device_map=device_map,
        batch_size=batch_size,
        max_length=max_length,
    )
    generator = AxisGenerator(model_name, catalog=catalog, config=config)
    if save_path:
        return generator.generate_and_save(save_path)
    return generator.generate()


def _build_catalog(roles: Optional[str], questions_path: Optional[str]) -> BehaviorCatalog:
    """Construct the catalog for the CLI: defaults plus any extra behaviors."""
    catalog = default_catalog()
    if roles == "submodule":
        for behavior in load_assistant_axis_roles():
            catalog.add(behavior, replace_existing=True)
    elif roles and roles != "default":
        for behavior in load_behaviors_from_json(roles, kind=BehaviorKind.CHARACTER):
            catalog.add(behavior, replace_existing=True)
    if questions_path:
        import json

        with open(questions_path) as f:
            texts = json.load(f)
        # Replace the default probes with the supplied ones.
        fresh = BehaviorCatalog(list(catalog))
        for text in texts:
            fresh.add_probe_text(str(text))
        catalog = fresh
    return catalog


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an assistant-axis vector file.")
    ap.add_argument("--model", required=True, help="HF model repo id or local path")
    ap.add_argument("--out", required=True, help="output path (.pt / .npz / .npy)")
    ap.add_argument(
        "--roles",
        default="default",
        help="'default' (built-in), 'submodule' (assistant-axis's 275 roles), "
        "or a path to a JSON file of personas to add",
    )
    ap.add_argument("--questions", default=None, help="JSON file of probe questions")
    ap.add_argument("--position", choices=["last", "mean"], default="last")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    catalog = _build_catalog(args.roles, args.questions)
    config = GenerationConfig(
        position=args.position,
        dtype=args.dtype,
        device_map=args.device_map,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    generator = AxisGenerator(args.model, catalog=catalog, config=config)
    print(f"generating axis for {args.model} from {catalog!r} ...")
    axis = generator.generate_and_save(args.out)
    print(
        f"wrote {args.out}: axis {axis.directions.shape} "
        f"(layers={axis.n_layers}, hidden={axis.hidden_dim})"
    )


if __name__ == "__main__":
    main()
