"""Train a sparse autoencoder on residual activations and label its features.

The recipe is Anthropic's (Bricken et al. 2023; Templeton et al. 2024, "Scaling
Monosemanticity"): a one-hidden-layer ReLU autoencoder with many more features
than residual dimensions, trained to reconstruct activations under an L1
sparsity penalty weighted by decoder-row norm::

    f = relu(W_enc (x - b_dec) + b_enc)
    x̂ = W_dec f + b_dec
    L = ||x - x̂||² + λ Σ_i f_i ||W_dec[i]||

Features are then described by the tokens they fire on most, which is how the
Golden Gate Bridge feature was found. Needs torch (``generate`` extra).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..axis.generate import TokenActivations
from .dictionary import FeatureDictionary, FloatArray


@dataclass(frozen=True, slots=True)
class SAEConfig:
    n_features: int = 4096
    l1_coef: float = 5.0
    lr: float = 1e-3
    steps: int = 5000
    batch_size: int = 1024
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.n_features < 1 or self.steps < 1 or self.batch_size < 1:
            raise ValueError("n_features, steps and batch_size must be >= 1")
        if self.l1_coef < 0:
            raise ValueError("l1_coef must be >= 0")


@dataclass(frozen=True, slots=True)
class FeatureExample:
    """A token that fired a feature, with its surrounding context."""

    token: str
    activation: float
    context: str


def train_sae(
    activations: FloatArray,
    *,
    layer: int,
    n_layers: int,
    config: SAEConfig | None = None,
    log_every: int = 0,
) -> FeatureDictionary:
    """Fit a sparse autoencoder on ``(n_samples, hidden)`` activations.

    Inputs are rescaled so the mean squared norm equals ``hidden`` (as in the
    paper) and the scale is folded back into the returned weights, so the
    dictionary encodes raw activations.
    """
    import torch

    cfg = config or SAEConfig()
    x_all = torch.as_tensor(np.asarray(activations, dtype=np.float32))
    if x_all.ndim != 2:
        raise ValueError("activations must be (n_samples, hidden)")
    n_samples, hidden = x_all.shape
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    scale = float(torch.sqrt(hidden / (x_all.pow(2).sum(dim=1).mean() + 1e-8)))
    x_all = (x_all * scale).to(device)

    w_dec = torch.randn(cfg.n_features, hidden, device=device)
    w_dec = 0.1 * w_dec / w_dec.norm(dim=1, keepdim=True)
    w_dec.requires_grad_(True)
    w_enc = w_dec.detach().T.clone().requires_grad_(True)
    b_enc = torch.zeros(cfg.n_features, device=device, requires_grad=True)
    b_dec = x_all.mean(dim=0).clone().requires_grad_(True)
    opt = torch.optim.Adam([w_enc, b_enc, w_dec, b_dec], lr=cfg.lr)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, n_samples, (min(cfg.batch_size, n_samples),), generator=gen)
        x = x_all[idx.to(device)]
        f = torch.relu((x - b_dec) @ w_enc + b_enc)
        recon = f @ w_dec + b_dec
        mse = (recon - x).pow(2).sum(dim=1).mean()
        l1 = (f * w_dec.norm(dim=1)).sum(dim=1).mean()
        loss = mse + cfg.l1_coef * l1
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            l0 = float((f > 0).float().sum(dim=1).mean())
            print(f"step {step}/{cfg.steps}  mse {mse.item():.4f}  l1 {l1.item():.4f}  L0 {l0:.1f}")

    with torch.no_grad():
        f_all = torch.relu((x_all - b_dec) @ w_enc + b_enc)
        max_acts = f_all.max(dim=0).values
        # ponytail: dead features are kept (no resampling); re-train with a lower
        # l1_coef or more data if too many come back with max_activation 0.
        return FeatureDictionary(
            (w_enc * scale).cpu().numpy(),
            b_enc.cpu().numpy(),
            (w_dec / scale).cpu().numpy(),
            (b_dec / scale).cpu().numpy(),
            layer=layer,
            n_layers=n_layers,
            max_activations=max_acts.cpu().numpy(),
            metadata={
                "n_features": cfg.n_features,
                "l1_coef": cfg.l1_coef,
                "steps": cfg.steps,
                "n_samples": int(n_samples),
                "dead_features": int((max_acts == 0).sum()),
            },
        )


def label_features(
    dictionary: FeatureDictionary,
    corpus: Sequence[TokenActivations],
    *,
    n_examples: int = 8,
    context: int = 6,
    label_tokens: int = 3,
) -> dict[int, list[FeatureExample]]:
    """Find each feature's top-activating tokens and label it by them.

    Returns the examples per feature and sets ``dictionary.labels`` to the most
    frequent tokens among them (e.g. ``"bridge / Golden / Gate"``). Edit the
    labels by hand, or with an LLM, before saving for a nicer dictionary.
    """
    heaps: list[list[tuple[float, str, str]]] = [[] for _ in range(dictionary.n_features)]
    for item in corpus:
        acts = dictionary.encode(item.activations)  # (seq, n_features)
        positions, feats = np.nonzero(acts)
        for pos, feat in zip(positions.tolist(), feats.tolist()):
            lo, hi = max(0, pos - context), min(len(item.tokens), pos + context + 1)
            snippet = (
                "".join(item.tokens[lo:pos])
                + f"[[{item.tokens[pos]}]]"
                + "".join(item.tokens[pos + 1 : hi])
            )
            heap = heaps[feat]
            heap.append((float(acts[pos, feat]), item.tokens[pos], snippet))
            if len(heap) > 4 * n_examples:
                heap.sort(reverse=True)
                del heap[n_examples:]

    examples: dict[int, list[FeatureExample]] = {}
    for feat, heap in enumerate(heaps):
        if not heap:
            continue
        heap.sort(reverse=True)
        top = heap[:n_examples]
        examples[feat] = [FeatureExample(tok, act, ctx) for act, tok, ctx in top]
        common = Counter(tok.strip() or tok for _, tok, _ in top).most_common(label_tokens)
        dictionary.labels[feat] = " / ".join(tok for tok, _ in common)
    return examples


def examples_to_json(examples: dict[int, list[FeatureExample]]) -> dict[str, Any]:
    return {
        str(k): [{"token": e.token, "activation": e.activation, "context": e.context} for e in v]
        for k, v in examples.items()
    }
