"""Pure metric primitives shared by training and evaluation.

Deliberately dependency-light: this module imports only ``torch``. Both the training
loop (``training_loop.py``) and the label-free evaluation harness (``evaluation.py``)
need these, and routing the eval import through ``training_loop`` would drag in Ray,
matplotlib and DDP for a handful of pure-tensor helpers. Keep it that way -- adding a
heavy import here re-couples evaluation to the training stack.

DDP- and matplotlib-coupled metric helpers (e.g. ``reduce_scalar``, the ``plot_*``
functions) intentionally stay in ``training_loop.py``.
"""

from __future__ import annotations

import torch

__all__ = [
    "prototype_usage",
    "prototype_entropy",
    "embedding_norm",
    "feature_std",
]


def prototype_usage(logits: torch.Tensor, num_prototypes: int) -> torch.Tensor:
    """Return normalized prototype assignment counts from argmax of logits.

    Args:
        logits (torch.Tensor): Prototype logits of shape `[N, num_prototypes]`.
        num_prototypes (int): Number of prototypes.

    Returns:
        torch.Tensor: Normalized usage distribution of shape `[num_prototypes]`.
    """
    assignments = logits.argmax(dim=-1)
    counts = torch.bincount(assignments, minlength=num_prototypes).to(dtype=torch.float32)
    return counts / counts.sum().clamp_min(1.0)


def prototype_entropy(probabilities: torch.Tensor) -> float:
    """Return the Shannon entropy of a prototype usage distribution.

    Args:
        probabilities (torch.Tensor): Usage probabilities of shape
            `[num_prototypes]`.

    Returns:
        float: Entropy `-sum p log p` (max = `log(K)`).
    """
    p = probabilities.clamp_min(1.0e-8)
    return float(-(p * p.log()).sum().item())


def embedding_norm(embeddings: torch.Tensor | None) -> float:
    """Return the mean L2 norm of embeddings (0 if None or empty).

    Args:
        embeddings (torch.Tensor | None): Embeddings of shape `[N, D]`, or None.

    Returns:
        float: Mean L2 norm across the last dimension.
    """
    if embeddings is None or embeddings.numel() == 0:
        return 0.0
    return float(embeddings.norm(dim=-1).mean().item())


def feature_std(features: torch.Tensor | None) -> float:
    """Return the mean per-feature standard deviation (0 if None or empty).

    Args:
        features (torch.Tensor | None): Features of shape `[N, D]`, or None.

    Returns:
        float: Mean standard deviation across the feature dimension.
    """
    if features is None or features.numel() == 0:
        return 0.0
    return float(features.std(dim=0).mean().item())
