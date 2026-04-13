from __future__ import annotations

"""Shared helpers for diagnostics scripts and notebooks."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ColliderMLDataset, collate_fn
from .model import PandaSelfDistillation


def create_dataloader(
    split: str,
    batch_size: int,
    dataset_name: str = "CERN/ColliderML-Release-1",
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
    dataset_revision: str | None = None,
    local_files_only: bool = False,
) -> DataLoader:
    """Create a dataloader for ColliderML diagnostics workflows."""
    dataset = ColliderMLDataset(
        dataset_name=dataset_name,
        split=split,
        dataset_type=dataset_type,
        pu_config=pu_config,
        object_types=["calo_hits"],
        cache_dir=cache_dir,
        dataset_revision=dataset_revision,
        local_files_only=local_files_only,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )


def load_events(
    split: str,
    batch_size: int = 64,
    dataset_name: str = "CERN/ColliderML-Release-1",
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
    dataset_revision: str | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load one batch of events for diagnostics or notebook exploration."""
    dataloader = create_dataloader(
        split=split,
        batch_size=batch_size,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        pu_config=pu_config,
        cache_dir=cache_dir,
        dataset_revision=dataset_revision,
        local_files_only=local_files_only,
    )
    return next(iter(dataloader))


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """Load a checkpoint into the model and report key mismatches."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint_path": checkpoint_path,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor and move it to NumPy for plotting."""
    return tensor.detach().cpu().numpy()


def radius(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the radial distance for hit dictionaries with x/y/z coordinates."""
    return torch.linalg.norm(
        torch.stack([values["x"], values["y"], values["z"]], dim=1), dim=1
    )


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Return lightweight shape and summary statistics for a tensor."""
    array = tensor.detach().cpu()
    return {
        "shape": list(array.shape),
        "min": float(array.min().item()),
        "max": float(array.max().item()),
        "mean": float(array.mean().item()),
        "std": float(array.std().item()) if array.numel() > 1 else 0.0,
    }


def compute_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project a 2D feature matrix onto the leading principal components."""
    if features.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    if features.shape[0] == 0:
        raise ValueError("PCA requires at least one sample.")

    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1 or centered.shape[1] == 1:
        result = np.zeros((centered.shape[0], n_components), dtype=np.float32)
        limit = min(n_components, centered.shape[1])
        result[:, :limit] = centered[:, :limit]
        return result

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T


def sample_indices(num_items: int, max_items: int, seed: int) -> np.ndarray:
    """Pick a reproducible subset of indices when a point cloud is too large to plot."""
    if num_items <= max_items:
        return np.arange(num_items)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_items, size=max_items, replace=False))


@torch.no_grad()
def encode_view(
    model: torch.nn.Module,
    view: dict[str, torch.Tensor],
    use_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode one point-view through the student or teacher pathway.

    The return value keeps a stable, plotting-friendly field set so the notebook and
    scripts do not need to care about internal model naming details.
    """
    encoded = (
        model.encode_teacher_view(view)
        if use_teacher
        else model.encode_student_view(view)
    )
    return {
        "point_features": encoded["point_features"],
        "point_projection": encoded["point_projection"],
        "pooled": encoded["pooled"],
        "projection": encoded["masked_pooled_projection"],
        "logits": encoded["masked_logits"],
        "point_logits": encoded["point_logits"],
        "offset": encoded["offset"],
        "source_index": encoded["source_index"],
        "mask": encoded["mask"],
    }
