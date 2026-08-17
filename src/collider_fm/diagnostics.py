from __future__ import annotations

"""Shared helpers for diagnostics scripts and notebooks."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ColliderMLDataset, collate_fn


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
    """Create a DataLoader for ColliderML diagnostics workflows.

    Args:
        split (str): Project split alias (e.g. `"val"` or `"val[:100]"`).
        batch_size (int): Batch size.
        dataset_name (str, optional): Hugging Face dataset name. Defaults to
            `"CERN/ColliderML-Release-1"`.
        dataset_type (str, optional): Dataset type. Defaults to `"ttbar"`.
        pu_config (str, optional): Pile-up config. Defaults to `"pu0"`.
        cache_dir (str, optional): HF cache directory. Defaults to the cluster
            path.
        dataset_revision (str | None, optional): HF dataset revision. Defaults
            to None.
        local_files_only (bool, optional): Whether to load from cache only.
            Defaults to False.

    Returns:
        DataLoader: A non-shuffling DataLoader over the calo_hits split.
    """
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


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
    """Load one batch of events for diagnostics or notebook exploration.

    Args:
        split (str): Project split alias (e.g. `"val"` or `"val[:100]"`).
        batch_size (int, optional): Batch size. Defaults to 64.
        dataset_name (str, optional): Hugging Face dataset name. Defaults to
            `"CERN/ColliderML-Release-1"`.
        dataset_type (str, optional): Dataset type. Defaults to `"ttbar"`.
        pu_config (str, optional): Pile-up config. Defaults to `"pu0"`.
        cache_dir (str, optional): HF cache directory. Defaults to the cluster
            path.
        dataset_revision (str | None, optional): HF dataset revision. Defaults
            to None.
        local_files_only (bool, optional): Whether to load from cache only.
            Defaults to False.

    Returns:
        list[dict[str, Any]]: The first batch of raw event dicts.
    """
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
    """Load a checkpoint into the model and report key mismatches.

    Args:
        model (torch.nn.Module): Model to load the checkpoint into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        dict[str, Any]: Dict with `checkpoint_path`, `missing_keys`, and
        `unexpected_keys` from the non-strict `load_state_dict` call.
    """
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
    """Detach a tensor and move it to NumPy for plotting.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: The detached, CPU-resident NumPy array.
    """
    return tensor.detach().cpu().numpy()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Return lightweight shape and summary statistics for a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        dict[str, Any]: Dict with `shape`, `min`, `max`, `mean`, and `std`.
    """
    array = tensor.detach().cpu()
    return {
        "shape": list(array.shape),
        "min": float(array.min().item()),
        "max": float(array.max().item()),
        "mean": float(array.mean().item()),
        "std": float(array.std().item()) if array.numel() > 1 else 0.0,
    }


def compute_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project a 2D feature matrix onto the leading principal components.

    Args:
        features (np.ndarray): Feature matrix of shape `[N, D]`.
        n_components (int, optional): Number of principal components. Defaults
            to 2.

    Returns:
        np.ndarray: Projected features of shape `[N, n_components]`.

    Raises:
        ValueError: If `features` is not 2-D or has zero samples.
    """
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


@torch.no_grad()
def encode_view(
    model: torch.nn.Module,
    view: dict[str, torch.Tensor],
    use_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode one point view through the student or teacher pathway.

    The return value keeps a stable, plotting-friendly field set so the notebook
    and scripts do not need to care about internal model naming details.

    Args:
        model (torch.nn.Module): Sonata model with `encode_student_view` /
            `encode_teacher_view` methods.
        view (dict[str, torch.Tensor]): Point view to encode.
        use_teacher (bool, optional): Whether to use the teacher pathway.
            Defaults to False.

    Returns:
        dict[str, torch.Tensor]: Plotting-friendly encoded outputs with keys
        `point_features`, `point_projection`, `pooled`, `projection`,
        `logits`, `point_logits`, `offset`, `source_index`, and `mask`.
    """
    encoded = model.encode_teacher_view(view) if use_teacher else model.encode_student_view(view)
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
