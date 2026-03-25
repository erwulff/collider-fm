from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ColliderMLDataset, collate_fn
from .model import PandaSelfDistillation, as_point_cloud, mean_pool_features


def create_dataloader(
    split: str,
    batch_size: int,
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
) -> DataLoader:
    dataset = ColliderMLDataset(
        split=split,
        dataset_type=dataset_type,
        pu_config=pu_config,
        object_types=["calo_hits"],
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )


def load_events(
    split: str,
    batch_size: int = 64,
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
) -> list[dict[str, Any]]:
    dataloader = create_dataloader(
        split=split,
        batch_size=batch_size,
        dataset_type=dataset_type,
        pu_config=pu_config,
        cache_dir=cache_dir,
    )
    return next(iter(dataloader))


def load_checkpoint(
    model: PandaSelfDistillation, checkpoint_path: str
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint_path": checkpoint_path,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def radius(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.linalg.norm(
        torch.stack([values["x"], values["y"], values["z"]], dim=1), dim=1
    )


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu()
    return {
        "shape": list(array.shape),
        "min": float(array.min().item()),
        "max": float(array.max().item()),
        "mean": float(array.mean().item()),
        "std": float(array.std().item()) if array.numel() > 1 else 0.0,
    }


def compute_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    if features.shape[0] == 0:
        raise ValueError("PCA requires at least one sample.")

    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1 or centered.shape[1] == 1:
        result = np.zeros((centered.shape[0], n_components), dtype=np.float32)
        result[:, : min(n_components, centered.shape[1])] = centered[
            :, : min(n_components, centered.shape[1])
        ]
        return result

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


def sample_indices(num_items: int, max_items: int, seed: int) -> np.ndarray:
    if num_items <= max_items:
        return np.arange(num_items)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_items, size=max_items, replace=False))


@torch.no_grad()
def encode_view(
    model: PandaSelfDistillation,
    view: dict[str, torch.Tensor],
    use_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    point = as_point_cloud(view, default_grid_size=model.grid_size)
    backbone = model.teacher_backbone if use_teacher else model.student_backbone
    projector = model.teacher_projector if use_teacher else model.student_projector
    predictor = None if use_teacher else model.student_predictor

    encoded = backbone(point)
    pooled = mean_pool_features(encoded.feat, getattr(encoded, "offset", None))
    projection = projector(pooled)
    if predictor is not None:
        projection = predictor(projection)
    projection = F.normalize(projection, dim=-1)
    logits = model.prototype_head(projection)
    return {
        "point_features": encoded.feat,
        "pooled": pooled,
        "projection": projection,
        "logits": logits,
        "offset": point.offset,
    }
