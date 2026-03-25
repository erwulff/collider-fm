from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ColliderMLDataset, collate_fn
from .model import MultimodalPandaSSL, PandaSelfDistillation, as_point_cloud, mean_pool_features


def create_dataloader(
    split: str,
    batch_size: int,
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
) -> DataLoader:
    """Create a dataloader for ColliderML diagnostics workflows."""
    dataset = ColliderMLDataset(
        split=split,
        dataset_type=dataset_type,
        pu_config=pu_config,
        object_types=["tracker_hits", "calo_hits", "particles"],
        cache_dir=cache_dir,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def load_events(
    split: str,
    batch_size: int = 64,
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
) -> list[dict[str, Any]]:
    """Load one batch of events for diagnostics or notebook exploration."""
    dataloader = create_dataloader(
        split=split,
        batch_size=batch_size,
        dataset_type=dataset_type,
        pu_config=pu_config,
        cache_dir=cache_dir,
    )
    return next(iter(dataloader))


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """Load a checkpoint into a model and report key mismatches."""
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
    return torch.linalg.norm(torch.stack([values["x"], values["y"], values["z"]], dim=1), dim=1)


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
    model: PandaSelfDistillation,
    view: Mapping[str, torch.Tensor],
    use_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode one legacy point-view through the student or teacher pathway."""
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


@torch.no_grad()
def encode_ssl_view(
    model: MultimodalPandaSSL,
    view: Mapping[str, Any],
    use_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode one structured multimodal SSL view through the student or teacher path."""
    if use_teacher:
        tracker_stem = model.teacher_tracker_stem
        calo_stem = model.teacher_calo_stem
        fusion = model.teacher_fusion
        backbone = model.teacher_backbone
        projector = model.teacher_projector
        predictor = None
    else:
        tracker_stem = model.student_tracker_stem
        calo_stem = model.student_calo_stem
        fusion = model.student_fusion
        backbone = model.student_backbone
        projector = model.student_projector
        predictor = model.student_predictor

    tracker_continuous = torch.as_tensor(view["tracker_continuous"], dtype=torch.float32)
    calo_continuous = torch.as_tensor(view["calo_continuous"], dtype=torch.float32, device=tracker_continuous.device)
    tracker_features = tracker_stem(tracker_continuous, dict(view["tracker_categorical"]))
    calo_features = calo_stem(calo_continuous, dict(view["calo_categorical"]))
    fused_features = fusion(
        tracker_features=tracker_features,
        calo_features=calo_features,
        modality_id=torch.as_tensor(view["modality_id"], dtype=torch.long, device=tracker_features.device),
        tracker_index=view.get("tracker_index"),
        calo_index=view.get("calo_index"),
    )

    point = as_point_cloud(
        {
            "coord": view["coord"],
            "feat": fused_features,
            "offset": view["offset"],
            "grid_size": view.get("grid_size", model.grid_size),
        },
        default_grid_size=model.grid_size,
    )
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
        "point_id": torch.as_tensor(view["point_id"], dtype=torch.long, device=encoded.feat.device),
        "modality_id": torch.as_tensor(view["modality_id"], dtype=torch.long, device=encoded.feat.device),
    }
