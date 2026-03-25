from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch

DEFAULT_POINT_GRID_SIZE = 10.0
POINT_FEATURE_DIM = 4


class PointView(TypedDict, total=False):
    coord: torch.Tensor
    feat: torch.Tensor
    offset: torch.Tensor
    grid_size: torch.Tensor
    source_index: torch.Tensor
    energy: torch.Tensor
    patch_id: torch.Tensor
    mask: torch.Tensor


def normalize_feature(values: torch.Tensor) -> torch.Tensor:
    """Scale a 1D feature to a stable range using its largest absolute value."""
    scale = values.abs().max().clamp_min(1.0)
    return values / scale


def _normalize_grid_size(grid_size: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(grid_size, dtype=torch.float32, device=device).flatten()
    if tensor.numel() != 1:
        raise ValueError("'grid_size' must be a scalar.")
    return tensor[0]


def _require_hit_tensor(hits: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    if key not in hits:
        raise KeyError(f"Missing required hit field '{key}'.")
    return torch.as_tensor(hits[key], dtype=torch.float32, device=device).flatten()


def _resolve_calo_energy(calo_hits: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    for key in ("energy", "totalenergy", "total_energy"):
        if key in calo_hits:
            return torch.as_tensor(calo_hits[key], dtype=torch.float32, device=device).flatten()
    raise KeyError("Calorimeter hits must provide 'energy', 'totalenergy', or 'total_energy'.")


def _default_source_index(num_points: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_points, device=device, dtype=torch.long)


def _default_patch_id(coord: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
    grid_coord = torch.div(coord - coord.min(dim=0).values, grid_size, rounding_mode="floor").long()
    base = grid_coord.max(dim=0).values + 1
    stride_y = base[2].clamp_min(1)
    stride_x = (base[1] * stride_y).clamp_min(1)
    return grid_coord[:, 0] * stride_x + grid_coord[:, 1] * stride_y + grid_coord[:, 2]


def assemble_point_features(coord: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    """Build the project-standard per-point feature tensor `[x, y, z, energy]`."""
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")

    energy = torch.as_tensor(energy, dtype=torch.float32, device=coord.device).flatten()
    if energy.shape[0] != coord.shape[0]:
        raise ValueError("'energy' must contain one value per point.")

    return torch.cat([coord, energy.unsqueeze(1)], dim=1)


def validate_point_view(view: Mapping[str, Any]) -> PointView:
    """Normalize and validate a point-view mapping used by the model pipeline."""
    required_keys = {"coord", "feat"}
    missing_keys = required_keys.difference(view.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Point view is missing required keys: {missing}.")

    coord = torch.as_tensor(view["coord"], dtype=torch.float32)
    feat = torch.as_tensor(view["feat"], dtype=torch.float32, device=coord.device)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")
    if feat.ndim != 2 or feat.shape[1] != POINT_FEATURE_DIM:
        raise ValueError(f"'feat' must have shape [num_points, {POINT_FEATURE_DIM}].")
    if feat.shape[0] != coord.shape[0]:
        raise ValueError("'coord' and 'feat' must have the same number of points.")

    offset_value = view.get("offset", [coord.shape[0]])
    offset = torch.as_tensor(offset_value, dtype=torch.long, device=coord.device).flatten()
    if offset.numel() == 0:
        raise ValueError("'offset' must contain at least one event boundary.")
    if offset[-1].item() != coord.shape[0]:
        raise ValueError("The final offset must equal the number of points.")

    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    if torch.any(counts <= 0):
        raise ValueError("'offset' must be a strictly increasing cumulative count.")

    grid_size = _normalize_grid_size(view.get("grid_size", DEFAULT_POINT_GRID_SIZE), coord.device)
    energy = torch.as_tensor(view.get("energy", feat[:, 3]), dtype=torch.float32, device=coord.device).flatten()
    if energy.shape[0] != coord.shape[0]:
        raise ValueError("'energy' must contain one value per point.")

    source_index = torch.as_tensor(
        view.get("source_index", _default_source_index(coord.shape[0], coord.device)),
        dtype=torch.long,
        device=coord.device,
    ).flatten()
    if source_index.shape[0] != coord.shape[0]:
        raise ValueError("'source_index' must contain one value per point.")

    patch_id = torch.as_tensor(
        view.get("patch_id", _default_patch_id(coord, grid_size)),
        dtype=torch.long,
        device=coord.device,
    ).flatten()
    if patch_id.shape[0] != coord.shape[0]:
        raise ValueError("'patch_id' must contain one value per point.")

    mask = torch.as_tensor(
        view.get("mask", torch.zeros(coord.shape[0], device=coord.device)),
        dtype=torch.bool,
        device=coord.device,
    ).flatten()
    if mask.shape[0] != coord.shape[0]:
        raise ValueError("'mask' must contain one value per point.")

    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "grid_size": grid_size,
        "source_index": source_index,
        "energy": energy,
        "patch_id": patch_id,
        "mask": mask,
    }


def sample_hit_indices(num_hits: int, max_hits: int | None, device: torch.device) -> torch.Tensor:
    """Return evenly spaced hit indices, optionally downsampling to ``max_hits``."""
    if max_hits is None or num_hits <= max_hits:
        return torch.arange(num_hits, device=device)
    return torch.linspace(0, num_hits - 1, steps=max_hits, device=device).round().long()


def build_point_view_from_event(
    event: Mapping[str, Any],
    device: torch.device,
    max_calo_hits: int | None = None,
    grid_size: float = DEFAULT_POINT_GRID_SIZE,
) -> PointView:
    """Convert one raw calorimeter event into the point-view format expected by the model."""
    calo_hits = event["calo_hits"]
    source_device = torch.as_tensor(calo_hits["x"]).device
    if len(calo_hits["x"]) == 0:
        raise ValueError("Cannot build a point view from an event with zero calorimeter hits.")
    calo_indices = sample_hit_indices(
        num_hits=len(calo_hits["x"]),
        max_hits=max_calo_hits,
        device=source_device,
    )

    coord = torch.stack(
        [
            _require_hit_tensor(calo_hits, "x", source_device)[calo_indices],
            _require_hit_tensor(calo_hits, "y", source_device)[calo_indices],
            _require_hit_tensor(calo_hits, "z", source_device)[calo_indices],
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)

    energy = _resolve_calo_energy(calo_hits, device=source_device)[calo_indices].to(device=device, dtype=torch.float32)
    source_index = torch.as_tensor(calo_indices, dtype=torch.long, device=device)
    grid_size_tensor = torch.tensor(grid_size, dtype=torch.float32, device=device)

    return validate_point_view(
        {
            "coord": coord,
            "feat": assemble_point_features(coord, energy),
            "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
            "grid_size": grid_size_tensor,
            "source_index": source_index,
            "energy": energy,
            "patch_id": _default_patch_id(coord, grid_size_tensor),
            "mask": torch.zeros(coord.shape[0], dtype=torch.bool, device=device),
        }
    )


def augment_point_view(
    view: Mapping[str, torch.Tensor],
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
) -> PointView:
    """Create a noisy copy of a calorimeter point view for self-distillation training."""
    base_view = validate_point_view(view)
    coord = base_view["coord"].clone()
    energy = base_view["energy"].clone()

    coord = coord + torch.randn_like(coord) * coord_noise_scale
    energy = (energy * (1.0 + torch.randn_like(energy) * feat_noise_scale)).clamp_min(0.0)

    return {
        "coord": coord,
        "feat": assemble_point_features(coord, energy),
        "offset": base_view["offset"].clone(),
        "grid_size": base_view["grid_size"].clone(),
        "source_index": base_view["source_index"].clone(),
        "energy": energy,
        "patch_id": _default_patch_id(coord, base_view["grid_size"]),
        "mask": base_view["mask"].clone(),
    }


def batch_point_views(views: Sequence[Mapping[str, torch.Tensor]]) -> PointView:
    """Concatenate per-event point views into one batched point-cloud mapping."""
    if not views:
        raise ValueError("At least one point view is required to create a batch.")

    normalized_views = [validate_point_view(view) for view in views]
    grid_size = normalized_views[0]["grid_size"]
    for view in normalized_views[1:]:
        if not torch.allclose(view["grid_size"], grid_size):
            raise ValueError("All point views in a batch must share the same grid size.")

    coord = torch.cat([view["coord"] for view in normalized_views], dim=0)
    feat = torch.cat([view["feat"] for view in normalized_views], dim=0)
    energy = torch.cat([view["energy"] for view in normalized_views], dim=0)
    source_index_parts = []
    patch_id_parts = []
    source_offset = 0
    patch_offset = 0
    for view in normalized_views:
        source_index_parts.append(view["source_index"] + source_offset)
        patch_id_parts.append(view["patch_id"] + patch_offset)
        source_offset += int(view["source_index"].max().item()) + 1
        patch_offset += int(view["patch_id"].max().item()) + 1
    source_index = torch.cat(source_index_parts, dim=0)
    patch_id = torch.cat(patch_id_parts, dim=0)
    mask = torch.cat([view["mask"] for view in normalized_views], dim=0)
    counts = torch.tensor(
        [view["coord"].shape[0] for view in normalized_views],
        device=coord.device,
        dtype=torch.long,
    )
    return {
        "coord": coord,
        "feat": feat,
        "offset": torch.cumsum(counts, dim=0),
        "grid_size": grid_size.clone(),
        "source_index": source_index,
        "energy": energy,
        "patch_id": patch_id,
        "mask": mask,
    }


def build_distillation_views(
    events: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_calo_hits: int | None = None,
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
    num_augmentations: int = 2,
) -> list[PointView]:
    """Create independently augmented batched views from the same calorimeter events."""
    if num_augmentations < 2:
        raise ValueError("Self-distillation requires at least two augmented views.")

    base_views = [
        build_point_view_from_event(
            event,
            device=device,
            max_calo_hits=max_calo_hits,
        )
        for event in events
    ]

    def build_augmented_batch() -> PointView:
        return batch_point_views(
            [
                augment_point_view(
                    view,
                    coord_noise_scale=coord_noise_scale,
                    feat_noise_scale=feat_noise_scale,
                )
                for view in base_views
            ]
        )

    return [build_augmented_batch() for _ in range(num_augmentations)]


def make_random_view(num_points: int, in_channels: int, device: torch.device) -> PointView:
    """Generate a synthetic calorimeter point view for quick smoke tests."""
    if in_channels != POINT_FEATURE_DIM:
        raise ValueError(f"Random views in this project require {POINT_FEATURE_DIM} input channels.")

    coord = torch.rand(num_points, 3, device=device)
    energy = torch.rand(num_points, device=device)
    grid_size = torch.tensor(DEFAULT_POINT_GRID_SIZE, device=device)
    return {
        "coord": coord,
        "feat": assemble_point_features(coord, energy),
        "offset": torch.tensor([num_points], dtype=torch.long, device=device),
        "grid_size": grid_size,
        "source_index": _default_source_index(num_points, device),
        "energy": energy,
        "patch_id": _default_patch_id(coord, grid_size),
        "mask": torch.zeros(num_points, dtype=torch.bool, device=device),
    }
