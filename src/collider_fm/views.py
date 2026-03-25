from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch


DEFAULT_POINT_GRID_SIZE = 10.0
POINT_FEATURE_DIM = 6


class PointView(TypedDict):
    coord: torch.Tensor
    feat: torch.Tensor
    offset: torch.Tensor
    grid_size: torch.Tensor


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


def assemble_point_features(
    coord: torch.Tensor,
    signal: torch.Tensor,
    detector_type: torch.Tensor,
) -> torch.Tensor:
    """Build the six-channel per-point feature tensor from its source components."""
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")

    signal = torch.as_tensor(signal, dtype=torch.float32, device=coord.device).flatten()
    detector_type = torch.as_tensor(detector_type, dtype=torch.float32, device=coord.device).flatten()
    if signal.shape[0] != coord.shape[0]:
        raise ValueError("'signal' must contain one value per point.")
    if detector_type.shape[0] != coord.shape[0]:
        raise ValueError("'detector_type' must contain one value per point.")

    radius = normalize_feature(torch.linalg.norm(coord, dim=1))
    return torch.cat(
        [coord, radius.unsqueeze(1), signal.unsqueeze(1), detector_type.unsqueeze(1)],
        dim=1,
    )


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
    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "grid_size": grid_size,
    }


def sample_hit_indices(num_hits: int, max_hits: int | None, device: torch.device) -> torch.Tensor:
    """Return evenly spaced hit indices, optionally downsampling to ``max_hits``."""
    if max_hits is None or num_hits <= max_hits:
        return torch.arange(num_hits, device=device)
    return torch.linspace(0, num_hits - 1, steps=max_hits, device=device).round().long()


def build_point_view_from_event(
    event: Mapping[str, Any],
    device: torch.device,
    max_tracker_hits: int | None = None,
    max_calo_hits: int | None = None,
    grid_size: float = DEFAULT_POINT_GRID_SIZE,
) -> PointView:
    """Convert one raw event into the point-view format expected by the model.

    The returned mapping matches the Panda point-cloud convention used elsewhere in
    the repo: coordinates, per-point features, cumulative offsets, and grid size.
    """
    tracker_hits = event["tracker_hits"]
    calo_hits = event["calo_hits"]
    source_device = torch.as_tensor(tracker_hits["x"]).device
    if torch.as_tensor(calo_hits["x"]).device != source_device:
        raise ValueError("Tracker-hit and calo-hit tensors must be on the same source device.")

    tracker_indices = sample_hit_indices(
        num_hits=len(tracker_hits["x"]),
        max_hits=max_tracker_hits,
        device=source_device,
    )
    calo_indices = sample_hit_indices(
        num_hits=len(calo_hits["x"]),
        max_hits=max_calo_hits,
        device=source_device,
    )

    # Build a single coordinate tensor by concatenating tracker and calorimeter hits.
    tracker_coord = torch.stack(
        [
            _require_hit_tensor(tracker_hits, "x", source_device)[tracker_indices],
            _require_hit_tensor(tracker_hits, "y", source_device)[tracker_indices],
            _require_hit_tensor(tracker_hits, "z", source_device)[tracker_indices],
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)
    calo_coord = torch.stack(
        [
            _require_hit_tensor(calo_hits, "x", source_device)[calo_indices],
            _require_hit_tensor(calo_hits, "y", source_device)[calo_indices],
            _require_hit_tensor(calo_hits, "z", source_device)[calo_indices],
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)

    tracker_time = normalize_feature(
        torch.as_tensor(
            tracker_hits.get("time", torch.zeros(len(tracker_hits["x"]), device=source_device))[tracker_indices],
            dtype=torch.float32,
            device=device,
        )
    )
    calo_energy = normalize_feature(
        torch.as_tensor(
            calo_hits.get("total_energy", torch.zeros(len(calo_hits["x"]), device=source_device))[calo_indices],
            dtype=torch.float32,
            device=device,
        )
    )

    coord = torch.cat([tracker_coord, calo_coord], dim=0)
    signal = torch.cat([tracker_time, calo_energy], dim=0)
    # Encode detector origin as a simple binary feature: 0 for tracker, 1 for calo.
    detector_type = torch.cat(
        [
            torch.zeros(tracker_coord.shape[0], device=device),
            torch.ones(calo_coord.shape[0], device=device),
        ],
        dim=0,
    )

    return validate_point_view(
        {
            "coord": coord,
            "feat": assemble_point_features(coord, signal, detector_type),
            "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
            "grid_size": torch.tensor(grid_size, dtype=torch.float32, device=device),
        }
    )


def augment_point_view(
    view: Mapping[str, torch.Tensor],
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
) -> PointView:
    """Create a noisy copy of a point view for self-distillation training."""
    base_view = validate_point_view(view)
    coord = base_view["coord"].clone()
    signal = base_view["feat"][:, 4].clone()
    detector_type = base_view["feat"][:, 5].clone()

    # Rebuild derived features from the perturbed coordinates to keep the view consistent.
    coord = coord + torch.randn_like(coord) * coord_noise_scale
    signal = signal + torch.randn_like(signal) * feat_noise_scale

    return {
        "coord": coord,
        "feat": assemble_point_features(coord, signal, detector_type),
        "offset": base_view["offset"].clone(),
        "grid_size": base_view["grid_size"].clone(),
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
    # Offsets store cumulative point counts so the backbone can recover event boundaries.
    counts = torch.tensor([view["coord"].shape[0] for view in normalized_views], device=coord.device, dtype=torch.long)
    return {
        "coord": coord,
        "feat": feat,
        "offset": torch.cumsum(counts, dim=0),
        "grid_size": grid_size.clone(),
    }


def build_distillation_views(
    events: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_tracker_hits: int | None = None,
    max_calo_hits: int | None = None,
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
    num_augmentations: int = 2,
) -> list[PointView]:
    """Create independently augmented batched views from the same events."""
    if num_augmentations < 2:
        raise ValueError("Self-distillation requires at least two augmented views.")

    base_views = [
        build_point_view_from_event(
            event,
            device=device,
            max_tracker_hits=max_tracker_hits,
            max_calo_hits=max_calo_hits,
        )
        for event in events
    ]

    def build_augmented_batch() -> PointView:
        return batch_point_views(
            [
                augment_point_view(view, coord_noise_scale=coord_noise_scale, feat_noise_scale=feat_noise_scale)
                for view in base_views
            ]
        )

    return [build_augmented_batch() for _ in range(num_augmentations)]


def make_random_view(num_points: int, in_channels: int, device: torch.device) -> PointView:
    """Generate a synthetic point view for quick smoke tests."""
    if in_channels != POINT_FEATURE_DIM:
        raise ValueError(f"Random views in this project require {POINT_FEATURE_DIM} input channels.")

    coord = torch.rand(num_points, 3, device=device)
    signal = torch.rand(num_points, device=device)
    detector_type = torch.randint(0, 2, (num_points,), device=device, dtype=torch.int64).to(torch.float32)
    return {
        "coord": coord,
        "feat": assemble_point_features(coord, signal, detector_type),
        "offset": torch.tensor([num_points], dtype=torch.long, device=device),
        "grid_size": torch.tensor(DEFAULT_POINT_GRID_SIZE, device=device),
    }
