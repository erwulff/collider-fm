from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def normalize_feature(values: torch.Tensor) -> torch.Tensor:
    """Scale a 1D feature to a stable range using its largest absolute value."""
    scale = values.abs().max().clamp_min(1.0)
    return values / scale


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
    grid_size: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Convert one raw event into the point-view format expected by the model.

    The returned mapping matches the Panda point-cloud convention used elsewhere in
    the repo: coordinates, per-point features, cumulative offsets, and grid size.
    """
    tracker_hits = event["tracker_hits"]
    calo_hits = event["calo_hits"]
    source_device = tracker_hits["x"].device
    if calo_hits["x"].device != source_device:
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
        [tracker_hits["x"][tracker_indices], tracker_hits["y"][tracker_indices], tracker_hits["z"][tracker_indices]],
        dim=1,
    ).to(device=device, dtype=torch.float32)
    calo_coord = torch.stack(
        [calo_hits["x"][calo_indices], calo_hits["y"][calo_indices], calo_hits["z"][calo_indices]],
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
    radius = normalize_feature(torch.linalg.norm(coord, dim=1))
    signal = torch.cat([tracker_time, calo_energy], dim=0)
    # Encode detector origin as a simple binary feature: 0 for tracker, 1 for calo.
    detector_type = torch.cat(
        [
            torch.zeros(tracker_coord.shape[0], device=device),
            torch.ones(calo_coord.shape[0], device=device),
        ],
        dim=0,
    )

    feat = torch.cat(
        [
            coord,
            radius.unsqueeze(1),
            signal.unsqueeze(1),
            detector_type.unsqueeze(1),
        ],
        dim=1,
    )
    return {
        "coord": coord,
        "feat": feat,
        "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
        "grid_size": torch.tensor(grid_size, dtype=torch.float32, device=device),
    }


def augment_point_view(
    view: Mapping[str, torch.Tensor],
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Create a noisy copy of a point view for self-distillation training."""
    coord = view["coord"].clone()
    feat = view["feat"].clone()

    # Perturb coordinates directly, then keep the coordinate feature channels in sync.
    coord = coord + torch.randn_like(coord) * coord_noise_scale
    feat[:, :3] = coord
    feat[:, 3:5] = feat[:, 3:5] + torch.randn_like(feat[:, 3:5]) * feat_noise_scale

    return {
        "coord": coord,
        "feat": feat,
        "offset": view["offset"].clone(),
        "grid_size": view["grid_size"].clone(),
    }


def batch_point_views(views: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate per-event point views into one batched point-cloud mapping."""
    if not views:
        raise ValueError("At least one point view is required to create a batch.")

    coord = torch.cat([view["coord"] for view in views], dim=0)
    feat = torch.cat([view["feat"] for view in views], dim=0)
    # Offsets store cumulative point counts so the backbone can recover event boundaries.
    counts = torch.tensor([view["coord"].shape[0] for view in views], device=coord.device, dtype=torch.long)
    offset = torch.cumsum(counts, dim=0)
    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "grid_size": views[0]["grid_size"].clone(),
    }


def build_distillation_views(
    events: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_tracker_hits: int | None = None,
    max_calo_hits: int | None = None,
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
) -> list[dict[str, torch.Tensor]]:
    """Create two independently augmented batched views from the same events."""
    base_views = [
        build_point_view_from_event(
            event,
            device=device,
            max_tracker_hits=max_tracker_hits,
            max_calo_hits=max_calo_hits,
        )
        for event in events
    ]
    return [
        # Student and teacher both see the same events through different noise realizations.
        batch_point_views([augment_point_view(view, coord_noise_scale=coord_noise_scale, feat_noise_scale=feat_noise_scale) for view in base_views]),
        batch_point_views([augment_point_view(view, coord_noise_scale=coord_noise_scale, feat_noise_scale=feat_noise_scale) for view in base_views]),
    ]


def make_random_view(num_points: int, in_channels: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate a synthetic point view for quick smoke tests."""
    return {
        "coord": torch.rand(num_points, 3, device=device),
        "feat": torch.rand(num_points, in_channels, device=device),
        "offset": torch.tensor([num_points], dtype=torch.long, device=device),
        "grid_size": torch.tensor(0.05, device=device),
    }
