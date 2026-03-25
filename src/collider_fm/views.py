from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


DEFAULT_POINT_GRID_SIZE = 10.0
POINT_FEATURE_DIM = 2
DEFAULT_MASK_FRACTION = 0.3

# ColliderML calorimeter detector IDs split naturally into two coarse groups.
# For this phase we keep the detector story intentionally simple: 9/10/11 are
# treated as ECal and 12/13/14 are treated as HCal.
ECAL_DETECTOR_IDS = frozenset({9, 10, 11})
HCAL_DETECTOR_IDS = frozenset({12, 13, 14})
CALO_TYPE_NAMES = {0: "ecal", 1: "hcal"}
PointView = dict[str, torch.Tensor]


def _normalize_grid_size(grid_size: Any, device: torch.device) -> torch.Tensor:
    grid_size_tensor = torch.as_tensor(
        grid_size, dtype=torch.float32, device=device
    ).flatten()
    if grid_size_tensor.numel() != 1:
        raise ValueError("'grid_size' must be a scalar.")
    return grid_size_tensor[0]


def _require_hit_tensor(
    hits: Mapping[str, Any],
    key: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if key not in hits:
        raise KeyError(f"Missing required hit field '{key}'.")
    return torch.as_tensor(hits[key], dtype=dtype, device=device).flatten()


def _resolve_calo_energy(
    calo_hits: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    for key in ("energy", "totalenergy", "total_energy"):
        if key in calo_hits:
            return torch.as_tensor(
                calo_hits[key], dtype=torch.float32, device=device
            ).flatten()
    raise KeyError(
        "Calorimeter hits must provide 'energy', 'totalenergy', or 'total_energy'."
    )


def calo_type_from_detector_id(detector_id: torch.Tensor) -> torch.Tensor:
    detector_id = torch.as_tensor(detector_id, dtype=torch.long)
    known_ids = torch.tensor(
        sorted(ECAL_DETECTOR_IDS | HCAL_DETECTOR_IDS), device=detector_id.device
    )
    if not torch.isin(detector_id, known_ids).all():
        raise ValueError("Encountered an unknown calorimeter detector ID.")

    return (detector_id >= 12).to(torch.float32)


def assemble_point_features(
    energy: torch.Tensor, calo_type: torch.Tensor
) -> torch.Tensor:
    """Build the simple per-point feature tensor `[energy, is_hcal]`."""
    energy = torch.as_tensor(energy, dtype=torch.float32).flatten()
    calo_type = torch.as_tensor(
        calo_type, dtype=torch.float32, device=energy.device
    ).flatten()
    if energy.shape[0] != calo_type.shape[0]:
        raise ValueError("'energy' and 'calo_type' must contain one value per point.")
    return torch.stack([energy, calo_type], dim=1)


def validate_point_view(view: Mapping[str, Any]) -> PointView:
    """Normalize a point-view mapping used by the model and scripts."""
    if "coord" not in view or "feat" not in view:
        raise KeyError("Point views must provide 'coord' and 'feat'.")

    coord = torch.as_tensor(view["coord"], dtype=torch.float32)
    feat = torch.as_tensor(view["feat"], dtype=torch.float32, device=coord.device)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")
    if feat.ndim != 2 or feat.shape[1] != POINT_FEATURE_DIM:
        raise ValueError(f"'feat' must have shape [num_points, {POINT_FEATURE_DIM}].")
    if coord.shape[0] != feat.shape[0]:
        raise ValueError("'coord' and 'feat' must describe the same number of points.")

    offset = torch.as_tensor(
        view.get("offset", [coord.shape[0]]), dtype=torch.long, device=coord.device
    ).flatten()
    if offset.numel() == 0 or offset[-1].item() != coord.shape[0]:
        raise ValueError("'offset' must end at the number of points.")

    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    if torch.any(counts <= 0):
        raise ValueError("'offset' must be strictly increasing.")

    energy = torch.as_tensor(
        view.get("energy", feat[:, 0]), dtype=torch.float32, device=coord.device
    ).flatten()
    calo_type = torch.as_tensor(
        view.get("calo_type", feat[:, 1]), dtype=torch.float32, device=coord.device
    ).flatten()
    detector_id = torch.as_tensor(
        view.get("detector_id", torch.full((coord.shape[0],), -1, device=coord.device)),
        dtype=torch.long,
        device=coord.device,
    ).flatten()
    mask = torch.as_tensor(
        view.get(
            "mask", torch.zeros(coord.shape[0], dtype=torch.bool, device=coord.device)
        ),
        dtype=torch.bool,
        device=coord.device,
    ).flatten()

    if (
        energy.shape[0] != coord.shape[0]
        or calo_type.shape[0] != coord.shape[0]
        or detector_id.shape[0] != coord.shape[0]
        or mask.shape[0] != coord.shape[0]
    ):
        raise ValueError("Point-view side channels must contain one value per point.")

    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "grid_size": _normalize_grid_size(
            view.get("grid_size", DEFAULT_POINT_GRID_SIZE), coord.device
        ),
        "energy": energy,
        "detector_id": detector_id,
        "calo_type": calo_type,
        "mask": mask,
    }


def sample_hit_indices(
    num_hits: int, max_hits: int | None, device: torch.device
) -> torch.Tensor:
    if max_hits is None or num_hits <= max_hits:
        return torch.arange(num_hits, device=device)
    return torch.linspace(0, num_hits - 1, steps=max_hits, device=device).round().long()


def build_point_view_from_event(
    event: Mapping[str, Any],
    device: torch.device,
    max_calo_hits: int | None = None,
    grid_size: float = DEFAULT_POINT_GRID_SIZE,
) -> PointView:
    """Convert one raw calorimeter event into the point-view format used here."""
    calo_hits = event["calo_hits"]
    source_device = torch.as_tensor(calo_hits["x"]).device
    if len(calo_hits["x"]) == 0:
        raise ValueError(
            "Cannot build a point view from an event with zero calorimeter hits."
        )

    indices = sample_hit_indices(len(calo_hits["x"]), max_calo_hits, source_device)
    coord = torch.stack(
        [
            _require_hit_tensor(calo_hits, "x", source_device)[indices],
            _require_hit_tensor(calo_hits, "y", source_device)[indices],
            _require_hit_tensor(calo_hits, "z", source_device)[indices],
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)
    energy = _resolve_calo_energy(calo_hits, source_device)[indices].to(
        device=device, dtype=torch.float32
    )
    detector_id = _require_hit_tensor(
        calo_hits, "detector", source_device, dtype=torch.long
    )[indices].to(device=device, dtype=torch.long)
    calo_type = calo_type_from_detector_id(detector_id).to(
        device=device, dtype=torch.float32
    )

    return validate_point_view(
        {
            "coord": coord,
            "feat": assemble_point_features(energy, calo_type),
            "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
            "grid_size": torch.tensor(grid_size, dtype=torch.float32, device=device),
            "energy": energy,
            "detector_id": detector_id,
            "calo_type": calo_type,
            "mask": torch.zeros(coord.shape[0], dtype=torch.bool, device=device),
        }
    )


def rotate_around_z(coord: torch.Tensor, angle_radians: float) -> torch.Tensor:
    cos_angle = torch.cos(
        torch.tensor(angle_radians, dtype=coord.dtype, device=coord.device)
    )
    sin_angle = torch.sin(
        torch.tensor(angle_radians, dtype=coord.dtype, device=coord.device)
    )
    rotated_x = cos_angle * coord[:, 0] - sin_angle * coord[:, 1]
    rotated_y = sin_angle * coord[:, 0] + cos_angle * coord[:, 1]
    return torch.stack([rotated_x, rotated_y, coord[:, 2]], dim=1)


def augment_point_view(
    view: Mapping[str, Any],
    max_rotation_degrees: float = 180.0,
    coord_noise_scale: float = 1.0,
    energy_jitter_scale: float = 0.05,
) -> PointView:
    """Make a simple global calorimeter augmentation without changing point order."""
    base_view = validate_point_view(view)
    angle_radians = (
        float((torch.rand(1, device=base_view["coord"].device) * 2.0 - 1.0).item())
        * torch.pi
        * max_rotation_degrees
        / 180.0
    )
    coord = rotate_around_z(base_view["coord"], angle_radians)
    coord = coord + torch.randn_like(coord) * coord_noise_scale
    energy = (
        base_view["energy"]
        * (1.0 + torch.randn_like(base_view["energy"]) * energy_jitter_scale)
    ).clamp_min(0.0)

    return {
        "coord": coord,
        "feat": assemble_point_features(energy, base_view["calo_type"]),
        "offset": base_view["offset"].clone(),
        "grid_size": base_view["grid_size"].clone(),
        "energy": energy,
        "detector_id": base_view["detector_id"].clone(),
        "calo_type": base_view["calo_type"].clone(),
        "mask": base_view["mask"].clone(),
    }


def mask_point_view(
    view: Mapping[str, Any], mask_fraction: float = DEFAULT_MASK_FRACTION
) -> PointView:
    """Hide a random fraction of point energies while keeping point order fixed."""
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError("'mask_fraction' must be between 0 and 1.")

    base_view = validate_point_view(view)
    num_points = base_view["coord"].shape[0]
    num_masked = int(round(num_points * mask_fraction))
    if mask_fraction > 0.0:
        num_masked = max(1, num_masked)
    num_masked = min(num_points, num_masked)

    mask = torch.zeros(num_points, dtype=torch.bool, device=base_view["coord"].device)
    if num_masked > 0:
        selected_indices = torch.randperm(num_points, device=mask.device)[:num_masked]
        mask[selected_indices] = True

    energy = base_view["energy"].clone()
    energy[mask] = 0.0
    return {
        "coord": base_view["coord"].clone(),
        "feat": assemble_point_features(energy, base_view["calo_type"]),
        "offset": base_view["offset"].clone(),
        "grid_size": base_view["grid_size"].clone(),
        "energy": energy,
        "detector_id": base_view["detector_id"].clone(),
        "calo_type": base_view["calo_type"].clone(),
        "mask": mask,
    }


def batch_point_views(views: Sequence[Mapping[str, Any]]) -> PointView:
    if not views:
        raise ValueError("At least one point view is required to create a batch.")

    normalized_views = [validate_point_view(view) for view in views]
    grid_size = normalized_views[0]["grid_size"]
    for view in normalized_views[1:]:
        if not torch.allclose(view["grid_size"], grid_size):
            raise ValueError(
                "All point views in a batch must share the same grid size."
            )

    counts = torch.tensor(
        [view["coord"].shape[0] for view in normalized_views],
        dtype=torch.long,
        device=grid_size.device,
    )
    return {
        "coord": torch.cat([view["coord"] for view in normalized_views], dim=0),
        "feat": torch.cat([view["feat"] for view in normalized_views], dim=0),
        "offset": torch.cumsum(counts, dim=0),
        "grid_size": grid_size.clone(),
        "energy": torch.cat([view["energy"] for view in normalized_views], dim=0),
        "detector_id": torch.cat(
            [view["detector_id"] for view in normalized_views], dim=0
        ),
        "calo_type": torch.cat([view["calo_type"] for view in normalized_views], dim=0),
        "mask": torch.cat([view["mask"] for view in normalized_views], dim=0),
    }


def build_distillation_views(
    events: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_calo_hits: int | None = None,
    num_augmentations: int = 2,
    add_masked_view: bool = False,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> list[PointView]:
    if num_augmentations < 2:
        raise ValueError("Self-distillation requires at least two augmented views.")

    base_views = [
        build_point_view_from_event(event, device=device, max_calo_hits=max_calo_hits)
        for event in events
    ]
    views = [
        batch_point_views([augment_point_view(view) for view in base_views])
        for _ in range(num_augmentations)
    ]
    if add_masked_view:
        views.append(
            batch_point_views(
                [
                    mask_point_view(
                        augment_point_view(view), mask_fraction=mask_fraction
                    )
                    for view in base_views
                ]
            )
        )
    return views


def make_random_view(
    num_points: int, in_channels: int, device: torch.device
) -> PointView:
    if in_channels != POINT_FEATURE_DIM:
        raise ValueError(
            f"Random views in this project require {POINT_FEATURE_DIM} input channels."
        )

    coord = torch.randn(num_points, 3, device=device) * 1000.0
    energy = torch.rand(num_points, device=device)
    detector_id = torch.where(
        torch.rand(num_points, device=device) > 0.5,
        torch.tensor(13, device=device),
        torch.tensor(10, device=device),
    )
    calo_type = calo_type_from_detector_id(detector_id).to(torch.float32)
    return {
        "coord": coord,
        "feat": assemble_point_features(energy, calo_type),
        "offset": torch.tensor([num_points], dtype=torch.long, device=device),
        "grid_size": torch.tensor(
            DEFAULT_POINT_GRID_SIZE, dtype=torch.float32, device=device
        ),
        "energy": energy,
        "detector_id": detector_id.long(),
        "calo_type": calo_type,
        "mask": torch.zeros(num_points, dtype=torch.bool, device=device),
    }
