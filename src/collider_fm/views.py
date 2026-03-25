from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch

from .features import MultimodalPointBatch, build_multimodal_points


DEFAULT_POINT_GRID_SIZE = 10.0
POINT_FEATURE_DIM = 6


class PointView(TypedDict):
    coord: torch.Tensor
    feat: torch.Tensor
    offset: torch.Tensor
    grid_size: torch.Tensor


class SSLView(TypedDict):
    coord: torch.Tensor
    tracker_continuous: torch.Tensor
    calo_continuous: torch.Tensor
    tracker_categorical: dict[str, torch.Tensor]
    calo_categorical: dict[str, torch.Tensor]
    modality_id: torch.Tensor
    point_id: torch.Tensor
    event_id: torch.Tensor
    offset: torch.Tensor
    grid_size: torch.Tensor
    tracker_index: torch.Tensor
    calo_index: torch.Tensor
    view_type: str
    visible_point_mask: torch.Tensor


class SSLViewSet(TypedDict):
    teacher_global: list[SSLView]
    student_global: list[SSLView]
    student_local: list[SSLView]
    student_masked: list[SSLView]


SSL_VIEW_ORDER = (
    "teacher_global",
    "student_global",
    "student_local",
    "student_masked",
)


@dataclass(frozen=True)
class SSLViewConfig:
    teacher_global_views: int = 1
    student_global_views: int = 1
    student_local_views: int = 2
    student_masked_views: int = 1
    global_fraction_min: float = 0.6
    global_fraction_max: float = 1.0
    local_fraction_min: float = 0.2
    local_fraction_max: float = 0.5
    mask_fraction: float = 0.3
    phi_rotation_max: float = 0.2
    coord_jitter_scale: float = 0.02
    tracker_time_jitter_scale: float = 0.01
    calo_energy_jitter_scale: float = 0.01


def _require_sorted_indices(indices: torch.Tensor) -> torch.Tensor:
    indices = torch.as_tensor(indices, dtype=torch.long).flatten()
    if indices.numel() == 0:
        raise ValueError("Each SSL view must keep at least one point.")
    return torch.unique(indices, sorted=True)


def _sample_fraction(low: float, high: float) -> float:
    if not 0.0 < low <= high <= 1.0:
        raise ValueError("View fractions must satisfy 0 < low <= high <= 1.")
    if low == high:
        return low
    return float(torch.empty(1).uniform_(low, high).item())


def _balanced_sample_indices(modality_id: torch.Tensor, keep_count: int) -> torch.Tensor:
    modality_id = torch.as_tensor(modality_id, dtype=torch.long).flatten()
    num_points = modality_id.shape[0]
    if keep_count >= num_points:
        return torch.arange(num_points, device=modality_id.device)
    if keep_count <= 0:
        raise ValueError("keep_count must be positive.")

    selected: list[torch.Tensor] = []
    unique_modalities = torch.unique(modality_id)
    if keep_count >= unique_modalities.numel():
        for modality in unique_modalities.tolist():
            modality_indices = torch.nonzero(modality_id == modality, as_tuple=False).flatten()
            anchor = modality_indices[torch.randint(modality_indices.numel(), (1,), device=modality_id.device)]
            selected.append(anchor)

    already = torch.cat(selected, dim=0) if selected else modality_id.new_empty((0,), dtype=torch.long)
    remaining_budget = keep_count - already.numel()
    if remaining_budget > 0:
        all_indices = torch.randperm(num_points, device=modality_id.device)
        if already.numel() > 0:
            keep_mask = torch.ones(num_points, dtype=torch.bool, device=modality_id.device)
            keep_mask[already] = False
            all_indices = all_indices[keep_mask[all_indices]]
        selected.append(all_indices[:remaining_budget])

    return _require_sorted_indices(torch.cat(selected, dim=0))


def _subset_multimodal_points(
    points: MultimodalPointBatch,
    selected_indices: torch.Tensor,
    *,
    view_type: str,
    visible_point_mask: torch.Tensor | None = None,
    event_id: int = 0,
) -> SSLView:
    selected_indices = _require_sorted_indices(selected_indices.to(points.coord.device))
    num_tracker = points.tracker_continuous.shape[0]
    tracker_source_indices = selected_indices[selected_indices < num_tracker]
    calo_source_indices = selected_indices[selected_indices >= num_tracker] - num_tracker

    coord = points.coord[selected_indices]
    tracker_continuous = points.tracker_continuous[tracker_source_indices]
    calo_continuous = points.calo_continuous[calo_source_indices]

    tracker_categorical = {
        key: value[tracker_source_indices]
        for key, value in points.tracker_categorical.items()
    }
    calo_categorical = {
        key: value[calo_source_indices]
        for key, value in points.calo_categorical.items()
    }

    num_selected_tracker = tracker_source_indices.numel()
    num_selected_calo = calo_source_indices.numel()
    fused_count = num_selected_tracker + num_selected_calo
    modality_id = torch.cat(
        [
            torch.zeros(num_selected_tracker, dtype=torch.long, device=coord.device),
            torch.ones(num_selected_calo, dtype=torch.long, device=coord.device),
        ],
        dim=0,
    )
    event_id_tensor = torch.full((fused_count,), event_id, dtype=torch.long, device=coord.device)
    if visible_point_mask is None:
        visible_point_mask = torch.zeros(points.coord.shape[0], dtype=torch.bool, device=coord.device)
        visible_point_mask[selected_indices] = True

    return {
        "coord": coord,
        "tracker_continuous": tracker_continuous,
        "calo_continuous": calo_continuous,
        "tracker_categorical": tracker_categorical,
        "calo_categorical": calo_categorical,
        "modality_id": modality_id,
        "point_id": points.point_id[selected_indices],
        "event_id": event_id_tensor,
        "offset": torch.tensor([fused_count], dtype=torch.long, device=coord.device),
        "grid_size": points.grid_size.clone(),
        "tracker_index": torch.arange(num_selected_tracker, dtype=torch.long, device=coord.device),
        "calo_index": torch.arange(num_selected_calo, dtype=torch.long, device=coord.device) + num_selected_tracker,
        "view_type": view_type,
        "visible_point_mask": visible_point_mask,
    }


def _compute_cylindrical(coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]
    radius = torch.sqrt(x.square() + y.square())
    phi = torch.atan2(y, x)
    return radius, phi, z


def _apply_coord_jitter(coord: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return coord
    return coord + torch.randn_like(coord) * scale


def apply_phi_rotation(view: SSLView, angle: float) -> SSLView:
    if angle == 0.0:
        return view
    cos_angle = torch.cos(torch.tensor(angle, device=view["coord"].device, dtype=view["coord"].dtype))
    sin_angle = torch.sin(torch.tensor(angle, device=view["coord"].device, dtype=view["coord"].dtype))

    def rotate_xyz(values: torch.Tensor) -> torch.Tensor:
        rotated = values.clone()
        x = values[:, 0]
        y = values[:, 1]
        rotated[:, 0] = cos_angle * x - sin_angle * y
        rotated[:, 1] = sin_angle * x + cos_angle * y
        return rotated

    rotated_view = dict(view)
    rotated_view["coord"] = rotate_xyz(view["coord"])
    rotated_view["tracker_continuous"] = rotate_xyz(view["tracker_continuous"])
    rotated_view["calo_continuous"] = rotate_xyz(view["calo_continuous"])
    return rotated_view  # type: ignore[return-value]


def apply_tracker_time_jitter(view: SSLView, scale: float) -> SSLView:
    if scale <= 0.0 or view["tracker_continuous"].numel() == 0:
        return view
    updated = dict(view)
    tracker_continuous = view["tracker_continuous"].clone()
    tracker_continuous[:, 3] = tracker_continuous[:, 3] + torch.randn_like(tracker_continuous[:, 3]) * scale
    updated["tracker_continuous"] = tracker_continuous
    return updated  # type: ignore[return-value]


def apply_calo_energy_jitter(view: SSLView, scale: float) -> SSLView:
    if scale <= 0.0 or view["calo_continuous"].numel() == 0:
        return view
    updated = dict(view)
    calo_continuous = view["calo_continuous"].clone()
    calo_continuous[:, 3] = (calo_continuous[:, 3] + torch.randn_like(calo_continuous[:, 3]) * scale).clamp_min(0.0)
    updated["calo_continuous"] = calo_continuous
    return updated  # type: ignore[return-value]


def _apply_shared_view_augmentations(view: SSLView, config: SSLViewConfig) -> SSLView:
    angle = float(torch.empty(1).uniform_(-config.phi_rotation_max, config.phi_rotation_max).item())
    updated = apply_phi_rotation(view, angle=angle)
    augmented = dict(updated)
    augmented["coord"] = _apply_coord_jitter(updated["coord"], config.coord_jitter_scale)
    augmented["tracker_continuous"] = updated["tracker_continuous"].clone()
    augmented["calo_continuous"] = updated["calo_continuous"].clone()
    augmented = apply_tracker_time_jitter(augmented, config.tracker_time_jitter_scale)
    augmented = apply_calo_energy_jitter(augmented, config.calo_energy_jitter_scale)
    return augmented


def build_global_view(
    points: MultimodalPointBatch,
    config: SSLViewConfig,
    *,
    view_type: str,
) -> SSLView:
    keep_fraction = _sample_fraction(config.global_fraction_min, config.global_fraction_max)
    keep_count = max(1, int(round(points.coord.shape[0] * keep_fraction)))
    selected_indices = _balanced_sample_indices(points.modality_id, keep_count)
    view = _subset_multimodal_points(points, selected_indices, view_type=view_type)
    return _apply_shared_view_augmentations(view, config)


def build_local_view(
    points: MultimodalPointBatch,
    config: SSLViewConfig,
    *,
    view_type: str = "student_local",
) -> SSLView:
    keep_fraction = _sample_fraction(config.local_fraction_min, config.local_fraction_max)
    keep_count = max(1, int(round(points.coord.shape[0] * keep_fraction)))
    anchor_index = int(torch.randint(points.coord.shape[0], (1,), device=points.coord.device).item())
    radius, phi, z = _compute_cylindrical(points.coord)
    anchor_radius = radius[anchor_index]
    anchor_phi = phi[anchor_index]
    anchor_z = z[anchor_index]

    radius_scale = radius.std().clamp_min(1.0)
    z_scale = z.std().clamp_min(1.0)
    phi_delta = torch.atan2(torch.sin(phi - anchor_phi), torch.cos(phi - anchor_phi))
    distance = ((radius - anchor_radius) / radius_scale).square() + (phi_delta / torch.pi).square() + ((z - anchor_z) / z_scale).square()
    selected_indices = torch.argsort(distance)[:keep_count]
    view = _subset_multimodal_points(points, selected_indices, view_type=view_type)
    return _apply_shared_view_augmentations(view, config)


def build_masked_view(
    points: MultimodalPointBatch,
    config: SSLViewConfig,
    *,
    view_type: str = "student_masked",
) -> SSLView:
    base_view = build_global_view(points, config, view_type=view_type)
    base_point_id = base_view["point_id"]
    base_modality = base_view["modality_id"]
    if base_point_id.numel() <= 1:
        return base_view

    keep_mask = torch.ones(base_point_id.numel(), dtype=torch.bool, device=base_point_id.device)
    for modality in torch.unique(base_modality).tolist():
        modality_indices = torch.nonzero(base_modality == modality, as_tuple=False).flatten()
        if modality_indices.numel() <= 1:
            continue
        mask_count = min(modality_indices.numel() - 1, int(round(modality_indices.numel() * config.mask_fraction)))
        if mask_count <= 0:
            continue
        mask_order = torch.randperm(modality_indices.numel(), device=base_point_id.device)[:mask_count]
        keep_mask[modality_indices[mask_order]] = False

    selected_point_ids = base_point_id[keep_mask]
    selected_indices = torch.nonzero(torch.isin(points.point_id, selected_point_ids), as_tuple=False).flatten()
    visible_point_mask = torch.zeros(points.coord.shape[0], dtype=torch.bool, device=points.coord.device)
    visible_point_mask[selected_indices] = True
    view = _subset_multimodal_points(
        points,
        selected_indices,
        view_type=view_type,
        visible_point_mask=visible_point_mask,
    )
    return _apply_shared_view_augmentations(view, config)


def batch_ssl_views(views: Sequence[SSLView]) -> SSLView:
    if not views:
        raise ValueError("At least one SSL view is required to create a batch.")

    coord_device = views[0]["coord"].device
    grid_size = views[0]["grid_size"]
    batched_coord = []
    batched_tracker_continuous = []
    batched_calo_continuous = []
    batched_tracker_categorical: dict[str, list[torch.Tensor]] = {}
    batched_calo_categorical: dict[str, list[torch.Tensor]] = {}
    batched_modality_id = []
    batched_point_id = []
    batched_event_id = []
    batched_visible_mask = []
    batched_tracker_index = []
    batched_calo_index = []
    counts = []
    point_cursor = 0

    for event_index, view in enumerate(views):
        if not torch.allclose(view["grid_size"], grid_size):
            raise ValueError("All SSL views in a batch must share the same grid size.")

        num_points = view["coord"].shape[0]
        counts.append(num_points)
        batched_coord.append(view["coord"])
        batched_tracker_continuous.append(view["tracker_continuous"])
        batched_calo_continuous.append(view["calo_continuous"])
        batched_modality_id.append(view["modality_id"])
        batched_point_id.append(view["point_id"])
        batched_event_id.append(torch.full((num_points,), event_index, dtype=torch.long, device=coord_device))
        batched_visible_mask.append(view["visible_point_mask"])
        batched_tracker_index.append(view["tracker_index"] + point_cursor)
        batched_calo_index.append(view["calo_index"] + point_cursor)

        for key, value in view["tracker_categorical"].items():
            batched_tracker_categorical.setdefault(key, []).append(value)
        for key, value in view["calo_categorical"].items():
            batched_calo_categorical.setdefault(key, []).append(value)

        point_cursor += num_points

    return {
        "coord": torch.cat(batched_coord, dim=0),
        "tracker_continuous": torch.cat(batched_tracker_continuous, dim=0),
        "calo_continuous": torch.cat(batched_calo_continuous, dim=0),
        "tracker_categorical": {key: torch.cat(values, dim=0) for key, values in batched_tracker_categorical.items()},
        "calo_categorical": {key: torch.cat(values, dim=0) for key, values in batched_calo_categorical.items()},
        "modality_id": torch.cat(batched_modality_id, dim=0),
        "point_id": torch.cat(batched_point_id, dim=0),
        "event_id": torch.cat(batched_event_id, dim=0),
        "offset": torch.cumsum(torch.tensor(counts, dtype=torch.long, device=coord_device), dim=0),
        "grid_size": grid_size.clone(),
        "tracker_index": torch.cat(batched_tracker_index, dim=0),
        "calo_index": torch.cat(batched_calo_index, dim=0),
        "view_type": views[0]["view_type"],
        "visible_point_mask": torch.cat(batched_visible_mask, dim=0),
    }


def build_ssl_views(
    events: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_tracker_hits: int | None = None,
    max_calo_hits: int | None = None,
    config: SSLViewConfig | None = None,
) -> SSLViewSet:
    """Create structured multimodal SSL views over a batch of events.

    This new path is additive: it sits alongside the legacy six-channel point-view
    helpers so the repo can migrate incrementally.
    """
    resolved_config = config or SSLViewConfig()
    base_points = [
        build_multimodal_points(
            event,
            device=device,
            max_tracker_hits=max_tracker_hits,
            max_calo_hits=max_calo_hits,
        )
        for event in events
    ]

    def build_batched(kind: str) -> SSLView:
        builders = {
            "teacher_global": lambda points: build_global_view(points, resolved_config, view_type="teacher_global"),
            "student_global": lambda points: build_global_view(points, resolved_config, view_type="student_global"),
            "student_local": lambda points: build_local_view(points, resolved_config, view_type="student_local"),
            "student_masked": lambda points: build_masked_view(points, resolved_config, view_type="student_masked"),
        }
        return batch_ssl_views([builders[kind](points) for points in base_points])

    return {
        "teacher_global": [build_batched("teacher_global") for _ in range(resolved_config.teacher_global_views)],
        "student_global": [build_batched("student_global") for _ in range(resolved_config.student_global_views)],
        "student_local": [build_batched("student_local") for _ in range(resolved_config.student_local_views)],
        "student_masked": [build_batched("student_masked") for _ in range(resolved_config.student_masked_views)],
    }


def flatten_ssl_view_set(view_set: SSLViewSet) -> list[SSLView]:
    flattened: list[SSLView] = []
    for view_type in SSL_VIEW_ORDER:
        flattened.extend(view_set[view_type])
    return flattened


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
