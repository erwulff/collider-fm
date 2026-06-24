from __future__ import annotations

"""Point-view utilities for the calo-only self-distillation pipeline.

The project keeps the runtime point contract intentionally small and explicit:
each point carries `[x, y, z, total_energy]`, plus a few bookkeeping tensors that
make masking, batching, and teacher/student alignment easy to follow.
"""

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

import numpy as np
import torch

DEFAULT_POINT_GRID_SIZE = 10.0
POINT_FEATURE_DIM = 4


class PointView(TypedDict):
    coord: torch.Tensor
    origin_coord: torch.Tensor
    feat: torch.Tensor
    offset: torch.Tensor
    grid_size: torch.Tensor
    source_index: torch.Tensor
    total_energy: torch.Tensor
    patch_id: torch.Tensor
    mask: torch.Tensor
    view_kind: str


class SonataBatch(TypedDict):
    """Packed global/local views consumed by the Sonata training path."""

    global_coord: torch.Tensor
    global_origin_coord: torch.Tensor
    global_feat: torch.Tensor
    global_offset: torch.Tensor
    local_coord: torch.Tensor
    local_origin_coord: torch.Tensor
    local_feat: torch.Tensor
    local_offset: torch.Tensor
    grid_size: torch.Tensor


def _default_source_index(num_points: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_points, device=device, dtype=torch.long)


def _default_patch_id(coord: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
    grid_coord = torch.div(
        coord - coord.min(dim=0).values, grid_size, rounding_mode="floor"
    ).long()
    base = grid_coord.max(dim=0).values + 1
    stride_y = base[2].clamp_min(1)
    stride_x = (base[1] * stride_y).clamp_min(1)
    return grid_coord[:, 0] * stride_x + grid_coord[:, 1] * stride_y + grid_coord[:, 2]


def _fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.uint64, copy=True)
    hashed = np.full(arr.shape[0], np.uint64(14695981039346656037))
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])
    return hashed


def grid_sample(coord: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Return indices of one representative point per voxel (train-mode random pick).

    Points are quantized onto a regular grid with spacing *grid_size*.  When
    multiple points fall into the same voxel, one is selected at random.  The
    returned index tensor can be used to index into coord and any per-point
    tensors (energy, source_index, etc.).  This uses a small NumPy/CPU path to
    match the reference-style voxel hashing code and only returns the chosen
    indices to the original device.
    """
    if coord.numel() == 0:
        return torch.arange(0, device=coord.device, dtype=torch.long)

    scaled = coord.cpu().numpy() / np.array(grid_size)
    grid_coord = np.floor(scaled).astype(np.int64)
    grid_coord -= grid_coord.min(axis=0)

    key = _fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[:-1])
        + np.random.randint(0, int(count.max()), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    return torch.from_numpy(idx_unique).to(device=coord.device, dtype=torch.long)


def assemble_point_features(
    coord: torch.Tensor, total_energy: torch.Tensor
) -> torch.Tensor:
    """Build the simple project feature contract `[x, y, z, total_energy]`."""
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")

    total_energy = torch.as_tensor(
        total_energy, dtype=torch.float32, device=coord.device
    ).flatten()
    if total_energy.shape[0] != coord.shape[0]:
        raise ValueError("'total_energy' must contain one value per point.")

    return torch.cat([coord, total_energy.unsqueeze(1)], dim=1)


def transform_total_energy(
    total_energy: torch.Tensor,
    *,
    transform: str = "raw",
    min_val: float = 1.0e-2,
    max_val: float = 20.0,
) -> torch.Tensor:
    """Transform raw hit energy into the feature space used by the model."""

    energy = torch.as_tensor(total_energy, dtype=torch.float32).flatten()
    if transform == "raw":
        return energy
    if transform != "log":
        raise ValueError(f"Unsupported energy transform: {transform}.")

    y0 = torch.log10(torch.tensor(min_val, dtype=energy.dtype, device=energy.device))
    y1 = torch.log10(
        torch.tensor(max_val + min_val, dtype=energy.dtype, device=energy.device)
    )
    transformed = 2 * (torch.log10(energy + min_val) - y0) / (y1 - y0) - 1
    return transformed


def normalize_coord(
    coord: torch.Tensor,
    *,
    center: Sequence[float] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Optionally shift and scale coordinates before view construction."""

    normalized = torch.as_tensor(coord, dtype=torch.float32).clone()
    if center is None:
        center_tensor = normalized.new_zeros(3)
    else:
        center_tensor = torch.as_tensor(
            center, dtype=normalized.dtype, device=normalized.device
        )
        if center_tensor.shape != (3,):
            raise ValueError("'center' must contain exactly three coordinates.")
    normalized = normalized - center_tensor
    if scale is not None:
        if scale <= 0:
            raise ValueError("'scale' must be positive when provided.")
        normalized = normalized / float(scale)
    return normalized


def validate_point_view(view: Mapping[str, Any]) -> PointView:
    """Normalize one point-view mapping to the shared project contract."""
    required_keys = {"coord", "feat"}
    missing_keys = required_keys.difference(view.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Point view is missing required keys: {missing}.")

    coord = torch.as_tensor(view["coord"], dtype=torch.float32)
    origin_coord = torch.as_tensor(
        view.get("origin_coord", coord), dtype=torch.float32, device=coord.device
    )
    feat = torch.as_tensor(view["feat"], dtype=torch.float32, device=coord.device)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")
    if origin_coord.ndim != 2 or origin_coord.shape[1] != 3:
        raise ValueError("'origin_coord' must have shape [num_points, 3].")
    if feat.ndim != 2 or feat.shape[1] != POINT_FEATURE_DIM:
        raise ValueError(f"'feat' must have shape [num_points, {POINT_FEATURE_DIM}].")
    if feat.shape[0] != coord.shape[0]:
        raise ValueError("'coord' and 'feat' must have the same number of points.")
    if origin_coord.shape[0] != coord.shape[0]:
        raise ValueError(
            "'origin_coord' and 'coord' must have the same number of points."
        )

    offset_value = view.get("offset", [coord.shape[0]])
    offset = torch.as_tensor(
        offset_value, dtype=torch.long, device=coord.device
    ).flatten()
    if offset.numel() == 0:
        raise ValueError("'offset' must contain at least one event boundary.")
    if offset[-1].item() != coord.shape[0]:
        raise ValueError("The final offset must equal the number of points.")

    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    if torch.any(counts <= 0):
        raise ValueError("'offset' must be a strictly increasing cumulative count.")

    grid_size = torch.as_tensor(
        view.get("grid_size", DEFAULT_POINT_GRID_SIZE),
        dtype=torch.float32,
        device=coord.device,
    ).flatten()
    if grid_size.numel() != 1:
        raise ValueError("'grid_size' must be a scalar.")
    grid_size = grid_size[0]
    total_energy = torch.as_tensor(
        view.get("total_energy", feat[:, 3]), dtype=torch.float32, device=coord.device
    ).flatten()
    if total_energy.shape[0] != coord.shape[0]:
        raise ValueError("'total_energy' must contain one value per point.")

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
        "origin_coord": origin_coord,
        "feat": feat,
        "offset": offset,
        "grid_size": grid_size,
        "source_index": source_index,
        "total_energy": total_energy,
        "patch_id": patch_id,
        "mask": mask,
        "view_kind": str(view.get("view_kind", "unknown")),
    }


def sample_hit_indices(
    num_hits: int, max_hits: int | None, device: torch.device
) -> torch.Tensor:
    """Return evenly spaced hit indices, optionally downsampling to ``max_hits``."""
    if max_hits is None or num_hits <= max_hits:
        return torch.arange(num_hits, device=device)
    return torch.linspace(0, num_hits - 1, steps=max_hits, device=device).round().long()


def build_point_view_from_event(
    event: Mapping[str, Any],
    device: torch.device,
    max_calo_hits: int | None = None,
    grid_size: float = DEFAULT_POINT_GRID_SIZE,
    coord_center: Sequence[float] | None = None,
    coord_scale: float | None = None,
    energy_transform: str = "raw",
    energy_min: float = 1.0e-2,
    energy_max: float = 20.0,
    grid_sample_enabled: bool = True,
    grid_sample_size: float = 0.002,
) -> PointView:
    """Convert one raw ColliderML event into a single point view.

    The returned mapping is the canonical representation used throughout the repo:
    point coordinates in `coord`, model features in `feat`, cumulative event
    boundaries in `offset`, and bookkeeping tensors for matching masked student
    points back to the original calorimeter hits.
    """
    calo_hits = event["calo_hits"]
    coord_x = cast(torch.Tensor, calo_hits["x"])
    coord_y = cast(torch.Tensor, calo_hits["y"])
    coord_z = cast(torch.Tensor, calo_hits["z"])
    total_energy = cast(torch.Tensor, calo_hits["total_energy"])
    source_device = coord_x.device
    if coord_x.numel() == 0:
        raise ValueError(
            "Cannot build a point view from an event with zero calorimeter hits."
        )

    calo_indices = sample_hit_indices(
        num_hits=coord_x.shape[0],
        max_hits=max_calo_hits,
        device=source_device,
    )
    raw_coord = torch.stack(
        [
            coord_x[calo_indices],
            coord_y[calo_indices],
            coord_z[calo_indices],
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)
    coord = normalize_coord(raw_coord, center=coord_center, scale=coord_scale)
    raw_total_energy = total_energy[calo_indices].to(device=device, dtype=torch.float32)
    total_energy = transform_total_energy(
        raw_total_energy,
        transform=energy_transform,
        min_val=energy_min,
        max_val=energy_max,
    )
    source_index = torch.as_tensor(calo_indices, dtype=torch.long, device=device)
    grid_size_tensor = torch.tensor(grid_size, dtype=torch.float32, device=device)

    if grid_sample_enabled:
        gs_indices = grid_sample(coord, grid_size=grid_sample_size)
        coord = coord[gs_indices]
        total_energy = total_energy[gs_indices]
        # Keep the original hit indices after subsampling so masking/matching can
        # still refer back to the underlying event layout.
        source_index = source_index[gs_indices]

    return validate_point_view(
        {
            "coord": coord,
            "origin_coord": coord.clone(),
            "feat": assemble_point_features(coord, total_energy),
            "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
            "grid_size": grid_size_tensor,
            "source_index": source_index,
            "total_energy": total_energy,
            "patch_id": _default_patch_id(coord, grid_size_tensor),
            "mask": torch.zeros(coord.shape[0], dtype=torch.bool, device=device),
            "view_kind": "base",
        }
    )


def rotate_around_beam_axis(coord: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate x/y coordinates while leaving z unchanged."""
    cosine = torch.cos(torch.tensor(angle, dtype=coord.dtype, device=coord.device))
    sine = torch.sin(torch.tensor(angle, dtype=coord.dtype, device=coord.device))
    rotated = coord.clone()
    x = coord[:, 0]
    y = coord[:, 1]
    rotated[:, 0] = cosine * x - sine * y
    rotated[:, 1] = sine * x + cosine * y
    return rotated


def crop_point_view(
    view: Mapping[str, Any],
    keep_ratio: float,
    center_coord: torch.Tensor | None = None,
) -> PointView:
    """Keep the nearest points to a center to create a contiguous crop.

    When *center_coord* is ``None`` the crop center is drawn uniformly at random
    from the view's own points (Sonata default). When provided, the crop is
    centered on that coordinate instead, e.g. to constrain a secondary view to
    lie within the footprint of a principal view.
    """
    base_view = validate_point_view(view)
    num_points = base_view["coord"].shape[0]
    keep_count = max(1, min(num_points, int(round(num_points * keep_ratio))))
    if keep_count >= num_points:
        return base_view

    if center_coord is None:
        center_index = int(
            torch.randint(0, num_points, (1,), device=base_view["coord"].device).item()
        )
        center = base_view["coord"][center_index]
    else:
        center = torch.as_tensor(
            center_coord,
            dtype=base_view["coord"].dtype,
            device=base_view["coord"].device,
        )
    distances = torch.sum((base_view["coord"] - center) ** 2, dim=1)
    keep_indices = torch.argsort(distances)[:keep_count]
    keep_indices, _ = torch.sort(keep_indices)

    return validate_point_view(
        {
            "coord": cast(torch.Tensor, base_view["coord"])[keep_indices],
            "origin_coord": cast(torch.Tensor, base_view["origin_coord"])[keep_indices],
            "feat": cast(torch.Tensor, base_view["feat"])[keep_indices],
            "offset": torch.tensor(
                [keep_count], dtype=torch.long, device=base_view["coord"].device
            ),
            "grid_size": base_view["grid_size"],
            "source_index": cast(torch.Tensor, base_view["source_index"])[keep_indices],
            "total_energy": cast(torch.Tensor, base_view["total_energy"])[keep_indices],
            "patch_id": cast(torch.Tensor, base_view["patch_id"])[keep_indices],
            "mask": cast(torch.Tensor, base_view["mask"])[keep_indices],
            "view_kind": base_view["view_kind"],
        }
    )


def augment_point_view(
    view: Mapping[str, Any],
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
    phi_rotation_max: float = 0.25,
    point_dropout: float = 0.0,
    crop_keep_ratio: float = 1.0,
    view_kind: str = "augmented",
    center_coord: torch.Tensor | None = None,
) -> PointView:
    """Create a simple collider-safe augmentation of one point view.

    The transform intentionally stays small and readable:
    - optional contiguous crop
    - small rotation around the beam axis
    - coordinate jitter
    - multiplicative energy jitter
    - random point dropout

    Masking is handled by the Sonata model during forward(), not here. When
    *center_coord* is provided it is forwarded to :func:`crop_point_view` to
    constrain the crop center.
    """
    working_view = crop_point_view(
        view, keep_ratio=crop_keep_ratio, center_coord=center_coord
    )
    coord = working_view["coord"].clone()
    origin_coord = working_view["origin_coord"].clone()
    total_energy = working_view["total_energy"].clone()
    source_index = working_view["source_index"].clone()
    patch_id = working_view["patch_id"].clone()

    if phi_rotation_max > 0.0:
        angle = (
            torch.rand(1, device=coord.device).item() * 2.0 - 1.0
        ) * phi_rotation_max
        coord = rotate_around_beam_axis(coord, angle)
    if coord_noise_scale > 0.0:
        coord = coord + torch.randn_like(coord) * coord_noise_scale
    if feat_noise_scale > 0.0:
        jitter = torch.randn_like(total_energy) * feat_noise_scale
        jitter = jitter.clamp(-feat_noise_scale, feat_noise_scale)
        total_energy = (total_energy * (1.0 + jitter)).clamp_min(0.0)

    if point_dropout > 0.0 and coord.shape[0] > 1:
        keep_mask = torch.rand(coord.shape[0], device=coord.device) >= point_dropout
        if not torch.any(keep_mask):
            keep_mask[torch.randint(0, coord.shape[0], (1,), device=coord.device)] = (
                True
            )
        coord = coord[keep_mask]
        origin_coord = origin_coord[keep_mask]
        total_energy = total_energy[keep_mask]
        source_index = source_index[keep_mask]
        patch_id = patch_id[keep_mask]

    return validate_point_view(
        {
            "coord": coord,
            "origin_coord": origin_coord,
            "feat": assemble_point_features(coord, total_energy),
            "offset": torch.tensor(
                [coord.shape[0]], dtype=torch.long, device=coord.device
            ),
            "grid_size": working_view["grid_size"].clone(),
            "source_index": source_index,
            "total_energy": total_energy,
            "patch_id": patch_id,
            "mask": torch.zeros(coord.shape[0], dtype=torch.bool, device=coord.device),
            "view_kind": view_kind,
        }
    )


def batch_point_views(views: Sequence[Mapping[str, Any]]) -> PointView:
    """Concatenate per-event point views into one batched mapping.

    Source indices and patch ids are offset so they remain unique across events.
    That keeps the later point-level matching logic simple and local to the batch.
    """
    if not views:
        raise ValueError("At least one point view is required to create a batch.")

    normalized_views = [validate_point_view(view) for view in views]
    grid_size = normalized_views[0]["grid_size"]
    view_kind = normalized_views[0].get("view_kind", "unknown")
    for view in normalized_views[1:]:
        if not torch.allclose(view["grid_size"], grid_size):
            raise ValueError(
                "All point views in a batch must share the same grid size."
            )

    coord = torch.cat([view["coord"] for view in normalized_views], dim=0)
    origin_coord = torch.cat([view["origin_coord"] for view in normalized_views], dim=0)
    total_energy = torch.cat([view["total_energy"] for view in normalized_views], dim=0)
    source_index_parts = []
    patch_id_parts = []
    mask = torch.cat([view["mask"] for view in normalized_views], dim=0)
    counts = []
    source_offset = 0
    patch_offset = 0
    for view in normalized_views:
        counts.append(view["coord"].shape[0])
        source_index_parts.append(view["source_index"] + source_offset)
        patch_id_parts.append(view["patch_id"] + patch_offset)
        source_offset += int(view["source_index"].max().item()) + 1
        patch_offset += int(view["patch_id"].max().item()) + 1

    source_index = torch.cat(source_index_parts, dim=0)
    patch_id = torch.cat(patch_id_parts, dim=0)
    counts_tensor = torch.tensor(counts, device=coord.device, dtype=torch.long)

    return validate_point_view(
        {
            "coord": coord,
            "origin_coord": origin_coord,
            "feat": assemble_point_features(coord, total_energy),
            "offset": torch.cumsum(counts_tensor, dim=0),
            "grid_size": grid_size.clone(),
            "source_index": source_index,
            "total_energy": total_energy,
            "patch_id": patch_id,
            "mask": mask,
            "view_kind": view_kind,
        }
    )


def _sample_crop_keep_ratio(
    min_ratio: float, max_ratio: float, device: torch.device
) -> float:
    if min_ratio <= 0 or max_ratio <= 0:
        raise ValueError("Crop ratios must be positive.")
    if min_ratio > max_ratio:
        raise ValueError("Minimum crop ratio cannot exceed maximum crop ratio.")
    if min_ratio == max_ratio:
        return float(min_ratio)
    return float(torch.empty(1, device=device).uniform_(min_ratio, max_ratio).item())


def _sample_center_from_view(view: PointView) -> torch.Tensor:
    """Draw a random point coordinate from a view's pre-augmentation origin."""
    origin = view["origin_coord"]
    index = torch.randint(0, origin.shape[0], (1,), device=origin.device)
    return origin[index[0]]


def build_sonata_batch(
    events: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    max_calo_hits: int | None = None,
    grid_size: float = DEFAULT_POINT_GRID_SIZE,
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
    phi_rotation_max: float = 0.25,
    point_dropout: float = 0.05,
    num_global_views: int = 2,
    num_local_views: int = 4,
    global_crop_min_ratio: float = 0.4,
    global_crop_max_ratio: float = 1.0,
    local_crop_min_ratio: float = 0.1,
    local_crop_max_ratio: float = 0.4,
    coord_center: Sequence[float] | None = None,
    coord_scale: float | None = None,
    energy_transform: str = "raw",
    energy_min: float = 1.0e-2,
    energy_max: float = 20.0,
    grid_sample_enabled: bool = True,
    grid_sample_size: float = 0.002,
    constrain_to_principal: bool = True,
) -> SonataBatch:
    """Build the packed global/local multiview batch expected by Sonata.

    When *constrain_to_principal* is true (Sonata default, App. A.1) the crop
    center of every non-principal global view and every local view is drawn from
    the principal (first) global view's pre-augmentation coordinates, so those
    crops are centered within the principal footprint while still sampling from
    the full event. The first global view keeps an unconstrained random center.
    """

    global_views: list[PointView] = []
    local_views: list[PointView] = []
    for event in events:
        base_view = build_point_view_from_event(
            event,
            device=device,
            max_calo_hits=max_calo_hits,
            grid_size=grid_size,
            coord_center=coord_center,
            coord_scale=coord_scale,
            energy_transform=energy_transform,
            energy_min=energy_min,
            energy_max=energy_max,
            grid_sample_enabled=grid_sample_enabled,
            grid_sample_size=grid_sample_size,
        )

        principal: PointView | None = None
        for view_index in range(num_global_views):
            keep_ratio = _sample_crop_keep_ratio(
                global_crop_min_ratio, global_crop_max_ratio, base_view["coord"].device
            )
            center_coord = None
            if constrain_to_principal and view_index > 0 and principal is not None:
                center_coord = _sample_center_from_view(principal)
            view = augment_point_view(
                base_view,
                coord_noise_scale=coord_noise_scale,
                feat_noise_scale=feat_noise_scale,
                phi_rotation_max=phi_rotation_max,
                crop_keep_ratio=keep_ratio,
                point_dropout=point_dropout,
                view_kind=f"sonata_global_{view_index}",
                center_coord=center_coord,
            )
            if constrain_to_principal and view_index == 0:
                principal = view
            global_views.append(view)

        for view_index in range(num_local_views):
            keep_ratio = _sample_crop_keep_ratio(
                local_crop_min_ratio, local_crop_max_ratio, base_view["coord"].device
            )
            center_coord = None
            if constrain_to_principal and principal is not None:
                center_coord = _sample_center_from_view(principal)
            local_views.append(
                augment_point_view(
                    base_view,
                    coord_noise_scale=coord_noise_scale,
                    feat_noise_scale=feat_noise_scale,
                    phi_rotation_max=phi_rotation_max,
                    crop_keep_ratio=keep_ratio,
                    point_dropout=point_dropout,
                    view_kind=f"sonata_local_{view_index}",
                    center_coord=center_coord,
                )
            )

    global_batch = batch_point_views(global_views)
    local_batch = batch_point_views(local_views)
    return {
        "global_coord": global_batch["coord"],
        "global_origin_coord": global_batch["origin_coord"],
        "global_feat": global_batch["feat"],
        "global_offset": global_batch["offset"],
        "local_coord": local_batch["coord"],
        "local_origin_coord": local_batch["origin_coord"],
        "local_feat": local_batch["feat"],
        "local_offset": local_batch["offset"],
        "grid_size": torch.tensor([grid_size], dtype=torch.float32, device=device),
    }
