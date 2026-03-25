from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data import RawColliderEvent


DEFAULT_MULTIMODAL_GRID_SIZE = 10.0


@dataclass(frozen=True)
class MultimodalPointBatch:
    """Model-facing tensors for one fused tracker-plus-calo event.

    The batch preserves modality separation for stem inputs while also exposing a
    fused sparse point layout that later view builders and backbones can share.
    """

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


def _sample_indices(num_items: int, max_items: int | None, device: torch.device) -> torch.Tensor:
    """Match the current repo downsampling policy with evenly spaced selection."""
    if max_items is None or num_items <= max_items:
        return torch.arange(num_items, device=device)
    return torch.linspace(0, num_items - 1, steps=max_items, device=device).round().long()


def _require_tensor(
    values: dict[str, Any],
    key: str,
    device: torch.device,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key not in values:
        raise KeyError(f"Missing required field '{key}'.")
    return torch.as_tensor(values[key], dtype=dtype, device=device).flatten()


def _optional_long_tensor(
    values: dict[str, Any],
    key: str,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if key not in values:
        return torch.zeros(length, dtype=torch.long, device=device)
    return torch.as_tensor(values[key], dtype=torch.long, device=device).flatten()


def sample_tracker_hits(
    event: RawColliderEvent,
    device: torch.device,
    max_tracker_hits: int | None = None,
) -> dict[str, torch.Tensor]:
    """Extract tracker tensors needed by the multimodal encoder path."""
    tracker_hits = dict(event["tracker_hits"])
    x = _require_tensor(tracker_hits, "x", device, dtype=torch.float32)
    indices = _sample_indices(x.numel(), max_tracker_hits, device)

    sampled = {
        "x": x[indices],
        "y": _require_tensor(tracker_hits, "y", device, dtype=torch.float32)[indices],
        "z": _require_tensor(tracker_hits, "z", device, dtype=torch.float32)[indices],
        "time": _require_tensor(tracker_hits, "time", device, dtype=torch.float32)[indices],
        "detector": _optional_long_tensor(tracker_hits, "detector", x.numel(), device)[indices],
        "volume_id": _optional_long_tensor(tracker_hits, "volume_id", x.numel(), device)[indices],
        "layer_id": _optional_long_tensor(tracker_hits, "layer_id", x.numel(), device)[indices],
        "indices": indices,
    }
    if "surface_id" in tracker_hits:
        sampled["surface_id"] = _optional_long_tensor(tracker_hits, "surface_id", x.numel(), device)[indices]
    return sampled


def sample_calo_hits(
    event: RawColliderEvent,
    device: torch.device,
    max_calo_hits: int | None = None,
) -> dict[str, torch.Tensor]:
    """Extract calorimeter tensors needed by the multimodal encoder path."""
    calo_hits = dict(event["calo_hits"])
    x = _require_tensor(calo_hits, "x", device, dtype=torch.float32)
    indices = _sample_indices(x.numel(), max_calo_hits, device)

    sampled = {
        "x": x[indices],
        "y": _require_tensor(calo_hits, "y", device, dtype=torch.float32)[indices],
        "z": _require_tensor(calo_hits, "z", device, dtype=torch.float32)[indices],
        "total_energy": _require_tensor(calo_hits, "total_energy", device, dtype=torch.float32)[indices],
        "detector": _optional_long_tensor(calo_hits, "detector", x.numel(), device)[indices],
        "indices": indices,
    }
    return sampled


def assign_point_ids(
    event: RawColliderEvent,
    tracker_indices: torch.Tensor,
    calo_indices: torch.Tensor,
) -> torch.Tensor:
    """Create stable per-event point IDs before any crop or mask transforms.

    Tracker points keep their original row positions. Calorimeter points are offset
    by the full tracker-hit count so IDs stay unique within the fused event.
    """
    tracker_count = len(event["tracker_hits"]["x"])
    tracker_point_id = tracker_indices.to(dtype=torch.long)
    calo_point_id = calo_indices.to(dtype=torch.long) + tracker_count
    return torch.cat([tracker_point_id, calo_point_id], dim=0)


def build_multimodal_points(
    event: RawColliderEvent,
    device: torch.device,
    max_tracker_hits: int | None = None,
    max_calo_hits: int | None = None,
    grid_size: float = DEFAULT_MULTIMODAL_GRID_SIZE,
) -> MultimodalPointBatch:
    """Build the fused tracker-plus-calo point payload for the new model path."""
    tracker = sample_tracker_hits(event, device=device, max_tracker_hits=max_tracker_hits)
    calo = sample_calo_hits(event, device=device, max_calo_hits=max_calo_hits)

    tracker_coord = torch.stack([tracker["x"], tracker["y"], tracker["z"]], dim=1)
    calo_coord = torch.stack([calo["x"], calo["y"], calo["z"]], dim=1)
    coord = torch.cat([tracker_coord, calo_coord], dim=0)

    tracker_continuous = torch.stack(
        [tracker["x"], tracker["y"], tracker["z"], tracker["time"]],
        dim=1,
    )
    calo_continuous = torch.stack(
        [calo["x"], calo["y"], calo["z"], torch.log1p(calo["total_energy"].clamp_min(0.0))],
        dim=1,
    )

    tracker_index = torch.arange(tracker_coord.shape[0], device=device, dtype=torch.long)
    calo_index = torch.arange(calo_coord.shape[0], device=device, dtype=torch.long) + tracker_coord.shape[0]
    modality_id = torch.cat(
        [
            torch.zeros(tracker_coord.shape[0], dtype=torch.long, device=device),
            torch.ones(calo_coord.shape[0], dtype=torch.long, device=device),
        ],
        dim=0,
    )
    point_id = assign_point_ids(event, tracker["indices"], calo["indices"])
    event_id = torch.zeros(coord.shape[0], dtype=torch.long, device=device)

    tracker_categorical = {
        "detector": tracker["detector"],
        "volume_id": tracker["volume_id"],
        "layer_id": tracker["layer_id"],
    }
    if "surface_id" in tracker:
        tracker_categorical["surface_id"] = tracker["surface_id"]

    calo_categorical = {"detector": calo["detector"]}

    return MultimodalPointBatch(
        coord=coord,
        tracker_continuous=tracker_continuous,
        calo_continuous=calo_continuous,
        tracker_categorical=tracker_categorical,
        calo_categorical=calo_categorical,
        modality_id=modality_id,
        point_id=point_id,
        event_id=event_id,
        offset=torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
        grid_size=torch.tensor(grid_size, dtype=torch.float32, device=device),
        tracker_index=tracker_index,
        calo_index=calo_index,
    )


def build_model_inputs(points: MultimodalPointBatch) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Return the model-facing tensors and metadata needed by the new encoder path."""
    return {
        "coord": points.coord,
        "tracker_continuous": points.tracker_continuous,
        "calo_continuous": points.calo_continuous,
        "tracker_categorical": points.tracker_categorical,
        "calo_categorical": points.calo_categorical,
        "modality_id": points.modality_id,
        "point_id": points.point_id,
        "event_id": points.event_id,
        "offset": points.offset,
        "grid_size": points.grid_size,
        "tracker_index": points.tracker_index,
        "calo_index": points.calo_index,
    }
