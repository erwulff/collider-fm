from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.model import PandaSelfDistillation

SMOKE_TEST_MAX_TRACKER_HITS = 128
SMOKE_TEST_MAX_CALO_HITS = 256


def _normalize_feature(values: torch.Tensor) -> torch.Tensor:
    scale = values.abs().max().clamp_min(1.0)
    return values / scale


def _sample_hit_indices(num_hits: int, max_hits: int, device: torch.device) -> torch.Tensor:
    if num_hits <= max_hits:
        return torch.arange(num_hits, device=device)
    return torch.linspace(0, num_hits - 1, steps=max_hits, device=device).round().long()


def build_point_view_from_event(event: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tracker_hits = event["tracker_hits"]
    calo_hits = event["calo_hits"]

    tracker_indices = _sample_hit_indices(
        num_hits=len(tracker_hits["x"]),
        max_hits=SMOKE_TEST_MAX_TRACKER_HITS,
        device=device,
    )
    calo_indices = _sample_hit_indices(
        num_hits=len(calo_hits["x"]),
        max_hits=SMOKE_TEST_MAX_CALO_HITS,
        device=device,
    )

    tracker_coord = torch.stack(
        [tracker_hits["x"][tracker_indices], tracker_hits["y"][tracker_indices], tracker_hits["z"][tracker_indices]],
        dim=1,
    ).to(device=device, dtype=torch.float32)
    calo_coord = torch.stack(
        [calo_hits["x"][calo_indices], calo_hits["y"][calo_indices], calo_hits["z"][calo_indices]],
        dim=1,
    ).to(device=device, dtype=torch.float32)

    tracker_time = _normalize_feature(
        torch.as_tensor(
            tracker_hits.get("time", torch.zeros(len(tracker_hits["x"])))[tracker_indices],
            dtype=torch.float32,
            device=device,
        )
    )
    calo_energy = _normalize_feature(
        torch.as_tensor(
            calo_hits.get("total_energy", torch.zeros(len(calo_hits["x"])))[calo_indices],
            dtype=torch.float32,
            device=device,
        )
    )

    coord = torch.cat([tracker_coord, calo_coord], dim=0)
    radius = _normalize_feature(torch.linalg.norm(coord, dim=1))
    signal = torch.cat([tracker_time, calo_energy], dim=0)
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
        "grid_size": torch.tensor(10.0, dtype=torch.float32, device=device),
    }


def augment_point_view(
    view: Mapping[str, torch.Tensor],
    coord_noise_scale: float = 0.5,
    feat_noise_scale: float = 0.01,
) -> dict[str, torch.Tensor]:
    coord = view["coord"].clone()
    feat = view["feat"].clone()

    coord = coord + torch.randn_like(coord) * coord_noise_scale
    feat[:, :3] = coord
    feat[:, 3:5] = feat[:, 3:5] + torch.randn_like(feat[:, 3:5]) * feat_noise_scale

    return {
        "coord": coord,
        "feat": feat,
        "offset": view["offset"].clone(),
        "grid_size": view["grid_size"].clone(),
    }


def make_random_view(num_points: int, in_channels: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "coord": torch.rand(num_points, 3, device=device),
        "feat": torch.rand(num_points, in_channels, device=device),
        "offset": torch.tensor([num_points], dtype=torch.long, device=device),
        "grid_size": torch.tensor(0.05, device=device),
    }


def load_smoke_test_views(device: torch.device) -> tuple[list[dict[str, torch.Tensor]], str]:
    try:
        dataset = ColliderMLDataset(
            split="train[:1]",
            object_types=["tracker_hits", "calo_hits"],
        )
        event = dataset[0]
        base_view = build_point_view_from_event(event, device=device)
        views = [base_view, augment_point_view(base_view)]
        return views, "ColliderML cached event"
    except Exception as exc:
        views = [make_random_view(num_points=64, in_channels=6, device=device) for _ in range(2)]
        return views, f"synthetic fallback ({type(exc).__name__}: {exc})"


def run_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PandaSelfDistillation(
        in_channels=6,
        embed_channels=8,
        num_prototypes=32,
        projection_dim=8,
        prediction_dim=16,
        backbone_kwargs={
            "enc_depths": (1, 1, 1, 1, 1),
            "enc_channels": (8, 12, 16, 24, 32),
            "enc_num_head": (1, 1, 2, 4, 4),
            "enc_patch_size": (4, 4, 4, 4, 4),
            "shuffle_orders": False,
            "enable_flash": False,
        },
    ).to(device)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    if device.type != "cuda":
        print("Skipping forward pass because the Panda smoke test is intended for a CUDA-enabled node.")
        return

    views, data_source = load_smoke_test_views(device)
    print(f"Smoke test data source: {data_source}")
    print(
        f"Smoke test points per view: {views[0]['coord'].shape[0]} "
        f"(tracker<={SMOKE_TEST_MAX_TRACKER_HITS}, calo<={SMOKE_TEST_MAX_CALO_HITS})"
    )
    student_outputs, teacher_outputs = model(views)
    loss = model.distillation_loss(student_outputs, teacher_outputs)
    model.update_center(teacher_outputs)
    model.update_teacher(momentum=0.99)

    print(f"Forward pass successful. Student output shape: {tuple(student_outputs[0].shape)}")
    print(f"Distillation loss: {loss.item():.4f}")


if __name__ == "__main__":
    run_smoke_test()
