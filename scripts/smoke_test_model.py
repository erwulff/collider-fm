from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.model import PandaSelfDistillation
from collider_fm.views import augment_point_view, build_point_view_from_event, make_random_view

SMOKE_TEST_MAX_TRACKER_HITS = 128
SMOKE_TEST_MAX_CALO_HITS = 256


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
    print(f"Smoke test points per view: {views[0]['coord'].shape[0]} " f"(tracker<={SMOKE_TEST_MAX_TRACKER_HITS}, calo<={SMOKE_TEST_MAX_CALO_HITS})")
    student_outputs, teacher_outputs = model(views)
    loss = model.distillation_loss(student_outputs, teacher_outputs)
    model.update_center(teacher_outputs)
    model.update_teacher(momentum=0.99)

    print(f"Forward pass successful. Student output shape: {tuple(student_outputs[0].shape)}")
    print(f"Distillation loss: {loss.item():.4f}")


if __name__ == "__main__":
    run_smoke_test()
