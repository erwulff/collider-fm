from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.model import create_small_panda_model
from collider_fm.views import (
    POINT_FEATURE_DIM,
    augment_point_view,
    build_point_view_from_event,
    make_random_view,
)

SMOKE_TEST_MAX_CALO_HITS = 256


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compact calo-only Panda-style smoke test."
    )
    parser.add_argument("--train-split", default="train[:1]")
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    return parser


def load_smoke_test_views(
    args: argparse.Namespace, device: torch.device
) -> tuple[list[dict[str, torch.Tensor]], str]:
    try:
        dataset = ColliderMLDataset(
            split=args.train_split,
            dataset_type=args.dataset_type,
            pu_config=args.pu_config,
            cache_dir=args.cache_dir,
            object_types=["calo_hits"],
        )
        event = dataset[0]
        base_view = build_point_view_from_event(
            event, device=device, max_calo_hits=SMOKE_TEST_MAX_CALO_HITS
        )
        return [
            base_view,
            augment_point_view(base_view),
        ], "ColliderML cached calo event"
    except Exception as exc:
        if not args.allow_synthetic_fallback:
            raise RuntimeError(
                "Could not load cached ColliderML calo data for the smoke test."
            ) from exc
        views = [
            make_random_view(
                num_points=64, in_channels=POINT_FEATURE_DIM, device=device
            )
            for _ in range(2)
        ]
        return views, f"synthetic fallback ({type(exc).__name__}: {exc})"


def run_smoke_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_small_panda_model(device=device)
    if device.type != "cuda":
        print(
            "Skipping forward pass because the current PTv3/spconv path is CUDA-only."
        )
        return

    views, source = load_smoke_test_views(args, device)
    print(f"Smoke test data source: {source}")
    print(
        f"Smoke test points per view: {views[0]['coord'].shape[0]} (calo<={SMOKE_TEST_MAX_CALO_HITS})"
    )

    student_outputs, teacher_outputs = model(views)
    loss = model.distillation_loss(student_outputs, teacher_outputs)
    model.update_center(teacher_outputs)
    model.update_teacher(momentum=0.99)

    print(
        f"Forward pass successful. Student output shape: {tuple(student_outputs[0].shape)}"
    )
    print(f"Distillation loss: {loss.item():.4f}")


if __name__ == "__main__":
    run_smoke_test(build_arg_parser().parse_args())
