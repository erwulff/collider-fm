from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.model import create_small_panda_model
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    model_factory_kwargs,
)
from collider_fm.views import DistillationBatch, build_distillation_views


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Run a compact Panda-style smoke test on ColliderML point views.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/smoke_test_model.py\n"
            "  uv run python scripts/smoke_test_model.py smoke_test.train_split=train[:4]\n"
            "  uv run python scripts/smoke_test_model.py data.local_files_only=true\n"
            "  uv run python scripts/smoke_test_model.py smoke_test.allow_synthetic_fallback=true"
        ),
        config_sections=("data", "model.diagnostics", "smoke_test"),
    )


def load_smoke_test_views(
    config: DictConfig, device: torch.device
) -> tuple[DistillationBatch, str]:
    data_config = config.data
    smoke_config = config.smoke_test
    try:
        dataset = ColliderMLDataset(
            dataset_name=data_config.dataset_name,
            split=smoke_config.train_split,
            dataset_type=data_config.dataset_type,
            pu_config=data_config.pu_config,
            cache_dir=data_config.cache_dir,
            object_types=["calo_hits"],
            dataset_revision=data_config.dataset_revision,
            local_files_only=data_config.local_files_only,
        )
        event = dataset[0]
        distillation_batch = build_distillation_views(
            [event],
            device=device,
            max_calo_hits=smoke_config.max_calo_hits,
        )
        return distillation_batch, "ColliderML cached event"
    except Exception as exc:
        if not smoke_config.get("allow_synthetic_fallback", False):
            raise RuntimeError(
                "ColliderML smoke test could not load cached data. "
                "Download or cache the dataset first, or rerun with "
                "smoke_test.allow_synthetic_fallback=true "
                "for a CUDA-only synthetic check."
            ) from exc
        coord = torch.rand(64, 3, device=device)
        energy = torch.rand(64, device=device)
        base_event = {
            "calo_hits": {
                "x": coord[:, 0],
                "y": coord[:, 1],
                "z": coord[:, 2],
                "energy": energy,
            }
        }
        distillation_batch = build_distillation_views(
            [base_event],
            device=device,
            max_calo_hits=smoke_config.max_calo_hits,
        )
        return distillation_batch, f"synthetic fallback ({type(exc).__name__}: {exc})"


def run_smoke_test(project_config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_small_panda_model(
        device=device,
        **model_factory_kwargs(project_config.model.diagnostics),
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    if device.type != "cuda":
        print(
            "Skipping forward pass because the Panda smoke test is intended for a CUDA-enabled node."
        )
        return

    batch, data_source = load_smoke_test_views(project_config, device)
    max_calo_hits = project_config.smoke_test.max_calo_hits
    print(f"Smoke test data source: {data_source}")
    print(
        f"Smoke test student points: {batch['student_views'][0]['coord'].shape[0]} (calo<={max_calo_hits})"
    )
    student_outputs, teacher_outputs = model(batch)
    loss = model.distillation_loss(student_outputs, teacher_outputs)
    model.update_center(teacher_outputs)
    model.update_teacher(momentum=0.99)

    print(
        f"Forward pass successful. Student point-logit shape: {tuple(student_outputs[0]['point_logits'].shape)}"
    )
    print(f"Distillation loss: {loss.item():.4f}")


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_smoke_test(load_project_config(cli_args.config, cli_args.overrides))
