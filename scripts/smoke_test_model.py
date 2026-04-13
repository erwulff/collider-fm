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
from collider_fm.model import create_small_model
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    model_factory_kwargs,
    select_model_config,
)
from collider_fm.views import (
    DistillationBatch,
    SonataBatch,
    build_distillation_views,
    build_sonata_batch,
)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Run a compact Panda-style smoke test on ColliderML point views.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/smoke_test_model.py\n"
            "  uv run python scripts/smoke_test_model.py smoke_test.train_split=train[:4]\n"
            "  uv run python scripts/smoke_test_model.py data.local_files_only=true\n"
            "  uv run python scripts/smoke_test_model.py smoke_test.allow_synthetic_fallback=true\n"
            "  uv run python scripts/smoke_test_model.py model.recipe=sonata"
        ),
        config_sections=(
            "data",
            "views",
            "sonata_views",
            "model.recipe",
            "model.diagnostics",
            "model.sonata_diagnostics",
            "smoke_test",
        ),
    )


def load_smoke_test_views(
    config: DictConfig, device: torch.device
) -> tuple[DistillationBatch | SonataBatch, str]:
    data_config = config.data
    smoke_config = config.smoke_test
    recipe = str(config.model.get("recipe", "legacy"))
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
        if recipe == "legacy":
            model_inputs = build_distillation_views(
                [event],
                device=device,
                max_calo_hits=smoke_config.max_calo_hits,
            )
        else:
            model_inputs = build_sonata_batch(
                [event],
                device=device,
                max_calo_hits=config.sonata_views.max_calo_hits,
                grid_size=float(select_model_config(config, "diagnostics").grid_size),
                coord_noise_scale=config.sonata_views.coord_noise_scale,
                feat_noise_scale=config.sonata_views.energy_jitter_scale,
                point_dropout=config.sonata_views.point_dropout,
                num_global_views=config.sonata_views.num_global_views,
                num_local_views=config.sonata_views.num_local_views,
                global_crop_min_ratio=config.sonata_views.global_crop_min_ratio,
                global_crop_max_ratio=config.sonata_views.global_crop_max_ratio,
                local_crop_min_ratio=config.sonata_views.local_crop_min_ratio,
                local_crop_max_ratio=config.sonata_views.local_crop_max_ratio,
                coord_center=config.sonata_views.coord_center,
                coord_scale=config.sonata_views.coord_scale,
                energy_transform=config.sonata_views.energy_transform,
                energy_min=config.sonata_views.energy_min,
                energy_max=config.sonata_views.energy_max,
            )
        return model_inputs, "ColliderML cached event"
    except Exception as exc:
        if not smoke_config.get("allow_synthetic_fallback", False):
            raise RuntimeError(
                "ColliderML smoke test could not load cached data. "
                "Download or cache the dataset first, or rerun with "
                "smoke_test.allow_synthetic_fallback=true "
                "for a CUDA-only synthetic check."
            ) from exc
        coord = torch.rand(64, 3, device=device)
        total_energy = torch.rand(64, device=device)
        base_event = {
            "calo_hits": {
                "x": coord[:, 0],
                "y": coord[:, 1],
                "z": coord[:, 2],
                "total_energy": total_energy,
            }
        }
        if recipe == "legacy":
            model_inputs = build_distillation_views(
                [base_event],
                device=device,
                max_calo_hits=smoke_config.max_calo_hits,
            )
        else:
            model_inputs = build_sonata_batch(
                [base_event],
                device=device,
                max_calo_hits=config.sonata_views.max_calo_hits,
                grid_size=float(select_model_config(config, "diagnostics").grid_size),
                coord_noise_scale=config.sonata_views.coord_noise_scale,
                feat_noise_scale=config.sonata_views.energy_jitter_scale,
                point_dropout=config.sonata_views.point_dropout,
                num_global_views=config.sonata_views.num_global_views,
                num_local_views=config.sonata_views.num_local_views,
                global_crop_min_ratio=config.sonata_views.global_crop_min_ratio,
                global_crop_max_ratio=config.sonata_views.global_crop_max_ratio,
                local_crop_min_ratio=config.sonata_views.local_crop_min_ratio,
                local_crop_max_ratio=config.sonata_views.local_crop_max_ratio,
                coord_center=config.sonata_views.coord_center,
                coord_scale=config.sonata_views.coord_scale,
                energy_transform=config.sonata_views.energy_transform,
                energy_min=config.sonata_views.energy_min,
                energy_max=config.sonata_views.energy_max,
            )
        return model_inputs, f"synthetic fallback ({type(exc).__name__}: {exc})"


def run_smoke_test(project_config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    recipe = str(project_config.model.get("recipe", "legacy"))

    model = create_small_model(
        recipe=recipe,
        device=device,
        **model_factory_kwargs(select_model_config(project_config, "diagnostics")),
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
    if recipe == "legacy":
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
    else:
        model.setup_schedulers(total_steps=1)
        model.step_schedules()
        print(
            f"Smoke test global points: {batch['global_coord'].shape[0]} (calo<={max_calo_hits})"
        )
        result_dict = model(batch)
        loss = result_dict["loss"]
        model.update_teacher(momentum=None)
        monitoring = getattr(model, "last_monitoring_state", {})
        student_logits = monitoring.get("student_logits")
        print(
            f"Forward pass successful. Student point-logit shape: {tuple(student_logits.shape) if student_logits is not None else (0, 0)}"
        )
    print(f"Distillation loss: {loss.item():.4f}")


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_smoke_test(load_project_config(cli_args.config, cli_args.overrides))
