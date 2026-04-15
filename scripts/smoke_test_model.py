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
    sonata_batch_kwargs,
)
from collider_fm.views import (
    SonataBatch,
    build_sonata_batch,
)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Run a compact Sonata smoke test on ColliderML point views.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/smoke_test_model.py\n"
            "  uv run python scripts/smoke_test_model.py smoke_test.train_split=train[:4]\n"
            "  uv run python scripts/smoke_test_model.py data.local_files_only=true\n"
            "  uv run python scripts/smoke_test_model.py smoke_test.allow_synthetic_fallback=true"
        ),
        config_sections=(
            "data",
            "views",
            "model.diagnostics",
            "smoke_test",
        ),
    )


def load_smoke_test_views(
    config: DictConfig, device: torch.device, batch_kwargs: dict[str, object]
) -> tuple[SonataBatch, str]:
    """Load one cached event, or optionally synthesize one for a CUDA-only check."""

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
        model_inputs = build_sonata_batch([event], device=device, **batch_kwargs)
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
        model_inputs = build_sonata_batch([base_event], device=device, **batch_kwargs)
        return model_inputs, f"synthetic fallback ({type(exc).__name__}: {exc})"


def run_smoke_test(project_config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_kwargs = sonata_batch_kwargs(
        project_config,
        "diagnostics",
        max_calo_hits=project_config.views.max_calo_hits,
    )

    model = create_small_model(
        device=device,
        **model_factory_kwargs(select_model_config(project_config, "diagnostics")),
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    if device.type != "cuda":
        print(
            "Skipping forward pass because the smoke test is intended for a CUDA-enabled node."
        )
        return

    batch, data_source = load_smoke_test_views(project_config, device, batch_kwargs)
    max_calo_hits = batch_kwargs["max_calo_hits"]
    point_limit = (
        "full-event point views" if max_calo_hits is None else f"calo<={max_calo_hits}"
    )
    print(f"Smoke test data source: {data_source}")
    model.setup_schedulers(total_steps=1)
    model.step_schedules()
    print(f"Smoke test global points: {batch['global_coord'].shape[0]} ({point_limit})")
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
