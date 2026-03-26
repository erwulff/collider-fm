from __future__ import annotations

import argparse
import sys
from argparse import SUPPRESS
from typing import Any
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.model import create_small_panda_model
from collider_fm.project_config import (
    DEFAULT_CONFIG_PATH,
    load_project_config_from_cli,
    model_factory_kwargs,
    to_plain_container,
)
from collider_fm.views import build_distillation_views

SMOKE_TEST_MAX_CALO_HITS = 256


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compact Panda-style smoke test on ColliderML point views."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--train-split", default=SUPPRESS)
    parser.add_argument("--dataset-name", default=SUPPRESS)
    parser.add_argument("--dataset-type", default=SUPPRESS)
    parser.add_argument("--pu-config", default=SUPPRESS)
    parser.add_argument("--cache-dir", default=SUPPRESS)
    parser.add_argument("--dataset-revision", default=SUPPRESS)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=SUPPRESS,
        help="Load ColliderML only from the local cache.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Use a random synthetic point cloud when ColliderML data is unavailable.",
    )
    return parser


def resolve_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, object]]:
    project_config, resolved_config_path = load_project_config_from_cli(argv)
    parser = build_arg_parser()
    parsed_args = parser.parse_args(argv)
    cli_values = vars(parsed_args)
    data_config = to_plain_container(project_config.data)
    smoke_config = to_plain_container(project_config.smoke_test)
    merged = {
        "config": str(resolved_config_path),
        "train_split": smoke_config["train_split"],
        "dataset_name": data_config["dataset_name"],
        "dataset_type": data_config["dataset_type"],
        "pu_config": data_config["pu_config"],
        "cache_dir": data_config["cache_dir"],
        "dataset_revision": data_config["dataset_revision"],
        "local_files_only": data_config["local_files_only"],
        "allow_synthetic_fallback": False,
    } | cli_values
    return argparse.Namespace(**merged), to_plain_container(project_config)


def load_smoke_test_views(
    args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, Any], str]:
    try:
        dataset = ColliderMLDataset(
            dataset_name=args.dataset_name,
            split=args.train_split,
            dataset_type=args.dataset_type,
            pu_config=args.pu_config,
            cache_dir=args.cache_dir,
            object_types=["calo_hits"],
            dataset_revision=args.dataset_revision,
            local_files_only=args.local_files_only,
        )
        event = dataset[0]
        distillation_batch = build_distillation_views(
            [event],
            device=device,
            max_calo_hits=SMOKE_TEST_MAX_CALO_HITS,
        )
        return distillation_batch, "ColliderML cached event"
    except Exception as exc:
        if not args.allow_synthetic_fallback:
            raise RuntimeError(
                "ColliderML smoke test could not load cached data. "
                "Download or cache the dataset first, or rerun with --allow-synthetic-fallback "
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
            max_calo_hits=SMOKE_TEST_MAX_CALO_HITS,
        )
        return distillation_batch, f"synthetic fallback ({type(exc).__name__}: {exc})"


def run_smoke_test(args: argparse.Namespace, project_config: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_small_panda_model(
        device=device,
        **model_factory_kwargs(project_config["model"]["diagnostics"]),
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    if device.type != "cuda":
        print(
            "Skipping forward pass because the Panda smoke test is intended for a CUDA-enabled node."
        )
        return

    batch, data_source = load_smoke_test_views(args, device)
    print(f"Smoke test data source: {data_source}")
    print(
        f"Smoke test student points: {batch['student_views'][0]['coord'].shape[0]} (calo<={SMOKE_TEST_MAX_CALO_HITS})"
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
    resolved_args, resolved_project_config = resolve_args()
    run_smoke_test(resolved_args, resolved_project_config)
