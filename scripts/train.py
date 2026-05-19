"""Thin CLI driver that launches a Ray Train run for Sonata self-distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import Comet before torch so the optional backend can initialize cleanly.
import comet_ml
del comet_ml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import ray.train
from ray.train.torch import TorchTrainer

from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    to_plain_container,
)
from collider_fm.training_loop import train_loop_per_worker


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Train the ColliderFM Sonata model using Ray Train.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train.py\n"
            "  uv run python scripts/train.py training.batch_size=16 training.num_epochs=10\n"
            "  uv run python scripts/train.py training.num_gpus=4 data.local_files_only=true\n"
            "  uv run python scripts/train.py training.run_name=my_run training.log_backend=jsonl"
        ),
        config_sections=(
            "data",
            "views",
            "model.training",
            "training",
        ),
    )


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)
    training_config = config.training

    num_gpus = int(training_config.get("num_gpus", 1))
    run_name = str(training_config.get("run_name") or "sonata_run")
    storage_path = str(
        training_config.get("ray_storage_path", "/mnt/ceph/users/ewulff/raytrain_results/")
    )

    # Serialize the full config so each Ray worker gets a copy
    config_dict = to_plain_container(config)
    config_dict["_project_root"] = str(PROJECT_ROOT)
    config_dict["training"]["ray_run_name"] = run_name

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=config_dict,
        scaling_config=ray.train.ScalingConfig(
            num_workers=num_gpus,
            use_gpu=True,
        ),
        run_config=ray.train.RunConfig(
            name=run_name,
            storage_path=storage_path,
            failure_config=ray.train.FailureConfig(max_failures=3),
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    print(f"Launching Ray Train: {num_gpus} GPU(s), storage={storage_path}, name={run_name}")
    result = trainer.fit()
    print(f"Training finished. Best checkpoint: {result.checkpoint.path}")


if __name__ == "__main__":
    main()
