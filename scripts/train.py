"""Thin CLI driver that launches a Ray Train run for Sonata self-distillation."""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

# Import Comet before torch so the optional backend can initialize cleanly.
import comet_ml

del comet_ml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import ray.train
from omegaconf import DictConfig
from ray.train.torch import TorchTrainer

from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    resolve_run_lifecycle,
    to_plain_container,
)
from collider_fm.training_loop import train_loop_per_worker


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the train CLI.

    Returns:
        argparse.ArgumentParser: Configured parser with `--config` and dotlist
        override support.
    """
    return build_config_arg_parser(
        description="Train the ColliderFM Sonata model using Ray Train.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train.py\n"
            "  uv run python scripts/train.py training.batch_size=16 training.num_epochs=10\n"
            "  uv run python scripts/train.py training.num_gpus=4 data.local_files_only=true\n"
            "  uv run python scripts/train.py training.experiment_dir=/tmp/collider-runs training.run_name=my_run training.log_backend=jsonl"
        ),
        config_sections=(
            "data",
            "views",
            "model.training",
            "training",
        ),
    )


def should_eval_after_training(config: DictConfig, has_checkpoint: bool) -> tuple[bool, str]:
    """Decide whether to run the post-training evaluation.

    Args:
        config (DictConfig): Full project config.
        has_checkpoint (bool): Whether training produced a checkpoint.

    Returns:
        tuple[bool, str]: `(should_run, reason)`. `reason` explains a skip and is
        empty when `should_run` is True.
    """
    training_config = config.training
    if not bool(training_config.get("eval_after_training", True)):
        return False, "training.eval_after_training is false"
    if not has_checkpoint:
        return False, "training produced no checkpoint"
    # Batch-limited runs are debug/smoke runs; a full held-out eval would dwarf them.
    for key in ("max_train_batches", "max_val_batches"):
        if training_config.get(key) is not None:
            return False, f"training.{key} is set (batch-limited debug run)"
    return True, ""


def build_eval_config(config: DictConfig, run_dir: Path, cli_overrides: Sequence[str]) -> DictConfig:
    """Build the config for the post-training evaluation.

    Points `evaluation.checkpoint` at the training run dir (not the raw `model.pt`) so
    `evaluate.resolve_checkpoint` recovers the run's `config.json` and rebuilds the
    exact backbone, and picks the lowest-val_loss checkpoint. Enables the t-SNE/PCA
    phases unless the user set them explicitly on the command line.

    Args:
        config (DictConfig): Full project config.
        run_dir (Path): Resolved training run directory.
        cli_overrides (Sequence[str]): Raw dotlist overrides, used to detect explicit
            `evaluation.*` opt-outs.

    Returns:
        DictConfig: A deep copy configured for the post-training eval.
    """
    eval_config = copy.deepcopy(config)
    eval_config.evaluation.checkpoint = str(run_dir)
    # Leave evaluation.run_name unset so output nests at runs/<run>/eval/<checkpoint>/.
    eval_config.evaluation.run_name = None
    overridden = {override.split("=", 1)[0] for override in cli_overrides}
    for key in ("enable_tsne", "tsne_upcast2"):
        if f"evaluation.{key}" not in overridden:
            eval_config.evaluation[key] = True
    return eval_config


def main() -> None:
    """CLI entry point: resolves run lifecycle and launches a Ray Train `TorchTrainer`."""
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)
    training_config = config.training

    num_gpus = int(training_config.get("num_gpus", 1))
    resume = bool(training_config.get("resume", False))
    overwrite = bool(training_config.get("overwrite", False))
    storage_path = str(training_config.get("ray_storage_path", "/mnt/ceph/users/ewulff/raytrain_results/"))
    resolved_run_dir, resolved_run_name = resolve_run_lifecycle(
        PROJECT_ROOT,
        ray_storage_path=storage_path,
        experiment_dir=training_config.get("experiment_dir"),
        run_name=training_config.get("run_name"),
        resume=resume,
        overwrite=overwrite,
    )

    # Serialize the full config so each Ray worker gets a copy
    config_dict = to_plain_container(config)
    config_dict["_project_root"] = str(PROJECT_ROOT)
    config_dict["training"]["experiment_dir"] = str(resolved_run_dir.parent)
    config_dict["training"]["run_dir"] = str(resolved_run_dir)
    config_dict["training"]["run_name"] = resolved_run_name
    config_dict["training"]["resume"] = resume
    config_dict["training"]["overwrite"] = overwrite

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=config_dict,
        scaling_config=ray.train.ScalingConfig(
            num_workers=num_gpus,
            use_gpu=True,
        ),
        run_config=ray.train.RunConfig(
            name=resolved_run_name,
            storage_path=storage_path,
            failure_config=ray.train.FailureConfig(max_failures=3),
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    print(f"Launching Ray Train: {num_gpus} GPU(s), storage={storage_path}, " f"name={resolved_run_name}, resume={resume}")
    result = trainer.fit()
    has_checkpoint = result.checkpoint is not None
    if has_checkpoint:
        print(f"Training finished. Best checkpoint: {result.checkpoint.path}")
    else:
        print("Training finished. No checkpoint was produced.")

    # Post-training evaluation: runs here on the driver rather than inside the Ray
    # worker, so it executes once (not once per rank) with the GPUs already released.
    should_eval, skip_reason = should_eval_after_training(config, has_checkpoint)
    if not should_eval:
        print(f"Skipping post-training evaluation: {skip_reason}.")
        return

    # Imported lazily: scripts/ is not a package, and evaluate.py pulls in
    # sklearn/matplotlib that the training path does not need.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import run_evaluation

    print(f"\n{'=' * 64}\nRunning post-training evaluation\n{'=' * 64}")
    try:
        run_evaluation(build_eval_config(config, resolved_run_dir, cli_args.overrides))
    except Exception:
        # Training succeeded and its checkpoints are safe; an eval failure must not
        # mark a multi-day training job as failed. Warn loudly and exit 0.
        print("WARNING: post-training evaluation failed. Training itself succeeded " "and the checkpoints are intact; re-run scripts/evaluate.py manually.")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
