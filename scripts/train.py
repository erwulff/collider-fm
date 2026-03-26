from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import comet_ml
from omegaconf import DictConfig
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import Comet early for its import-time side effects; then remove unused name; the name is unused here.
del comet_ml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset, collate_fn
from collider_fm.experiment_logging import (
    create_experiment_logger,
    ensure_run_directory,
    write_run_config,
)
from collider_fm.model import PandaSelfDistillation, create_training_panda_model
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    model_factory_kwargs,
    to_plain_container,
)
from collider_fm.views import build_distillation_views


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Train the beginner-friendly ColliderFM self-distillation model.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train.py\n"
            "  uv run python scripts/train.py training.batch_size=16 training.num_epochs=10\n"
            "  uv run python scripts/train.py data.local_files_only=true training.log_backend=jsonl"
        ),
        config_sections=("data", "views", "model.training", "training"),
    )


def create_dataloader(config: DictConfig, split: str, shuffle: bool) -> DataLoader:
    """Create the calo-only dataloader used by train and validation."""

    data_config = config.data
    training_config = config.training
    dataset = ColliderMLDataset(
        dataset_name=data_config.dataset_name,
        split=split,
        dataset_type=data_config.dataset_type,
        pu_config=data_config.pu_config,
        object_types=["calo_hits"],
        cache_dir=data_config.cache_dir,
        dataset_revision=data_config.dataset_revision,
        local_files_only=data_config.local_files_only,
    )
    dataloader_kwargs = {
        "batch_size": training_config.batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": training_config.num_workers,
        "pin_memory": bool(training_config.pin_memory and torch.cuda.is_available()),
    }
    if training_config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = training_config.prefetch_factor

    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def resolve_epoch_batch_limit(
    dataloader: DataLoader, requested_max_batches: int | None, phase: str
) -> int:
    """Resolve an epoch batch limit, using the full dataloader when unset."""

    total_batches = len(dataloader)
    if total_batches <= 0:
        raise ValueError(f"The {phase} dataloader produced zero batches.")
    if requested_max_batches is None:
        return total_batches
    if requested_max_batches <= 0:
        raise ValueError(
            f"{phase} max_batches must be a positive integer or None, got {requested_max_batches}."
        )
    return min(total_batches, requested_max_batches)


def linear_warmup(value_start: float, value_end: float, progress: float) -> float:
    """Linearly interpolate between two values on `[0, 1]` progress."""

    progress = min(max(progress, 0.0), 1.0)
    return value_start + progress * (value_end - value_start)


def cosine_momentum(value_start: float, value_end: float, progress: float) -> float:
    """Cosine schedule used for the teacher EMA momentum."""

    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return value_end + (value_start - value_end) * cosine


def learning_rate(optimizer: AdamW) -> float:
    """Return the optimizer learning rate from the first parameter group."""

    return float(optimizer.param_groups[0]["lr"])


def center_norm(model: PandaSelfDistillation) -> float:
    """Return the current norm of the running teacher center."""

    return float(torch.linalg.vector_norm(model.center).item())


def current_teacher_temperature(training_config: DictConfig, epoch_index: int) -> float:
    """Warm the teacher temperature during the early epochs only."""

    if training_config.teacher_temperature_warmup_epochs <= 0:
        return training_config.teacher_temperature_end
    if epoch_index >= training_config.teacher_temperature_warmup_epochs:
        return training_config.teacher_temperature_end
    progress = epoch_index / max(
        1, training_config.teacher_temperature_warmup_epochs - 1
    )
    return linear_warmup(
        training_config.teacher_temperature_start,
        training_config.teacher_temperature_end,
        progress,
    )


def current_teacher_momentum(
    training_config: DictConfig, global_progress: float
) -> float:
    """Increase EMA momentum smoothly over training."""

    return cosine_momentum(
        training_config.teacher_momentum_start,
        training_config.teacher_momentum_end,
        global_progress,
    )


def prototype_usage(
    outputs: list[dict[str, torch.Tensor]], num_prototypes: int
) -> torch.Tensor:
    """Estimate prototype occupancy from the student point assignments."""

    point_logits = torch.cat([output["point_logits"] for output in outputs], dim=0)
    assignments = point_logits.argmax(dim=-1)
    counts = torch.bincount(assignments, minlength=num_prototypes).to(
        dtype=torch.float32
    )
    total = counts.sum().clamp_min(1.0)
    return counts / total


def prototype_entropy(probabilities: torch.Tensor) -> float:
    """Compute entropy for a normalized prototype-usage distribution."""

    probabilities = probabilities.clamp_min(1.0e-8)
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.item())


def embedding_norm(outputs: list[dict[str, torch.Tensor]]) -> float:
    """Average norm of the masked pooled student embeddings."""

    embeddings = torch.cat(
        [output["masked_pooled_projection"] for output in outputs], dim=0
    )
    return float(embeddings.norm(dim=-1).mean().item())


def save_checkpoint(
    run_dir: Path,
    model: PandaSelfDistillation,
    optimizer: AdamW,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    is_best: bool = False,
) -> Path:
    """Write epoch, optimizer, and model state to the run directory."""

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_path)
    if is_best:
        torch.save(payload, checkpoints_dir / "best.pt")
    torch.save(payload, checkpoints_dir / "latest.pt")
    return checkpoint_path


def run_epoch(
    model: PandaSelfDistillation,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    lr_scheduler: CosineAnnealingLR | None,
    max_batches: int,
    max_calo_hits: int,
    coord_noise_scale: float,
    energy_jitter_scale: float,
    global_crop_ratio: float,
    student_mask_fraction: float,
    point_dropout: float,
    teacher_momentum: float,
    phase: str,
) -> tuple[dict[str, float], int]:
    """Run one train or validation epoch over a bounded number of batches.

    The function keeps the control flow intentionally explicit: build batched
    teacher/student views, run the model, optionally step the optimizer, then
    accumulate a few simple metrics that are useful for Monday-style sanity plots.
    """

    is_training = optimizer is not None
    model.train(mode=is_training)
    totals = {
        "loss": 0.0,
        "prototype_entropy": 0.0,
        "embedding_norm": 0.0,
        "masked_fraction": 0.0,
        "data_wait_seconds": 0.0,
        "view_build_seconds": 0.0,
        "model_step_seconds": 0.0,
    }
    processed_batches = 0
    processed_events = 0

    progress_bar = tqdm(
        range(max_batches),
        total=max_batches,
        desc=phase,
        leave=True,
        dynamic_ncols=False,
        ascii=True,
    )

    data_iter = iter(dataloader)

    for batch_index in progress_bar:
        data_wait_start = time.perf_counter()
        try:
            events = next(data_iter)
        except StopIteration:
            break
        data_wait_seconds = time.perf_counter() - data_wait_start

        view_start = time.perf_counter()
        distillation_batch = build_distillation_views(
            events,
            device=device,
            max_calo_hits=max_calo_hits,
            coord_noise_scale=coord_noise_scale,
            feat_noise_scale=energy_jitter_scale,
            global_crop_ratio=global_crop_ratio,
            student_mask_fraction=student_mask_fraction,
            point_dropout=point_dropout,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        view_build_seconds = time.perf_counter() - view_start

        model_step_start = time.perf_counter()

        with torch.set_grad_enabled(is_training):
            student_outputs, teacher_outputs = model(distillation_batch)
            loss = model.distillation_loss(student_outputs, teacher_outputs)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            model.update_center(teacher_outputs)
            model.update_teacher(momentum=teacher_momentum)

        if device.type == "cuda":
            torch.cuda.synchronize()
        model_step_seconds = time.perf_counter() - model_step_start

        usage = prototype_usage(student_outputs, num_prototypes=model.num_prototypes)
        masked_fraction = (
            torch.cat([output["mask"].float() for output in student_outputs])
            .mean()
            .item()
        )
        totals["loss"] += float(loss.item())
        totals["prototype_entropy"] += prototype_entropy(usage)
        totals["embedding_norm"] += embedding_norm(student_outputs)
        totals["masked_fraction"] += float(masked_fraction)
        totals["data_wait_seconds"] += data_wait_seconds
        totals["view_build_seconds"] += view_build_seconds
        totals["model_step_seconds"] += model_step_seconds
        processed_batches += 1
        processed_events += len(events)

        progress_bar.set_postfix(
            data=f"{data_wait_seconds:.1f}s",
            view=f"{view_build_seconds:.1f}s",
            model=f"{model_step_seconds:.1f}s",
            loss=f"{loss.item():.4f}",
            masked=f"{masked_fraction:.3f}",
        )

    progress_bar.close()

    if processed_batches == 0:
        raise ValueError(
            f"No {phase} batches were processed. Check the chosen split and max batch count."
        )

    averaged_metrics = {key: value / processed_batches for key, value in totals.items()}
    averaged_metrics["events_per_second"] = processed_events / max(
        1.0e-6,
        totals["data_wait_seconds"]
        + totals["view_build_seconds"]
        + totals["model_step_seconds"],
    )
    return averaged_metrics, processed_batches


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)
    training_config = config.training
    view_config = config.views
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print(
            "Training requires a CUDA-enabled environment because the current PTv3/spconv stack is GPU-only."
        )
        print("Run this script on a GPU node, for example through a SLURM job.")
        return

    train_loader = create_dataloader(
        config, training_config.train_split, shuffle=training_config.train_shuffle
    )
    val_loader = create_dataloader(config, training_config.val_split, shuffle=False)
    max_train_batches = resolve_epoch_batch_limit(
        train_loader, training_config.max_train_batches, "train"
    )
    max_val_batches = resolve_epoch_batch_limit(
        val_loader, training_config.max_val_batches, "val"
    )

    run_dir, run_name = ensure_run_directory(
        PROJECT_ROOT,
        run_dir=training_config.get("run_dir"),
        run_name=training_config.get("run_name"),
    )
    logger = create_experiment_logger(
        training_config.log_backend, run_dir=run_dir, run_name=run_name
    )
    run_config = to_plain_container(config) | {
        "device": str(device),
        "run_dir": str(run_dir),
        "run_name": run_name,
    }
    config_path = write_run_config(run_dir, run_config)
    logger.log_params(run_config)
    print(f"Run directory: {run_dir}")
    print(f"Run config: {config_path}")

    model = create_training_panda_model(
        device=device,
        **model_factory_kwargs(config.model.training),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_config.num_epochs * max_train_batches),
        eta_min=training_config.min_learning_rate,
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    global_step = 0
    best_val_loss = float("inf")
    total_train_steps = max(1, training_config.num_epochs * max_train_batches)

    try:
        for epoch in range(training_config.num_epochs):
            print(f"Epoch {epoch + 1}/{training_config.num_epochs}")
            epoch_start = time.perf_counter()

            train_progress = global_step / total_train_steps
            current_momentum = current_teacher_momentum(training_config, train_progress)
            current_temperature = current_teacher_temperature(training_config, epoch)
            model.temp_teacher = current_temperature

            train_metrics, train_batches = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                max_batches=max_train_batches,
                max_calo_hits=view_config.max_calo_hits,
                coord_noise_scale=view_config.coord_noise_scale,
                energy_jitter_scale=view_config.energy_jitter_scale,
                global_crop_ratio=view_config.global_crop_ratio,
                student_mask_fraction=view_config.student_mask_fraction,
                point_dropout=view_config.point_dropout,
                teacher_momentum=current_momentum,
                phase="train",
            )
            global_step += train_batches

            val_metrics, _ = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                lr_scheduler=None,
                max_batches=max_val_batches,
                max_calo_hits=view_config.max_calo_hits,
                coord_noise_scale=view_config.coord_noise_scale,
                energy_jitter_scale=view_config.energy_jitter_scale,
                global_crop_ratio=view_config.global_crop_ratio,
                student_mask_fraction=view_config.student_mask_fraction,
                point_dropout=view_config.point_dropout,
                teacher_momentum=current_momentum,
                phase="val",
            )

            epoch_time_seconds = time.perf_counter() - epoch_start
            epoch_metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_prototype_entropy": train_metrics["prototype_entropy"],
                "val_prototype_entropy": val_metrics["prototype_entropy"],
                "train_embedding_norm": train_metrics["embedding_norm"],
                "val_embedding_norm": val_metrics["embedding_norm"],
                "train_masked_fraction": train_metrics["masked_fraction"],
                "val_masked_fraction": val_metrics["masked_fraction"],
                "train_data_wait_seconds": train_metrics["data_wait_seconds"],
                "val_data_wait_seconds": val_metrics["data_wait_seconds"],
                "train_view_build_seconds": train_metrics["view_build_seconds"],
                "val_view_build_seconds": val_metrics["view_build_seconds"],
                "train_model_step_seconds": train_metrics["model_step_seconds"],
                "val_model_step_seconds": val_metrics["model_step_seconds"],
                "train_events_per_second": train_metrics["events_per_second"],
                "val_events_per_second": val_metrics["events_per_second"],
                "learning_rate": learning_rate(optimizer),
                "teacher_momentum": current_momentum,
                "teacher_temperature": current_temperature,
                "epoch_time_seconds": epoch_time_seconds,
                "center_norm": center_norm(model),
            }
            logger.log_metrics(epoch_metrics, step=global_step)
            print("epoch summary: " + json.dumps(epoch_metrics, sort_keys=True))

            is_best = epoch_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = epoch_metrics["val_loss"]
            if (epoch + 1) % training_config.checkpoint_every_epochs == 0 or is_best:
                checkpoint_path = save_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    metrics=epoch_metrics,
                    is_best=is_best,
                )
                print(f"Saved checkpoint: {checkpoint_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
