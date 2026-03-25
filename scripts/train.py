from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import comet_ml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
from collider_fm.views import build_distillation_views


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the beginner-friendly ColliderFM self-distillation model."
    )
    parser.add_argument("--train-split", default="train[:64]")
    parser.add_argument("--val-split", default="train[64:80]")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-momentum-start", type=float, default=0.99)
    parser.add_argument("--teacher-momentum-end", type=float, default=0.999)
    parser.add_argument("--teacher-temperature-start", type=float, default=0.04)
    parser.add_argument("--teacher-temperature-end", type=float, default=0.07)
    parser.add_argument("--teacher-temperature-warmup-epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument("--max-calo-hits", type=int, default=512)
    parser.add_argument("--coord-noise-scale", type=float, default=0.5)
    parser.add_argument("--energy-jitter-scale", type=float, default=0.01)
    parser.add_argument("--global-crop-ratio", type=float, default=0.9)
    parser.add_argument("--student-mask-fraction", type=float, default=0.4)
    parser.add_argument("--point-dropout", type=float, default=0.05)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument(
        "--log-backend", choices=["none", "jsonl", "auto", "comet"], default="auto"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    return parser


def create_dataloader(
    args: argparse.Namespace, split: str, shuffle: bool
) -> DataLoader:
    """Create the calo-only dataloader used by train and validation."""

    dataset = ColliderMLDataset(
        split=split,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        object_types=["calo_hits"],
        cache_dir=args.cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def linear_warmup(value_start: float, value_end: float, progress: float) -> float:
    """Linearly interpolate between two values on `[0, 1]` progress."""

    progress = min(max(progress, 0.0), 1.0)
    return value_start + progress * (value_end - value_start)


def cosine_decay(value_start: float, value_end: float, progress: float) -> float:
    """Cosine decay schedule from `value_start` to `value_end`."""

    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return value_end + (value_start - value_end) * cosine


def learning_rate(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def center_norm(model: PandaSelfDistillation) -> float:
    return float(model.center.norm().item())


def current_teacher_temperature(args: argparse.Namespace, epoch_index: int) -> float:
    """Warm the teacher temperature during the early epochs only."""

    if args.teacher_temperature_warmup_epochs <= 0:
        return args.teacher_temperature_end
    if epoch_index >= args.teacher_temperature_warmup_epochs:
        return args.teacher_temperature_end
    progress = epoch_index / max(1, args.teacher_temperature_warmup_epochs - 1)
    return linear_warmup(
        args.teacher_temperature_start, args.teacher_temperature_end, progress
    )


def current_teacher_momentum(args: argparse.Namespace, global_progress: float) -> float:
    """Increase EMA momentum smoothly over training."""

    return cosine_decay(
        args.teacher_momentum_start, args.teacher_momentum_end, global_progress
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
    probabilities = probabilities.clamp_min(1.0e-8)
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.item())


def embedding_norm(outputs: list[dict[str, torch.Tensor]]) -> float:
    """Average norm of the masked pooled student embeddings."""

    embeddings = torch.cat(
        [output["masked_pooled_projection"] for output in outputs], dim=0
    )
    return float(embeddings.norm(dim=-1).mean().item())


def apply_learning_rate(optimizer: AdamW, lr_value: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_value


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
    }
    processed_batches = 0

    progress_bar = tqdm(
        dataloader,
        total=max_batches,
        desc=phase,
        leave=True,
        dynamic_ncols=False,
        ascii=True,
    )

    for batch_index, events in enumerate(progress_bar):
        if batch_index >= max_batches:
            break

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

        with torch.set_grad_enabled(is_training):
            student_outputs, teacher_outputs = model(distillation_batch)
            loss = model.distillation_loss(student_outputs, teacher_outputs)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_center(teacher_outputs)
            model.update_teacher(momentum=teacher_momentum)

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
        processed_batches += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            entropy=f"{prototype_entropy(usage):.3f}",
            masked=f"{masked_fraction:.3f}",
        )

    progress_bar.close()

    if processed_batches == 0:
        raise ValueError(
            f"No {phase} batches were processed. Check the chosen split and max batch count."
        )

    return {
        key: value / processed_batches for key, value in totals.items()
    }, processed_batches


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print(
            "Training requires a CUDA-enabled environment because the current PTv3/spconv stack is GPU-only."
        )
        print("Run this script on a GPU node, for example through a SLURM job.")
        return

    train_loader = create_dataloader(args, args.train_split, shuffle=True)
    val_loader = create_dataloader(args, args.val_split, shuffle=False)

    run_dir, run_name = ensure_run_directory(
        PROJECT_ROOT, run_dir=args.run_dir, run_name=args.run_name
    )
    logger = create_experiment_logger(
        args.log_backend, run_dir=run_dir, run_name=run_name
    )
    config = vars(args) | {
        "device": str(device),
        "run_dir": str(run_dir),
        "run_name": run_name,
    }
    config_path = write_run_config(run_dir, config)
    logger.log_params(config)
    print(f"Run directory: {run_dir}")
    print(f"Run config: {config_path}")

    model = create_training_panda_model(device=device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    global_step = 0
    best_val_loss = float("inf")
    total_train_steps = max(1, args.num_epochs * args.max_train_batches)

    try:
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            epoch_start = time.perf_counter()

            train_progress = global_step / total_train_steps
            current_lr = cosine_decay(
                args.learning_rate, args.min_learning_rate, train_progress
            )
            current_momentum = current_teacher_momentum(args, train_progress)
            current_temperature = current_teacher_temperature(args, epoch)
            model.temp_teacher = current_temperature
            apply_learning_rate(optimizer, current_lr)

            train_metrics, train_batches = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
                max_calo_hits=args.max_calo_hits,
                coord_noise_scale=args.coord_noise_scale,
                energy_jitter_scale=args.energy_jitter_scale,
                global_crop_ratio=args.global_crop_ratio,
                student_mask_fraction=args.student_mask_fraction,
                point_dropout=args.point_dropout,
                teacher_momentum=current_momentum,
                phase="train",
            )
            global_step += train_batches

            val_metrics, _ = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                max_batches=args.max_val_batches,
                max_calo_hits=args.max_calo_hits,
                coord_noise_scale=args.coord_noise_scale,
                energy_jitter_scale=args.energy_jitter_scale,
                global_crop_ratio=args.global_crop_ratio,
                student_mask_fraction=args.student_mask_fraction,
                point_dropout=args.point_dropout,
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
            if (epoch + 1) % args.checkpoint_every_epochs == 0 or is_best:
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
