from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import comet_ml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
from collider_fm.model import PandaSelfDistillation, create_small_panda_model
from collider_fm.views import build_distillation_views


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the simplified ColliderFM self-distillation model.")
    parser.add_argument("--train-split", default="train[:8]")
    parser.add_argument("--val-split", default="train[8:10]")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-momentum", type=float, default=0.99)
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--max-tracker-hits", type=int, default=128)
    parser.add_argument("--max-calo-hits", type=int, default=256)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument("--log-backend", choices=["none", "jsonl", "auto", "comet"], default="auto")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)
    return parser


def create_dataloader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = ColliderMLDataset(
        split=split,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        object_types=["tracker_hits", "calo_hits"],
        cache_dir=args.cache_dir,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


def learning_rate(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def center_norm(model: PandaSelfDistillation) -> float:
    return float(model.center.norm().item())


def run_epoch(
    model: PandaSelfDistillation,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    max_batches: int,
    teacher_momentum: float,
    max_tracker_hits: int,
    max_calo_hits: int,
    phase: str,
) -> tuple[float, int]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    num_batches = 0

    for batch_index, events in enumerate(dataloader):
        if batch_index >= max_batches:
            break

        views = build_distillation_views(
            events,
            device=device,
            max_tracker_hits=max_tracker_hits,
            max_calo_hits=max_calo_hits,
        )

        with torch.set_grad_enabled(is_training):
            student_outputs, teacher_outputs = model(views)
            loss = model.distillation_loss(student_outputs, teacher_outputs)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_center(teacher_outputs)
            model.update_teacher(momentum=teacher_momentum)

        total_loss += loss.item()
        num_batches += 1
        print(f"{phase} batch {batch_index + 1}: loss={loss.item():.4f}")

    if num_batches == 0:
        raise ValueError(f"No {phase} batches were processed. Check the chosen split and max batch count.")
    return total_loss / num_batches, num_batches


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print("Training requires a CUDA-enabled environment because the current PTv3/spconv stack is GPU-only.")
        print("Run this script on a GPU node, for example through a SLURM job.")
        return

    train_loader = create_dataloader(args, args.train_split)
    val_loader = create_dataloader(args, args.val_split)

    run_dir, run_name = ensure_run_directory(PROJECT_ROOT, run_dir=args.run_dir, run_name=args.run_name)
    logger = create_experiment_logger(args.log_backend, run_dir=run_dir, run_name=run_name)
    config = vars(args) | {
        "device": str(device),
        "run_dir": str(run_dir),
        "run_name": run_name,
    }
    config_path = write_run_config(run_dir, config)
    logger.log_params(
        {
            "run_name": run_name,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "teacher_momentum": args.teacher_momentum,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "max_tracker_hits": args.max_tracker_hits,
            "max_calo_hits": args.max_calo_hits,
            "dataset_type": args.dataset_type,
            "pu_config": args.pu_config,
            "log_backend": args.log_backend,
        }
    )
    print(f"Run directory: {run_dir}")
    print(f"Run config: {config_path}")

    model = create_small_panda_model(device=device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    global_step = 0

    try:
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            epoch_start = time.perf_counter()
            train_loss, train_batches = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
                teacher_momentum=args.teacher_momentum,
                max_tracker_hits=args.max_tracker_hits,
                max_calo_hits=args.max_calo_hits,
                phase="train",
            )
            global_step += train_batches
            val_loss, _ = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                max_batches=args.max_val_batches,
                teacher_momentum=args.teacher_momentum,
                max_tracker_hits=args.max_tracker_hits,
                max_calo_hits=args.max_calo_hits,
                phase="val",
            )
            epoch_time_seconds = time.perf_counter() - epoch_start
            epoch_metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate(optimizer),
                "epoch_time_seconds": epoch_time_seconds,
                "center_norm": center_norm(model),
            }
            logger.log_metrics(epoch_metrics, step=global_step)
            print("epoch summary: " + json.dumps(epoch_metrics, sort_keys=True))
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
