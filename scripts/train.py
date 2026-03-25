from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import comet_ml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(
        description="Train the small calo-only ColliderFM model."
    )
    parser.add_argument("--train-split", default="train[:8]")
    parser.add_argument("--val-split", default="train[8:10]")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-momentum", type=float, default=0.99)
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--max-calo-hits", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--add-local-view", action="store_true")
    parser.add_argument("--local-fraction", type=float, default=0.5)
    parser.add_argument("--add-masked-view", action="store_true")
    parser.add_argument("--mask-fraction", type=float, default=0.3)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument(
        "--log-backend", choices=["none", "jsonl", "auto", "comet"], default="auto"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", default=None)
    return parser


def create_dataloader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = ColliderMLDataset(
        split=split,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        object_types=["calo_hits"],
        cache_dir=args.cache_dir,
    )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )


def learning_rate(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def log(message: str) -> None:
    print(message, flush=True)


def center_norm(model: PandaSelfDistillation) -> float:
    return float(cast(torch.Tensor, model.center).norm().item())


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pt"


def target_num_batches(dataloader: DataLoader, max_batches: int) -> int:
    return min(len(dataloader), max_batches)


def create_progress_bar(phase: str, num_batches: int, log_every: int) -> Any:
    return create_progress_bar_for_stream(
        phase=phase,
        num_batches=num_batches,
        log_every=log_every,
        stream=sys.stdout,
    )


def create_progress_bar_for_stream(
    phase: str,
    num_batches: int,
    log_every: int,
    stream: Any,
) -> Any:
    return tqdm(
        total=num_batches,
        desc=phase,
        unit="batch",
        file=stream,
        ascii=True,
        dynamic_ncols=False,
        leave=True,
        miniters=max(1, log_every),
    )


def save_checkpoint(
    run_dir: Path,
    model: PandaSelfDistillation,
    optimizer: AdamW,
    epoch: int,
    global_step: int,
) -> Path:
    path = checkpoint_path(run_dir)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str, model: PandaSelfDistillation, optimizer: AdamW
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def run_epoch(
    model: PandaSelfDistillation,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    max_batches: int,
    teacher_momentum: float,
    max_calo_hits: int,
    add_local_view: bool,
    local_fraction: float,
    add_masked_view: bool,
    mask_fraction: float,
    log_every: int,
    phase: str,
) -> tuple[float, int]:
    """Run one short training or validation pass over a dataloader."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    num_batches = 0
    planned_batches = target_num_batches(dataloader, max_batches)
    progress = create_progress_bar(phase, planned_batches, log_every)

    try:
        for batch_index, events in enumerate(dataloader):
            if batch_index >= max_batches:
                break

            views = build_distillation_views(
                events,
                device=device,
                max_calo_hits=max_calo_hits,
                add_local_view=add_local_view,
                local_fraction=local_fraction,
                add_masked_view=add_masked_view,
                mask_fraction=mask_fraction,
            )
            loss_masks = [view["loss_mask"] for view in views]
            with torch.set_grad_enabled(is_training):
                student_outputs, teacher_outputs = model(views)
                loss = model.distillation_loss(
                    student_outputs, teacher_outputs, loss_masks=loss_masks
                )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.update_center(teacher_outputs)
                model.update_teacher(momentum=teacher_momentum)

            total_loss += float(loss.item())
            num_batches += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    finally:
        progress.close()

    if num_batches == 0:
        raise ValueError(f"No {phase} batches were processed.")
    return total_loss / num_batches, num_batches


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    if device.type != "cuda":
        log("Training requires CUDA because the current PTv3/spconv stack is GPU-only.")
        return

    train_loader = create_dataloader(args, args.train_split)
    val_loader = create_dataloader(args, args.val_split)

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
    log(f"Run directory: {run_dir}")
    log(f"Run config: {config_path}")

    model = create_small_panda_model(device=device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    global_step = 0

    if args.resume_from is not None:
        start_epoch, global_step = load_checkpoint(args.resume_from, model, optimizer)
        log(
            f"Resumed from {args.resume_from} at epoch {start_epoch}, step {global_step}"
        )

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
            "max_calo_hits": args.max_calo_hits,
            "log_every": args.log_every,
            "add_local_view": args.add_local_view,
            "local_fraction": args.local_fraction,
            "add_masked_view": args.add_masked_view,
            "mask_fraction": args.mask_fraction,
            "dataset_type": args.dataset_type,
            "pu_config": args.pu_config,
            "resume_from": args.resume_from,
        }
    )

    try:
        for epoch in range(start_epoch, args.num_epochs):
            log(f"Epoch {epoch + 1}/{args.num_epochs}")
            epoch_start = time.perf_counter()
            train_loss, train_batches = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
                teacher_momentum=args.teacher_momentum,
                max_calo_hits=args.max_calo_hits,
                add_local_view=args.add_local_view,
                local_fraction=args.local_fraction,
                add_masked_view=args.add_masked_view,
                mask_fraction=args.mask_fraction,
                log_every=args.log_every,
                phase="train",
            )
            global_step += train_batches
            val_loss, val_batches = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                max_batches=args.max_val_batches,
                teacher_momentum=args.teacher_momentum,
                max_calo_hits=args.max_calo_hits,
                add_local_view=args.add_local_view,
                local_fraction=args.local_fraction,
                add_masked_view=args.add_masked_view,
                mask_fraction=args.mask_fraction,
                log_every=args.log_every,
                phase="val",
            )
            checkpoint = save_checkpoint(
                run_dir, model, optimizer, epoch + 1, global_step
            )
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_batches": train_batches,
                "val_loss": val_loss,
                "val_batches": val_batches,
                "learning_rate": learning_rate(optimizer),
                "epoch_time_seconds": time.perf_counter() - epoch_start,
                "center_norm": center_norm(model),
                "checkpoint": str(checkpoint),
            }
            logger.log_metrics(metrics, step=global_step)
            log("epoch summary: " + json.dumps(metrics, sort_keys=True))
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
