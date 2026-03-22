from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset, collate_fn
from collider_fm.model import PandaSelfDistillation
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


def create_model(device: torch.device) -> PandaSelfDistillation:
    model = PandaSelfDistillation(
        in_channels=6,
        embed_channels=8,
        num_prototypes=32,
        projection_dim=8,
        prediction_dim=16,
        backbone_kwargs={
            "enc_depths": (1, 1, 1, 1, 1),
            "enc_channels": (8, 12, 16, 24, 32),
            "enc_num_head": (1, 1, 2, 4, 4),
            "enc_patch_size": (4, 4, 4, 4, 4),
            "shuffle_orders": False,
            "enable_flash": False,
        },
    ).to(device)
    return model


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
) -> float:
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
    return total_loss / num_batches


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

    model = create_model(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = run_epoch(
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
        val_loss = run_epoch(
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
        print(f"epoch {epoch + 1} summary: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
