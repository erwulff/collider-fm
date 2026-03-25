from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.diagnostics import encode_view, load_checkpoint
from collider_fm.model import create_small_panda_model
from collider_fm.views import build_point_view_from_event


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export frozen embeddings for calo-only ColliderML events."
    )
    parser.add_argument("--split", default="train[:10]")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-calo-hits", type=int, default=256)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    return parser


def default_output_path() -> Path:
    return PROJECT_ROOT / "runs" / "exported_embeddings.pt"


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "Embedding export requires CUDA because the current PTv3/spconv path is GPU-only."
        )
        return

    dataset = ColliderMLDataset(
        split=args.split,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        object_types=["calo_hits"],
        cache_dir=args.cache_dir,
    )

    model = create_small_panda_model(device=device)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)
    model.eval()

    exported = []
    for event_index in range(len(dataset)):
        event = dataset[event_index]
        view = build_point_view_from_event(
            event, device=device, max_calo_hits=args.max_calo_hits
        )
        encoding = encode_view(model, view)
        exported.append(
            {
                "event_index": event_index,
                "coord": view["coord"].detach().cpu(),
                "energy": view["energy"].detach().cpu(),
                "detector_id": view["detector_id"].detach().cpu(),
                "calo_type": view["calo_type"].detach().cpu(),
                "point_features": encoding["point_features"].detach().cpu(),
                "pooled_embedding": encoding["pooled"].detach().cpu(),
            }
        )

    output_path = (
        Path(args.output) if args.output is not None else default_output_path()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"split": args.split, "checkpoint": args.checkpoint, "events": exported},
        output_path,
    )
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
