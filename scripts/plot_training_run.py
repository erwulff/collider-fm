from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.diagnostics import encode_view, load_checkpoint
from collider_fm.model import create_small_panda_model
from collider_fm.views import augment_point_view, build_point_view_from_event


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one SSL training run and a few held-out validation views."
    )
    parser.add_argument("run_dir", help="Path to one run directory under runs/.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="train[240:280]")
    parser.add_argument("--max-events", type=int, default=16)
    parser.add_argument("--max-calo-hits", type=int, default=256)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument("--device", default="cuda")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def keep_last_run(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    runs: list[list[dict[str, Any]]] = [[]]
    previous_step = -1
    for record in records:
        current_step = int(record.get("global_step", 0))
        if current_step <= previous_step:
            runs.append([])
        runs[-1].append(record)
        previous_step = current_step
    return runs[-1]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_losses(metrics: list[dict[str, Any]], output_path: Path) -> None:
    steps = [record["global_step"] for record in metrics]
    train_loss = [record["train_loss"] for record in metrics]
    val_loss = [record["val_loss"] for record in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, train_loss, marker="o", label="train loss")
    ax.plot(steps, val_loss, marker="o", label="val loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    save_figure(fig, output_path)


def plot_center_norm(metrics: list[dict[str, Any]], output_path: Path) -> None:
    steps = [record["global_step"] for record in metrics]
    norms = [record["center_norm"] for record in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, norms, marker="o", color="tab:purple")
    ax.set_xlabel("global step")
    ax.set_ylabel("center norm")
    ax.set_title("Teacher-center norm over training")
    save_figure(fig, output_path)


def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item())


@torch.no_grad()
def collect_similarity_scores(
    model: Any,
    dataset: ColliderMLDataset,
    device: torch.device,
    max_events: int,
    max_calo_hits: int,
) -> tuple[list[float], list[float]]:
    pooled_embeddings = []
    same_event_scores = []

    for event_index in range(min(len(dataset), max_events)):
        event = dataset[event_index]
        base_view = build_point_view_from_event(
            event, device=device, max_calo_hits=max_calo_hits
        )
        aug_a = augment_point_view(base_view)
        aug_b = augment_point_view(base_view)
        pooled_a = encode_view(model, aug_a)["pooled"][0]
        pooled_b = encode_view(model, aug_b)["pooled"][0]
        pooled_embeddings.append(pooled_a)
        same_event_scores.append(cosine_similarity(pooled_a, pooled_b))

    different_event_scores = []
    for index in range(len(pooled_embeddings) - 1):
        different_event_scores.append(
            cosine_similarity(pooled_embeddings[index], pooled_embeddings[index + 1])
        )

    return same_event_scores, different_event_scores


def plot_similarity_histogram(
    same_event_scores: list[float],
    different_event_scores: list[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(same_event_scores, bins=15, alpha=0.7, label="same event, two views")
    ax.hist(different_event_scores, bins=15, alpha=0.7, label="different events")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title("Held-out SSL similarity check")
    ax.legend()
    save_figure(fig, output_path)


def write_summary(
    path: Path,
    metrics: list[dict[str, Any]],
    same_event_scores: list[float],
    different_event_scores: list[float],
    config: dict[str, Any],
) -> None:
    final_metrics = metrics[-1]
    summary = {
        "run_name": config.get("run_name"),
        "num_epochs": len(metrics),
        "final_train_loss": final_metrics["train_loss"],
        "final_val_loss": final_metrics["val_loss"],
        "final_center_norm": final_metrics["center_norm"],
        "mean_same_event_similarity": float(np.mean(same_event_scores))
        if same_event_scores
        else None,
        "mean_different_event_similarity": float(np.mean(different_event_scores))
        if different_event_scores
        else None,
    }
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_output_dir(run_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return ensure_output_dir(Path(output_dir))
    return ensure_output_dir(run_dir / "plots")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.pt"

    metrics = keep_last_run(read_metrics(metrics_path))
    config = read_json(config_path)
    output_dir = resolve_output_dir(run_dir, args.output_dir)

    plot_losses(metrics, output_dir / "loss_curve.png")
    plot_center_norm(metrics, output_dir / "center_norm.png")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        print("CUDA is unavailable, so held-out similarity plots are skipped.")
        write_summary(output_dir / "summary.json", metrics, [], [], config)
        print(f"Saved training plots to {output_dir}")
        return

    dataset = ColliderMLDataset(
        split=args.split,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        object_types=["calo_hits"],
        cache_dir=args.cache_dir,
    )
    model = create_small_panda_model(device=device)
    load_checkpoint(model, str(checkpoint_path))
    model.eval()

    same_event_scores, different_event_scores = collect_similarity_scores(
        model=model,
        dataset=dataset,
        device=device,
        max_events=args.max_events,
        max_calo_hits=args.max_calo_hits,
    )
    plot_similarity_histogram(
        same_event_scores,
        different_event_scores,
        output_dir / "heldout_similarity.png",
    )
    write_summary(
        output_dir / "summary.json",
        metrics,
        same_event_scores,
        different_event_scores,
        config,
    )
    print(f"Saved training plots to {output_dir}")


if __name__ == "__main__":
    main()
