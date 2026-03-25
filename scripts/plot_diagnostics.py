from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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

from collider_fm.diagnostics import (
    compute_pca,
    encode_view,
    load_checkpoint,
    load_events,
    tensor_summary,
    to_numpy,
)
from collider_fm.model import create_small_panda_model
from collider_fm.views import (
    CALO_TYPE_NAMES,
    augment_point_view,
    batch_point_views,
    build_point_view_from_event,
    mask_point_view,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate simple saved diagnostics for the calo-only pipeline."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--detail-split", default="train[0:1]")
    parser.add_argument("--representation-split", default="train[:10]")
    parser.add_argument("--max-calo-hits", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument("--top-k-prototypes", type=int, default=8)
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "diagnostics" / f"diagnostics_{timestamp}"


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "raw": root / "raw",
        "views": root / "views",
        "model": root / "model",
        "artifacts": root / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Model-backed plots will be skipped.")
        return torch.device("cpu")
    return torch.device(name)


def save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_raw_event(event: dict[str, Any], path: Path) -> None:
    calo_hits = event["calo_hits"]
    x = np.asarray(torch.as_tensor(calo_hits["x"]).tolist(), dtype=float)
    y = np.asarray(torch.as_tensor(calo_hits["y"]).tolist(), dtype=float)
    z = np.asarray(torch.as_tensor(calo_hits["z"]).tolist(), dtype=float)
    energy = np.asarray(torch.as_tensor(calo_hits["energy"]).tolist(), dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = getattr(ax, "scatter")(z, x, y, c=energy, s=4, alpha=0.6, cmap="inferno")
    fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1, label="energy")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title("Raw calorimeter event")
    save_figure(fig, path)


def plot_views(
    base_view: dict[str, torch.Tensor],
    aug_a: dict[str, torch.Tensor],
    aug_b: dict[str, torch.Tensor],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, label, view in zip(
        axes, ["base", "aug A", "aug B"], [base_view, aug_a, aug_b], strict=True
    ):
        coord = to_numpy(view["coord"])
        energy = to_numpy(view["energy"])
        axis.scatter(coord[:, 2], coord[:, 0], c=energy, s=4, alpha=0.6, cmap="inferno")
        axis.set_title(label)
        axis.set_xlabel("z")
        axis.set_ylabel("x")
    fig.suptitle("Base and augmented calorimeter views")
    save_figure(fig, path)


def plot_calo_type_counts(view: dict[str, torch.Tensor], path: Path) -> None:
    calo_type = view["calo_type"]
    counts = [
        int((calo_type == index).sum().item()) for index in sorted(CALO_TYPE_NAMES)
    ]
    labels = [CALO_TYPE_NAMES[index] for index in sorted(CALO_TYPE_NAMES)]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("points")
    ax.set_title("Calo type counts")
    save_figure(fig, path)


def plot_mask_counts(view: dict[str, torch.Tensor], path: Path) -> None:
    mask = view["mask"]
    counts = [int((~mask).sum().item()), int(mask.sum().item())]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["visible", "masked"], counts, color=["tab:blue", "tab:red"])
    ax.set_ylabel("points")
    ax.set_title("Mask counts")
    save_figure(fig, path)


def plot_prototype_usage(
    point_logits: torch.Tensor, path: Path, top_k: int
) -> dict[str, float]:
    assignments = point_logits.argmax(dim=-1)
    usage = torch.bincount(assignments, minlength=point_logits.shape[1]).to(
        torch.float32
    )
    top_values, top_indices = torch.topk(usage, k=min(top_k, usage.numel()))

    probabilities = F.softmax(point_logits, dim=-1)
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1))
        .mean()
        .item()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [str(int(index)) for index in top_indices],
        to_numpy(top_values),
        color="tab:green",
    )
    ax.set_xlabel("prototype")
    ax.set_ylabel("assigned points")
    ax.set_title("Most-used prototypes")
    save_figure(fig, path)

    return {
        "mean_entropy": entropy,
        "num_active_prototypes": float((usage > 0).sum().item()),
    }


def plot_point_feature_norms(point_features: torch.Tensor, path: Path) -> None:
    norms = torch.linalg.norm(point_features, dim=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(to_numpy(norms), bins=40, color="tab:purple")
    ax.set_xlabel("feature norm")
    ax.set_ylabel("points")
    ax.set_title("Backbone point-feature norms")
    save_figure(fig, path)


def plot_embedding_pca(pooled_embeddings: torch.Tensor, path: Path) -> None:
    projected = compute_pca(to_numpy(pooled_embeddings), n_components=2)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=np.arange(projected.shape[0]),
        cmap="tab10",
        s=50,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Pooled event embedding PCA")
    save_figure(fig, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    output_root = (
        Path(args.output_dir) if args.output_dir is not None else default_output_dir()
    )
    output_dirs = ensure_output_dirs(output_root)

    device = resolve_device(args.device)
    detail_events = load_events(
        args.detail_split,
        batch_size=1,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        cache_dir=args.cache_dir,
    )
    representation_events = load_events(
        args.representation_split,
        batch_size=32,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        cache_dir=args.cache_dir,
    )
    detail_event = detail_events[0]

    base_view = build_point_view_from_event(
        detail_event, device=device, max_calo_hits=args.max_calo_hits
    )
    aug_a = augment_point_view(base_view)
    aug_b = augment_point_view(base_view)
    masked_view = mask_point_view(augment_point_view(base_view))
    representation_views = [
        build_point_view_from_event(
            event, device=device, max_calo_hits=args.max_calo_hits
        )
        for event in representation_events
    ]

    plot_raw_event(detail_event, output_dirs["raw"] / "event_000_raw.png")
    plot_views(base_view, aug_a, aug_b, output_dirs["views"] / "event_000_views.png")
    plot_calo_type_counts(base_view, output_dirs["views"] / "event_000_calo_types.png")
    plot_mask_counts(masked_view, output_dirs["views"] / "event_000_mask_counts.png")

    summary: dict[str, Any] = {
        "device": str(device),
        "detail_event": {
            "num_points": int(base_view["coord"].shape[0]),
            "point_view": tensor_summary(base_view["feat"]),
        },
        "representation_sample": {"num_events": len(representation_views)},
        "checkpoint": args.checkpoint,
    }

    if device.type == "cuda":
        model = create_small_panda_model(device=device)
        if args.checkpoint is not None:
            summary["checkpoint_load"] = load_checkpoint(model, args.checkpoint)
        model.eval()

        student_outputs, teacher_outputs = model([base_view, aug_a, masked_view])
        prototype_summary = plot_prototype_usage(
            student_outputs[0],
            output_dirs["model"] / "prototype_usage.png",
            args.top_k_prototypes,
        )

        detail_encoding = encode_view(model, base_view)
        plot_point_feature_norms(
            detail_encoding["point_features"],
            output_dirs["model"] / "point_feature_norms.png",
        )

        representation_batch = batch_point_views(representation_views)
        representation_encoding = encode_view(model, representation_batch)
        plot_embedding_pca(
            representation_encoding["pooled"],
            output_dirs["model"] / "embedding_pca.png",
        )

        summary["model"] = {
            "student_logits": tensor_summary(student_outputs[0]),
            "teacher_logits": tensor_summary(teacher_outputs[0]),
            "masked_view": tensor_summary(masked_view["feat"]),
            "point_features": tensor_summary(detail_encoding["point_features"]),
            "pooled_embeddings": tensor_summary(representation_encoding["pooled"]),
            "prototype_usage": prototype_summary,
        }
    else:
        print("Skipping model-backed plots because CUDA is unavailable.")

    write_json(output_dirs["artifacts"] / "summary.json", summary)
    print(f"Saved diagnostics to {output_root}")


if __name__ == "__main__":
    main()
