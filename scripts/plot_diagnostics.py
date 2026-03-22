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
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset, collate_fn
from collider_fm.model import PandaSelfDistillation, as_point_cloud, mean_pool_features
from collider_fm.views import augment_point_view, batch_point_views, build_point_view_from_event


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostic plots for the ColliderFM data-to-view-to-representation pipeline. "
            "By default the script traces one detailed event and separately summarizes a 10-event sample for PCA plots."
        )
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--detail-split", default="train[0:1]")
    parser.add_argument("--representation-split", default="train[:10]")
    parser.add_argument("--max-tracker-hits", type=int, default=128)
    parser.add_argument("--max-calo-hits", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-type", default="ttbar")
    parser.add_argument("--pu-config", default="pu0")
    parser.add_argument("--cache-dir", default="/mnt/ceph/users/ewulff/data/hf")
    parser.add_argument("--point-feature-sample-size", type=int, default=2000)
    parser.add_argument("--top-k-prototypes", type=int, default=8)
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "diagnostics" / f"diagnostics_{timestamp}"


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    subdirs = {
        "root": root,
        "raw": root / "raw",
        "views": root / "views",
        "model": root / "model",
        "artifacts": root / "artifacts",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU for raw/view diagnostics only.")
        return torch.device("cpu")
    return torch.device(device_name)


def create_model(device: torch.device) -> PandaSelfDistillation:
    return PandaSelfDistillation(
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


def load_checkpoint(model: PandaSelfDistillation, checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint_path": checkpoint_path,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def create_dataloader(args: argparse.Namespace, split: str, batch_size: int) -> DataLoader:
    dataset = ColliderMLDataset(
        split=split,
        dataset_type=args.dataset_type,
        pu_config=args.pu_config,
        object_types=["tracker_hits", "calo_hits"],
        cache_dir=args.cache_dir,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def load_events(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    dataloader = create_dataloader(args, split=split, batch_size=64)
    return next(iter(dataloader))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def radius(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.linalg.norm(torch.stack([values["x"], values["y"], values["z"]], dim=1), dim=1)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu()
    return {
        "shape": list(array.shape),
        "min": float(array.min().item()),
        "max": float(array.max().item()),
        "mean": float(array.mean().item()),
        "std": float(array.std().item()) if array.numel() > 1 else 0.0,
    }


def compute_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    if features.shape[0] == 0:
        raise ValueError("PCA requires at least one sample.")

    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1 or centered.shape[1] == 1:
        result = np.zeros((centered.shape[0], n_components), dtype=np.float32)
        result[:, : min(n_components, centered.shape[1])] = centered[:, : min(n_components, centered.shape[1])]
        return result

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T


def sample_indices(num_items: int, max_items: int, seed: int) -> np.ndarray:
    if num_items <= max_items:
        return np.arange(num_items)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_items, size=max_items, replace=False))


@torch.no_grad()
def encode_view(
    model: PandaSelfDistillation,
    view: dict[str, torch.Tensor],
    use_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    point = as_point_cloud(view, default_grid_size=model.grid_size)
    backbone = model.teacher_backbone if use_teacher else model.student_backbone
    projector = model.teacher_projector if use_teacher else model.student_projector
    predictor = None if use_teacher else model.student_predictor

    encoded = backbone(point)
    pooled = mean_pool_features(encoded.feat, getattr(encoded, "offset", None))
    projection = projector(pooled)
    if predictor is not None:
        projection = predictor(projection)
    projection = F.normalize(projection, dim=-1)
    logits = model.prototype_head(projection)
    return {
        "point_features": encoded.feat,
        "pooled": pooled,
        "projection": projection,
        "logits": logits,
        "offset": point.offset,
    }


def plot_raw_geometry(event: dict[str, Any], path: Path) -> None:
    tracker_hits = event["tracker_hits"]
    calo_hits = event["calo_hits"]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        to_numpy(tracker_hits["z"]),
        to_numpy(tracker_hits["x"]),
        to_numpy(tracker_hits["y"]),
        s=2,
        alpha=0.45,
        label="tracker",
        color="tab:blue",
    )
    energy = to_numpy(calo_hits.get("total_energy", torch.zeros_like(calo_hits["x"], dtype=torch.float32)))
    positive_energy = energy[energy > 0]
    if positive_energy.size > 0:
        color_vmin = float(positive_energy.min())
        color_vmax = float(np.quantile(positive_energy, 0.99))
        if color_vmax <= color_vmin:
            color_vmax = float(positive_energy.max())
        if color_vmax <= color_vmin:
            color_vmax = color_vmin * 1.01
        marker_size = np.clip(np.log10(energy / color_vmin + 1.0) * 10.0, 4.0, 60.0)
        norm = LogNorm(vmin=color_vmin, vmax=color_vmax)
    else:
        marker_size = np.full_like(energy, 4.0)
        norm = None
    calo = ax.scatter(
        to_numpy(calo_hits["z"]),
        to_numpy(calo_hits["x"]),
        to_numpy(calo_hits["y"]),
        s=marker_size,
        alpha=0.55,
        c=energy,
        cmap="inferno",
        norm=norm,
        label="calo",
    )
    colorbar = fig.colorbar(calo, ax=ax, shrink=0.7, pad=0.1)
    colorbar.set_label("calo energy (log scale)")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title("Raw dataloader event geometry")
    ax.legend(loc="upper right")
    save_figure(fig, path)


def plot_raw_scalars(event: dict[str, Any], path: Path) -> None:
    tracker_hits = event["tracker_hits"]
    calo_hits = event["calo_hits"]
    tracker_radius = radius(tracker_hits)
    calo_radius = radius(calo_hits)
    tracker_time = tracker_hits.get("time", torch.zeros(1))
    calo_energy = calo_hits.get("total_energy", torch.zeros(1))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    axes[0].hist(to_numpy(tracker_hits.get("time", torch.zeros(len(tracker_hits["x"])))), bins=40, color="tab:blue")
    axes[0].set_title("Tracker time")
    axes[1].hist(
        to_numpy(calo_hits.get("total_energy", torch.zeros(len(calo_hits["x"])))),
        bins=40,
        color="tab:red",
    )
    axes[1].set_title("Calo energy")
    axes[2].hist(to_numpy(tracker_radius), bins=40, color="tab:cyan")
    axes[2].set_title("Tracker radius")
    axes[3].hist(to_numpy(calo_radius), bins=40, color="tab:orange")
    axes[3].set_title("Calo radius")
    axes[4].axis("off")
    axes[4].text(
        0.0,
        0.8,
        "\n".join(
            [
                f"tracker hits: {len(tracker_hits['x'])}",
                f"calo hits: {len(calo_hits['x'])}",
                (f"tracker time range: [{tracker_time.min().item():.2f}, " f"{tracker_time.max().item():.2f}]"),
                (f"calo energy range: [{calo_energy.min().item():.2f}, " f"{calo_energy.max().item():.2f}]"),
            ]
        ),
        fontsize=11,
        va="top",
    )
    axes[5].axis("off")
    fig.suptitle("Raw dataloader scalar summaries")
    save_figure(fig, path)


def plot_view_detector_type(view: dict[str, torch.Tensor], path: Path) -> None:
    coord = to_numpy(view["coord"])
    detector_type = to_numpy(view["feat"][:, 5])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    tracker_mask = detector_type == 0
    calo_mask = detector_type == 1
    ax.scatter(
        coord[tracker_mask, 2],
        coord[tracker_mask, 0],
        coord[tracker_mask, 1],
        color="tab:blue",
        s=4,
        alpha=0.7,
        label="tracker (0)",
    )
    ax.scatter(
        coord[calo_mask, 2],
        coord[calo_mask, 0],
        coord[calo_mask, 1],
        color="tab:red",
        s=4,
        alpha=0.7,
        label="calo (1)",
    )
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("Model input point cloud colored by detector type")
    ax.legend(loc="upper right")
    save_figure(fig, path)


def plot_view_signal(view: dict[str, torch.Tensor], path: Path) -> None:
    coord = to_numpy(view["coord"])
    signal = to_numpy(view["feat"][:, 4])
    features = to_numpy(view["feat"])
    detector_type = features[:, 5].astype(int)
    detector_counts = np.bincount(detector_type, minlength=2)

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(3, 2)
    ax_scatter = fig.add_subplot(grid[:, 0], projection="3d")
    scatter = ax_scatter.scatter(coord[:, 2], coord[:, 0], coord[:, 1], c=signal, cmap="viridis", s=4, alpha=0.7)
    fig.colorbar(scatter, ax=ax_scatter, shrink=0.7, pad=0.1, label="signal channel")
    ax_scatter.set_xlabel("z")
    ax_scatter.set_ylabel("x")
    ax_scatter.set_zlabel("y")
    ax_scatter.set_title("Model input colored by signal")

    axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 1])]
    axes[0].hist(features[:, 0], bins=40, alpha=0.8, label="x")
    axes[0].hist(features[:, 1], bins=40, alpha=0.6, label="y")
    axes[0].hist(features[:, 2], bins=40, alpha=0.5, label="z")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Coordinate feature histograms")
    axes[1].hist(features[:, 3], bins=40, alpha=0.8, label="radius")
    axes[1].hist(features[:, 4], bins=40, alpha=0.6, label="signal")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Continuous derived feature histograms")
    axes[2].bar([0, 1], detector_counts, color=["tab:blue", "tab:red"], tick_label=["tracker (0)", "calo (1)"])
    axes[2].set_title("Detector type counts")
    axes[2].set_ylabel("points")
    fig.suptitle("Model input feature summary")
    save_figure(fig, path)


def plot_augmentations(base_view: dict[str, torch.Tensor], aug_a: dict[str, torch.Tensor], aug_b: dict[str, torch.Tensor], path: Path) -> None:
    views = [("base", base_view), ("aug A", aug_a), ("aug B", aug_b)]
    fig = plt.figure(figsize=(18, 6))
    for index, (label, view) in enumerate(views, start=1):
        coord = to_numpy(view["coord"])
        detector_type = to_numpy(view["feat"][:, 5])
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.scatter(coord[:, 2], coord[:, 0], coord[:, 1], c=detector_type, cmap="coolwarm", s=4, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
    fig.suptitle("Detailed event: base and augmented views")
    save_figure(fig, path)


def plot_augmentation_delta(base_view: dict[str, torch.Tensor], aug_a: dict[str, torch.Tensor], aug_b: dict[str, torch.Tensor], path: Path) -> None:
    base_coord = base_view["coord"]
    base_signal = base_view["feat"][:, 4]
    deltas = {
        "aug A": {
            "coord": torch.linalg.norm(aug_a["coord"] - base_coord, dim=1),
            "signal": aug_a["feat"][:, 4] - base_signal,
        },
        "aug B": {
            "coord": torch.linalg.norm(aug_b["coord"] - base_coord, dim=1),
            "signal": aug_b["feat"][:, 4] - base_signal,
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, values in deltas.items():
        axes[0].hist(to_numpy(values["coord"]), bins=40, alpha=0.6, label=label)
        axes[1].hist(to_numpy(values["signal"]), bins=40, alpha=0.6, label=label)
    axes[0].set_title("Coordinate displacement magnitude")
    axes[1].set_title("Signal-channel perturbation")
    for axis in axes:
        axis.legend()
    fig.suptitle("Augmentation deltas relative to the base view")
    save_figure(fig, path)


def plot_batch_summary(views: list[dict[str, torch.Tensor]], path: Path) -> None:
    tracker_counts = []
    calo_counts = []
    total_counts = []
    for view in views:
        detector_type = view["feat"][:, 5]
        tracker_counts.append(int((detector_type == 0).sum().item()))
        calo_counts.append(int((detector_type == 1).sum().item()))
        total_counts.append(view["coord"].shape[0])

    indices = np.arange(len(views))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(indices, tracker_counts, label="tracker", color="tab:blue")
    axes[0].bar(indices, calo_counts, bottom=tracker_counts, label="calo", color="tab:red")
    axes[0].set_title("Subsampled point counts per event")
    axes[0].set_xlabel("event index")
    axes[0].set_ylabel("points")
    axes[0].legend()

    axes[1].axis("off")
    axes[1].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"events: {len(views)}",
                f"min points/event: {min(total_counts)}",
                f"max points/event: {max(total_counts)}",
                f"mean points/event: {np.mean(total_counts):.1f}",
            ]
        ),
        fontsize=12,
        va="top",
    )
    fig.suptitle("Representation-sample batch summary")
    save_figure(fig, path)


def plot_logits(student_probs: list[np.ndarray], teacher_probs: list[np.ndarray], path: Path, top_k: int) -> None:
    stacked = np.vstack(student_probs + teacher_probs)
    mean_scores = stacked.mean(axis=0)
    top_indices = np.argsort(mean_scores)[-top_k:]
    labels = [str(index) for index in top_indices]
    x = np.arange(len(top_indices))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    series = [
        ("student A", student_probs[0][top_indices]),
        ("student B", student_probs[1][top_indices]),
        ("teacher A", teacher_probs[0][top_indices]),
        ("teacher B", teacher_probs[1][top_indices]),
    ]
    for index, (label, values) in enumerate(series):
        ax.bar(x + (index - 1.5) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("prototype index")
    ax.set_ylabel("probability")
    ax.set_title("Detailed event: top prototype distributions")
    ax.legend()
    save_figure(fig, path)


def jensen_shannon_divergence(prob_a: torch.Tensor, prob_b: torch.Tensor) -> float:
    mean_prob = 0.5 * (prob_a + prob_b)
    js = 0.5 * F.kl_div(prob_a.log(), mean_prob, reduction="sum") + 0.5 * F.kl_div(prob_b.log(), mean_prob, reduction="sum")
    return float(js.item())


def plot_view_agreement(
    base_pooled: torch.Tensor,
    aug_a_pooled: torch.Tensor,
    aug_b_pooled: torch.Tensor,
    student_probs: list[torch.Tensor],
    teacher_probs: list[torch.Tensor],
    path: Path,
) -> None:
    cosine_values = {
        "base vs aug A": F.cosine_similarity(base_pooled, aug_a_pooled).item(),
        "base vs aug B": F.cosine_similarity(base_pooled, aug_b_pooled).item(),
        "aug A vs aug B": F.cosine_similarity(aug_a_pooled, aug_b_pooled).item(),
    }
    js_values = {
        "student": jensen_shannon_divergence(student_probs[0], student_probs[1]),
        "teacher": jensen_shannon_divergence(teacher_probs[0], teacher_probs[1]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(list(cosine_values.keys()), list(cosine_values.values()), color="tab:green")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Embedding cosine similarity")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(list(js_values.keys()), list(js_values.values()), color="tab:purple")
    axes[1].set_title("Jensen-Shannon divergence")
    fig.suptitle("Detailed event: view agreement")
    save_figure(fig, path)


def plot_embedding_pca(embeddings: torch.Tensor, event_labels: list[str], path: Path) -> None:
    projected = compute_pca(to_numpy(embeddings), n_components=2)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(projected[:, 0], projected[:, 1], c=np.arange(len(event_labels)), cmap="tab10", s=60)
    for index, label in enumerate(event_labels):
        ax.annotate(label, (projected[index, 0], projected[index, 1]), fontsize=8)
    ax.set_title("Pooled event embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_figure(fig, path)


def plot_point_feature_pca(
    point_features: torch.Tensor,
    detector_type: torch.Tensor,
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    point_features_np = to_numpy(point_features)
    detector_type_np = to_numpy(detector_type)
    selected = sample_indices(point_features_np.shape[0], max_points, seed)
    projected = compute_pca(point_features_np[selected], n_components=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=detector_type_np[selected],
        cmap="coolwarm",
        s=8,
        alpha=0.6,
    )
    fig.colorbar(scatter, ax=ax, ticks=[0, 1], label="detector type")
    ax.set_title("Backbone point-feature PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_figure(fig, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_summary(path: Path, device: torch.device, ran_model_plots: bool, weights_source: str) -> None:
    lines = [
        "ColliderFM diagnostics run",
        "",
        "This directory contains four plot stages:",
        "1. raw/: one detailed event exactly as emitted by the dataloader",
        "2. views/: the point-view tensors that go into the model, plus augmentations",
        "3. model/: learned representations and prototype outputs",
        "4. artifacts/: machine-readable summaries for reproducibility",
        "",
        f"Device used for this run: {device}",
        f"Weights source: {weights_source}",
        f"Model-backed representation plots generated: {'yes' if ran_model_plots else 'no'}",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_dir) if args.output_dir is not None else default_output_dir()
    output_dirs = ensure_output_dirs(output_root)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    detail_events = load_events(args, args.detail_split)
    if not detail_events:
        raise ValueError("The detailed diagnostic split returned no events.")
    detail_event = detail_events[0]

    representation_events = load_events(args, args.representation_split)
    if not representation_events:
        raise ValueError("The representation diagnostic split returned no events.")

    base_view = build_point_view_from_event(
        detail_event,
        device=device,
        max_tracker_hits=args.max_tracker_hits,
        max_calo_hits=args.max_calo_hits,
    )
    aug_a = augment_point_view(base_view)
    aug_b = augment_point_view(base_view)

    plot_raw_geometry(detail_event, output_dirs["raw"] / "event_000_geometry.png")
    plot_raw_scalars(detail_event, output_dirs["raw"] / "event_000_scalars.png")
    plot_view_detector_type(base_view, output_dirs["views"] / "event_000_input_detector_type.png")
    plot_view_signal(base_view, output_dirs["views"] / "event_000_input_signal.png")
    plot_augmentations(base_view, aug_a, aug_b, output_dirs["views"] / "event_000_augmentations.png")
    plot_augmentation_delta(base_view, aug_a, aug_b, output_dirs["views"] / "event_000_augmentation_delta.png")

    representation_views = [
        build_point_view_from_event(
            event,
            device=device,
            max_tracker_hits=args.max_tracker_hits,
            max_calo_hits=args.max_calo_hits,
        )
        for event in representation_events
    ]
    plot_batch_summary(representation_views, output_dirs["views"] / "batch_summary.png")

    model_artifact: dict[str, Any] = {
        "device": str(device),
        "checkpoint_path": args.checkpoint,
        "model_plots_generated": False,
        "weights_source": "fresh initialization" if args.checkpoint is None else f"checkpoint: {args.checkpoint}",
    }
    tensor_artifact: dict[str, Any] = {
        "detail_event": {
            "tracker_hits": int(len(detail_event["tracker_hits"]["x"])),
            "calo_hits": int(len(detail_event["calo_hits"]["x"])),
            "base_view_coord": tensor_summary(base_view["coord"]),
            "base_view_feat": tensor_summary(base_view["feat"]),
        },
        "representation_sample": {
            "num_events": len(representation_views),
            "tracker_points": [int((view["feat"][:, 5] == 0).sum().item()) for view in representation_views],
            "calo_points": [int((view["feat"][:, 5] == 1).sum().item()) for view in representation_views],
        },
    }

    if device.type == "cuda":
        model = create_model(device)
        checkpoint_artifact = None
        if args.checkpoint is not None:
            checkpoint_artifact = load_checkpoint(model, args.checkpoint)
        model.eval()
        num_params = sum(parameter.numel() for parameter in model.parameters())
        model_artifact.update(
            {
                "parameter_count": int(num_params),
                "parameter_count_millions": num_params / 1e6,
                "checkpoint": checkpoint_artifact,
            }
        )

        detail_base_encoding = encode_view(model, base_view, use_teacher=False)
        detail_aug_a_student = encode_view(model, aug_a, use_teacher=False)
        detail_aug_b_student = encode_view(model, aug_b, use_teacher=False)
        detail_aug_a_teacher = encode_view(model, aug_a, use_teacher=True)
        detail_aug_b_teacher = encode_view(model, aug_b, use_teacher=True)

        student_probs = [F.softmax(detail_aug_a_student["logits"][0], dim=-1), F.softmax(detail_aug_b_student["logits"][0], dim=-1)]
        teacher_probs = [F.softmax(detail_aug_a_teacher["logits"][0], dim=-1), F.softmax(detail_aug_b_teacher["logits"][0], dim=-1)]

        plot_logits(
            [to_numpy(prob) for prob in student_probs],
            [to_numpy(prob) for prob in teacher_probs],
            output_dirs["model"] / "event_000_logits.png",
            top_k=args.top_k_prototypes,
        )
        plot_view_agreement(
            detail_base_encoding["pooled"][0],
            detail_aug_a_student["pooled"][0],
            detail_aug_b_student["pooled"][0],
            student_probs,
            teacher_probs,
            output_dirs["model"] / "event_000_view_agreement.png",
        )

        representation_batch = batch_point_views(representation_views)
        representation_encoding = encode_view(model, representation_batch, use_teacher=False)
        event_labels = [f"event_{index:03d}" for index in range(representation_encoding["pooled"].shape[0])]
        plot_embedding_pca(representation_encoding["pooled"], event_labels, output_dirs["model"] / "batch_embedding_pca.png")
        plot_point_feature_pca(
            representation_encoding["point_features"],
            representation_batch["feat"][:, 5],
            output_dirs["model"] / "batch_point_feature_pca.png",
            max_points=args.point_feature_sample_size,
            seed=args.seed,
        )

        model_artifact["model_plots_generated"] = True
        tensor_artifact["detail_event"].update(
            {
                "base_pooled_embedding": tensor_summary(detail_base_encoding["pooled"]),
                "student_aug_a_logits": tensor_summary(detail_aug_a_student["logits"]),
                "teacher_aug_a_logits": tensor_summary(detail_aug_a_teacher["logits"]),
            }
        )
        tensor_artifact["representation_sample"].update(
            {
                "pooled_embeddings": tensor_summary(representation_encoding["pooled"]),
                "point_features": tensor_summary(representation_encoding["point_features"]),
            }
        )
    else:
        print("Skipping model-backed plots because the current PTv3/spconv path requires CUDA.")

    write_json(output_dirs["artifacts"] / "run_config.json", vars(args) | {"resolved_device": str(device)})
    write_json(output_dirs["artifacts"] / "tensor_summary.json", tensor_artifact)
    write_json(output_dirs["artifacts"] / "model_summary.json", model_artifact)
    write_summary(
        output_dirs["root"] / "summary.txt",
        device=device,
        ran_model_plots=model_artifact["model_plots_generated"],
        weights_source=model_artifact["weights_source"],
    )

    print(f"Saved diagnostics to {output_dirs['root']}")


if __name__ == "__main__":
    main()
