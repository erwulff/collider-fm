"""Generate diagnostic plots for the ColliderFM calorimeter pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
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
from collider_fm.model import create_small_panda_model, create_training_panda_model
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    model_factory_kwargs,
    to_plain_container,
)
from collider_fm.views import (
    augment_point_view,
    batch_point_views,
    build_point_view_from_event,
)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description=(
            "Generate diagnostic plots for the ColliderFM calorimeter data-to-view-to-representation pipeline. "
            "By default the script traces one detailed event and separately summarizes a 10-event sample for PCA plots."
        ),
        epilog=(
            "Examples:\n"
            "  uv run python scripts/plot_diagnostics.py\n"
            "  uv run python scripts/plot_diagnostics.py diagnostics.detail_split=train[0:2]\n"
            "  uv run python scripts/plot_diagnostics.py diagnostics.device=cpu data.local_files_only=true"
        ),
    )


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
        print(
            "CUDA was requested but is unavailable; falling back to CPU for raw/view diagnostics only."
        )
        return torch.device("cpu")
    return torch.device(device_name)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_raw_geometry(event: dict[str, Any], path: Path) -> None:
    calo_hits = event["calo_hits"]
    coord = np.stack(
        [to_numpy(calo_hits["z"]), to_numpy(calo_hits["x"]), to_numpy(calo_hits["y"])],
        axis=1,
    )
    energy = to_numpy(calo_hits["energy"])
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

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        coord[:, 0],
        coord[:, 1],
        coord[:, 2],
        s=marker_size,
        alpha=0.55,
        c=energy,
        cmap="inferno",
        norm=norm,
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    colorbar.set_label("calo energy")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title("Raw calorimeter event geometry")
    save_figure(fig, path)


def plot_raw_scalars(event: dict[str, Any], path: Path) -> None:
    calo_hits = event["calo_hits"]
    coord = torch.stack([calo_hits["x"], calo_hits["y"], calo_hits["z"]], dim=1)
    radius = torch.linalg.norm(coord, dim=1)
    energy = calo_hits["energy"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(to_numpy(energy), bins=40, color="tab:red")
    axes[0].set_title("Calo energy")
    axes[1].hist(to_numpy(radius), bins=40, color="tab:orange")
    axes[1].set_title("Calo radius")
    axes[2].axis("off")
    axes[2].text(
        0.0,
        0.8,
        "\n".join(
            [
                f"calo hits: {len(calo_hits['x'])}",
                f"energy range: [{energy.min().item():.2f}, {energy.max().item():.2f}]",
                f"radius range: [{radius.min().item():.2f}, {radius.max().item():.2f}]",
            ]
        ),
        fontsize=11,
        va="top",
    )
    fig.suptitle("Raw calorimeter scalar summaries")
    save_figure(fig, path)


def plot_view_signal(view: dict[str, torch.Tensor], path: Path) -> None:
    coord = to_numpy(view["coord"])
    energy = to_numpy(view["energy"])
    features = to_numpy(view["feat"])
    mask = to_numpy(view["mask"]).astype(int)

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(3, 2)
    ax_scatter = fig.add_subplot(grid[:, 0], projection="3d")
    scatter = ax_scatter.scatter(
        coord[:, 2], coord[:, 0], coord[:, 1], c=energy, cmap="inferno", s=4, alpha=0.7
    )
    fig.colorbar(scatter, ax=ax_scatter, shrink=0.7, pad=0.1, label="energy")
    ax_scatter.set_xlabel("z")
    ax_scatter.set_ylabel("x")
    ax_scatter.set_zlabel("y")
    ax_scatter.set_title("Model input colored by energy")

    axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 1]),
    ]
    axes[0].hist(features[:, 0], bins=40, alpha=0.8, label="x")
    axes[0].hist(features[:, 1], bins=40, alpha=0.6, label="y")
    axes[0].hist(features[:, 2], bins=40, alpha=0.5, label="z")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Coordinate feature histograms")
    axes[1].hist(features[:, 3], bins=40, alpha=0.8, label="energy")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Energy histogram")
    axes[2].bar(
        [0, 1],
        [int((mask == 0).sum()), int((mask == 1).sum())],
        color=["tab:blue", "tab:red"],
        tick_label=["visible", "masked"],
    )
    axes[2].set_title("Mask state counts")
    axes[2].set_ylabel("points")
    fig.suptitle("Model input feature summary")
    save_figure(fig, path)


def plot_augmentations(
    base_view: dict[str, torch.Tensor],
    aug_a: dict[str, torch.Tensor],
    aug_b: dict[str, torch.Tensor],
    path: Path,
) -> None:
    views = [("base", base_view), ("aug A", aug_a), ("aug B", aug_b)]
    fig = plt.figure(figsize=(18, 6))
    for index, (label, view) in enumerate(views, start=1):
        coord = to_numpy(view["coord"])
        energy = to_numpy(view["energy"])
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.scatter(
            coord[:, 2],
            coord[:, 0],
            coord[:, 1],
            c=energy,
            cmap="inferno",
            s=4,
            alpha=0.7,
        )
        ax.set_title(label)
        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
    fig.suptitle("Detailed event: base and augmented views")
    save_figure(fig, path)


def plot_augmentation_delta(
    base_view: dict[str, torch.Tensor],
    aug_a: dict[str, torch.Tensor],
    aug_b: dict[str, torch.Tensor],
    path: Path,
) -> None:
    base_coord = base_view["coord"]
    base_energy = base_view["energy"]
    deltas = {
        "aug A": {
            "coord": torch.linalg.norm(aug_a["coord"] - base_coord, dim=1),
            "energy": aug_a["energy"] - base_energy,
        },
        "aug B": {
            "coord": torch.linalg.norm(aug_b["coord"] - base_coord, dim=1),
            "energy": aug_b["energy"] - base_energy,
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, values in deltas.items():
        axes[0].hist(to_numpy(values["coord"]), bins=40, alpha=0.6, label=label)
        axes[1].hist(to_numpy(values["energy"]), bins=40, alpha=0.6, label=label)
    axes[0].set_title("Coordinate displacement magnitude")
    axes[1].set_title("Energy perturbation")
    for axis in axes:
        axis.legend()
    fig.suptitle("Augmentation deltas relative to the base view")
    save_figure(fig, path)


def plot_batch_summary(views: list[dict[str, torch.Tensor]], path: Path) -> None:
    total_counts = [view["coord"].shape[0] for view in views]
    masked_counts = [int(view["mask"].sum().item()) for view in views]

    indices = np.arange(len(views))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(indices, total_counts, label="points", color="tab:red")
    axes[0].bar(indices, masked_counts, label="masked", color="tab:blue")
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


def plot_logits(
    student_probs: list[np.ndarray],
    teacher_probs: list[np.ndarray],
    path: Path,
    top_k: int,
) -> None:
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
    js = 0.5 * F.kl_div(prob_a.log(), mean_prob, reduction="sum") + 0.5 * F.kl_div(
        prob_b.log(), mean_prob, reduction="sum"
    )
    return float(js.item())


def embedding_cosine_similarity(
    embedding_a: torch.Tensor, embedding_b: torch.Tensor
) -> float:
    return float(
        F.cosine_similarity(
            embedding_a.reshape(1, -1), embedding_b.reshape(1, -1), dim=1
        ).item()
    )


def plot_view_agreement(
    base_pooled: torch.Tensor,
    aug_a_pooled: torch.Tensor,
    aug_b_pooled: torch.Tensor,
    student_probs: list[torch.Tensor],
    teacher_probs: list[torch.Tensor],
    path: Path,
) -> None:
    cosine_values = {
        "base vs aug A": embedding_cosine_similarity(base_pooled, aug_a_pooled),
        "base vs aug B": embedding_cosine_similarity(base_pooled, aug_b_pooled),
        "aug A vs aug B": embedding_cosine_similarity(aug_a_pooled, aug_b_pooled),
    }
    js_values = {
        "student": jensen_shannon_divergence(student_probs[0], student_probs[1]),
        "teacher": jensen_shannon_divergence(teacher_probs[0], teacher_probs[1]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        list(cosine_values.keys()), list(cosine_values.values()), color="tab:green"
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Embedding cosine similarity")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(list(js_values.keys()), list(js_values.values()), color="tab:orange")
    axes[1].set_title("Jensen-Shannon divergence")
    fig.suptitle("Detailed event: view agreement")
    save_figure(fig, path)


def plot_embedding_pca(
    embeddings: torch.Tensor, event_labels: list[str], path: Path
) -> None:
    projected = compute_pca(to_numpy(embeddings), n_components=2)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=np.arange(len(event_labels)),
        cmap="tab10",
        s=60,
    )
    for index, label in enumerate(event_labels):
        ax.annotate(label, (projected[index, 0], projected[index, 1]), fontsize=8)
    ax.set_title("Pooled event embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_figure(fig, path)


def plot_point_feature_pca(
    point_features: torch.Tensor, path: Path, max_points: int, seed: int
) -> None:
    point_features_np = to_numpy(point_features)
    if point_features_np.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(point_features_np.shape[0], size=max_points, replace=False)
        )
    else:
        selected = np.arange(point_features_np.shape[0])
    projected = compute_pca(point_features_np[selected], n_components=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(projected[:, 0], projected[:, 1], s=8, alpha=0.6, color="tab:red")
    ax.set_title("Backbone point-feature PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_figure(fig, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_summary(
    path: Path, device: torch.device, ran_model_plots: bool, weights_source: str
) -> None:
    lines = [
        "ColliderFM diagnostics run",
        "",
        "This directory contains four plot stages:",
        "1. raw/: one detailed calorimeter event exactly as emitted by the dataloader",
        "2. views/: the point-view tensors that go into the model, plus augmentations",
        "3. model/: learned representations and prototype outputs",
        "4. artifacts/: machine-readable summaries for reproducibility",
        "",
        f"Device used for this run: {device}",
        f"Weights source: {weights_source}",
        f"Model-backed representation plots generated: {'yes' if ran_model_plots else 'no'}",
    ]
    path.write_text("\n".join(lines))


def infer_metrics_path(
    checkpoint_path: str | None, metrics_file: str | None
) -> Path | None:
    if metrics_file is not None:
        return Path(metrics_file)
    if checkpoint_path is None:
        return None

    checkpoint = Path(checkpoint_path)
    run_dir = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "checkpoints"
        else checkpoint.parent
    )
    candidate = run_dir / "metrics.jsonl"
    if candidate.exists():
        return candidate
    return None


def load_metric_records(metrics_path: Path) -> list[dict[str, Any]]:
    records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _metric_series(
    records: Sequence[dict[str, Any]], key: str
) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for record in records:
        if key not in record:
            continue
        xs.append(float(record.get("epoch", len(xs) + 1)))
        ys.append(float(record[key]))
    return xs, ys


def plot_metric_curves(records: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    curve_specs = [
        ("Loss", "train_loss", "val_loss"),
        ("Prototype entropy", "train_prototype_entropy", "val_prototype_entropy"),
        ("Embedding norm", "train_embedding_norm", "val_embedding_norm"),
    ]

    for axis, (title, train_key, val_key) in zip(axes, curve_specs):
        train_x, train_y = _metric_series(records, train_key)
        val_x, val_y = _metric_series(records, val_key)
        if train_y:
            axis.plot(train_x, train_y, marker="o", label="train")
        if val_y:
            axis.plot(val_x, val_y, marker="o", label="val")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.legend()

    fig.suptitle("Training metrics")
    save_figure(fig, path)


def plot_schedule_curves(records: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    curve_specs = [
        ("Learning rate", "learning_rate"),
        ("Teacher momentum", "teacher_momentum"),
        ("Teacher temperature", "teacher_temperature"),
    ]

    for axis, (title, key) in zip(axes, curve_specs):
        xs, ys = _metric_series(records, key)
        if ys:
            axis.plot(xs, ys, marker="o")
        axis.set_title(title)
        axis.set_xlabel("epoch")

    fig.suptitle("Training schedules")
    save_figure(fig, path)


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    project_config = to_plain_container(
        load_project_config(cli_args.config, cli_args.overrides)
    )
    data_config = project_config["data"]
    diagnostics_config = project_config["diagnostics"]
    set_seed(diagnostics_config["seed"])

    output_root = (
        Path(diagnostics_config.get("output_dir"))
        if diagnostics_config.get("output_dir") is not None
        else default_output_dir()
    )
    output_dirs = ensure_output_dirs(output_root)

    device = resolve_device(diagnostics_config["device"])
    print(f"Using device: {device}")

    detail_events = load_events(
        split=diagnostics_config["detail_split"],
        dataset_name=data_config["dataset_name"],
        dataset_type=data_config["dataset_type"],
        pu_config=data_config["pu_config"],
        cache_dir=data_config["cache_dir"],
        dataset_revision=data_config["dataset_revision"],
        local_files_only=data_config["local_files_only"],
    )
    if not detail_events:
        raise ValueError("The detailed diagnostic split returned no events.")
    detail_event = detail_events[0]

    representation_events = load_events(
        split=diagnostics_config["representation_split"],
        dataset_name=data_config["dataset_name"],
        dataset_type=data_config["dataset_type"],
        pu_config=data_config["pu_config"],
        cache_dir=data_config["cache_dir"],
        dataset_revision=data_config["dataset_revision"],
        local_files_only=data_config["local_files_only"],
    )
    if not representation_events:
        raise ValueError("The representation diagnostic split returned no events.")

    base_view = build_point_view_from_event(
        detail_event, device=device, max_calo_hits=diagnostics_config["max_calo_hits"]
    )
    aug_a = augment_point_view(base_view)
    aug_b = augment_point_view(base_view)

    plot_raw_geometry(detail_event, output_dirs["raw"] / "event_000_geometry.png")
    plot_raw_scalars(detail_event, output_dirs["raw"] / "event_000_scalars.png")
    plot_view_signal(base_view, output_dirs["views"] / "event_000_input_signal.png")
    plot_augmentations(
        base_view, aug_a, aug_b, output_dirs["views"] / "event_000_augmentations.png"
    )
    plot_augmentation_delta(
        base_view,
        aug_a,
        aug_b,
        output_dirs["views"] / "event_000_augmentation_delta.png",
    )

    representation_views = [
        build_point_view_from_event(
            event,
            device=device,
            max_calo_hits=diagnostics_config["max_calo_hits"],
        )
        for event in representation_events
    ]
    plot_batch_summary(representation_views, output_dirs["views"] / "batch_summary.png")

    model_artifact: dict[str, Any] = {
        "device": str(device),
        "checkpoint_path": diagnostics_config.get("checkpoint"),
        "metrics_path": diagnostics_config.get("metrics_file"),
        "model_plots_generated": False,
        "weights_source": "fresh initialization"
        if diagnostics_config.get("checkpoint") is None
        else f"checkpoint: {diagnostics_config['checkpoint']}",
    }
    tensor_artifact: dict[str, Any] = {
        "detail_event": {
            "calo_hits": int(len(detail_event["calo_hits"]["x"])),
            "base_view_coord": tensor_summary(base_view["coord"]),
            "base_view_feat": tensor_summary(base_view["feat"]),
        },
        "representation_sample": {
            "num_events": len(representation_views),
            "point_counts": [
                int(view["coord"].shape[0]) for view in representation_views
            ],
            "masked_counts": [
                int(view["mask"].sum().item()) for view in representation_views
            ],
        },
    }

    metrics_path = infer_metrics_path(
        diagnostics_config.get("checkpoint"), diagnostics_config.get("metrics_file")
    )
    if metrics_path is not None and metrics_path.exists():
        metric_records = load_metric_records(metrics_path)
        if metric_records:
            plot_metric_curves(
                metric_records, output_dirs["model"] / "training_metrics.png"
            )
            plot_schedule_curves(
                metric_records, output_dirs["model"] / "training_schedules.png"
            )
            model_artifact["metrics_path"] = str(metrics_path)
            model_artifact["num_metric_records"] = len(metric_records)

    if device.type == "cuda":
        model = (
            create_training_panda_model(
                device=device,
                **model_factory_kwargs(project_config["model"]["training"]),
            )
            if diagnostics_config.get("checkpoint") is not None
            else create_small_panda_model(
                device=device,
                **model_factory_kwargs(project_config["model"]["diagnostics"]),
            )
        )
        checkpoint_artifact = None
        if diagnostics_config.get("checkpoint") is not None:
            checkpoint_artifact = load_checkpoint(
                model, diagnostics_config["checkpoint"]
            )
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

        student_probs = [
            F.softmax(detail_aug_a_student["logits"][0], dim=-1),
            F.softmax(detail_aug_b_student["logits"][0], dim=-1),
        ]
        teacher_probs = [
            F.softmax(detail_aug_a_teacher["logits"][0], dim=-1),
            F.softmax(detail_aug_b_teacher["logits"][0], dim=-1),
        ]

        plot_logits(
            [to_numpy(prob) for prob in student_probs],
            [to_numpy(prob) for prob in teacher_probs],
            output_dirs["model"] / "event_000_logits.png",
            top_k=diagnostics_config["top_k_prototypes"],
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
        representation_encoding = encode_view(
            model, representation_batch, use_teacher=False
        )
        event_labels = [
            f"event_{index:03d}"
            for index in range(representation_encoding["pooled"].shape[0])
        ]
        plot_embedding_pca(
            representation_encoding["pooled"],
            event_labels,
            output_dirs["model"] / "batch_embedding_pca.png",
        )
        plot_point_feature_pca(
            representation_encoding["point_features"],
            output_dirs["model"] / "batch_point_feature_pca.png",
            max_points=diagnostics_config["point_feature_sample_size"],
            seed=diagnostics_config["seed"],
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
                "point_features": tensor_summary(
                    representation_encoding["point_features"]
                ),
            }
        )
    else:
        print(
            "Skipping model-backed plots because the current PTv3/spconv path requires CUDA."
        )

    write_json(
        output_dirs["artifacts"] / "run_config.json",
        project_config | {"resolved_device": str(device)},
    )
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
