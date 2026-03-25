"""Plot metric curves for a completed training run.

This script reads `config.json` and `metrics.jsonl` from a run directory created by
`scripts/train.py`, then writes a small set of run-level plots into
`<run_dir>/plots/` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved metrics for a completed ColliderFM training run."
    )
    parser.add_argument("run_dir", help="Path to a completed run directory.")
    parser.add_argument(
        "--output-subdir",
        default="plots",
        help="Subdirectory inside the run directory where plots will be saved.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metric_records(metrics_path: Path) -> list[dict[str, Any]]:
    records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    if not records:
        raise ValueError(f"No metric records found in {metrics_path}")
    return records


def metric_series(
    records: list[dict[str, Any]], key: str
) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for index, record in enumerate(records, start=1):
        if key not in record:
            continue
        xs.append(float(record.get("epoch", index)))
        ys.append(float(record[key]))
    return xs, ys


def metric_gap(
    records: list[dict[str, Any]], train_key: str, val_key: str
) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for index, record in enumerate(records, start=1):
        if train_key not in record or val_key not in record:
            continue
        xs.append(float(record.get("epoch", index)))
        ys.append(float(record[val_key]) - float(record[train_key]))
    return xs, ys


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def best_epoch(records: list[dict[str, Any]]) -> int | None:
    candidates = [record for record in records if "val_loss" in record]
    if not candidates:
        return None
    best_record = min(candidates, key=lambda record: float(record["val_loss"]))
    return int(best_record.get("epoch", 0))


def add_best_epoch_marker(axis: plt.Axes, best_epoch_index: int | None) -> None:
    if best_epoch_index is None:
        return
    axis.axvline(best_epoch_index, color="tab:gray", linestyle="--", linewidth=1)


def plot_metric_pair(
    axis: plt.Axes,
    records: list[dict[str, Any]],
    title: str,
    train_key: str,
    val_key: str,
    best_epoch_index: int | None,
) -> None:
    train_x, train_y = metric_series(records, train_key)
    val_x, val_y = metric_series(records, val_key)
    if not train_y and not val_y:
        axis.set_axis_off()
        return
    if train_y:
        axis.plot(train_x, train_y, marker="o", label="train")
    if val_y:
        axis.plot(val_x, val_y, marker="o", label="val")
    add_best_epoch_marker(axis, best_epoch_index)
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.legend()


def plot_metric_single(
    axis: plt.Axes,
    records: list[dict[str, Any]],
    title: str,
    key: str,
    best_epoch_index: int | None,
) -> None:
    xs, ys = metric_series(records, key)
    if not ys:
        axis.set_axis_off()
        return
    axis.plot(xs, ys, marker="o")
    add_best_epoch_marker(axis, best_epoch_index)
    axis.set_title(title)
    axis.set_xlabel("epoch")


def plot_gap_series(
    axis: plt.Axes,
    records: list[dict[str, Any]],
    title: str,
    train_key: str,
    val_key: str,
    best_epoch_index: int | None,
) -> None:
    xs, ys = metric_gap(records, train_key, val_key)
    if not ys:
        axis.set_axis_off()
        return
    axis.plot(xs, ys, marker="o", color="tab:purple")
    axis.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1)
    add_best_epoch_marker(axis, best_epoch_index)
    axis.set_title(title)
    axis.set_xlabel("epoch")


def plot_loss_curves(
    records: list[dict[str, Any]], path: Path, best_epoch_index: int | None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_metric_pair(ax, records, "Loss", "train_loss", "val_loss", best_epoch_index)
    save_figure(fig, path)


def plot_representation_curves(
    records: list[dict[str, Any]], path: Path, best_epoch_index: int | None
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_metric_pair(
        axes[0],
        records,
        "Prototype Entropy",
        "train_prototype_entropy",
        "val_prototype_entropy",
        best_epoch_index,
    )
    plot_metric_pair(
        axes[1],
        records,
        "Embedding Norm",
        "train_embedding_norm",
        "val_embedding_norm",
        best_epoch_index,
    )
    plot_metric_pair(
        axes[2],
        records,
        "Masked Fraction",
        "train_masked_fraction",
        "val_masked_fraction",
        best_epoch_index,
    )
    fig.suptitle("Representation Metrics")
    save_figure(fig, path)


def plot_optimization_curves(
    records: list[dict[str, Any]], path: Path, best_epoch_index: int | None
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_metric_single(
        axes[0, 0], records, "Learning Rate", "learning_rate", best_epoch_index
    )
    plot_metric_single(
        axes[0, 1], records, "Teacher Momentum", "teacher_momentum", best_epoch_index
    )
    plot_metric_single(
        axes[1, 0],
        records,
        "Teacher Temperature",
        "teacher_temperature",
        best_epoch_index,
    )
    plot_metric_single(
        axes[1, 1], records, "Center Norm", "center_norm", best_epoch_index
    )
    fig.suptitle("Optimization And Teacher Schedules")
    save_figure(fig, path)


def plot_run_health(
    records: list[dict[str, Any]], path: Path, best_epoch_index: int | None
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_gap_series(
        axes[0, 0],
        records,
        "Loss Gap (val - train)",
        "train_loss",
        "val_loss",
        best_epoch_index,
    )
    plot_gap_series(
        axes[0, 1],
        records,
        "Prototype Entropy Gap",
        "train_prototype_entropy",
        "val_prototype_entropy",
        best_epoch_index,
    )
    plot_gap_series(
        axes[1, 0],
        records,
        "Embedding Norm Gap",
        "train_embedding_norm",
        "val_embedding_norm",
        best_epoch_index,
    )
    plot_metric_single(
        axes[1, 1], records, "Epoch Time [s]", "epoch_time_seconds", best_epoch_index
    )
    fig.suptitle("Run Health")
    save_figure(fig, path)


def summarize_run(
    run_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    best_epoch_index = best_epoch(records)
    best_record = next(
        (
            record
            for record in records
            if int(record.get("epoch", -1)) == best_epoch_index
        ),
        records[-1],
    )
    final_record = records[-1]
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_files = (
        sorted(path.name for path in checkpoints_dir.iterdir())
        if checkpoints_dir.exists()
        else []
    )
    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "num_metric_records": len(records),
        "configured_num_epochs": config.get("num_epochs"),
        "best_epoch": best_epoch_index,
        "best_val_loss": best_record.get("val_loss"),
        "final_epoch": final_record.get("epoch"),
        "final_train_loss": final_record.get("train_loss"),
        "final_val_loss": final_record.get("val_loss"),
        "final_train_prototype_entropy": final_record.get("train_prototype_entropy"),
        "final_val_prototype_entropy": final_record.get("val_prototype_entropy"),
        "final_train_embedding_norm": final_record.get("train_embedding_norm"),
        "final_val_embedding_norm": final_record.get("val_embedding_norm"),
        "total_epoch_time_seconds": sum(
            float(record.get("epoch_time_seconds", 0.0)) for record in records
        ),
        "checkpoint_files": checkpoint_files,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "ColliderFM training-run plots",
        "",
        f"Run directory: {summary['run_dir']}",
        f"Plot directory: {summary['output_dir']}",
        f"Metric records: {summary['num_metric_records']}",
        f"Best epoch by val_loss: {summary['best_epoch']}",
        f"Best val_loss: {summary['best_val_loss']}",
        f"Final epoch: {summary['final_epoch']}",
        f"Final train_loss: {summary['final_train_loss']}",
        f"Final val_loss: {summary['final_val_loss']}",
        f"Total recorded epoch time [s]: {summary['total_epoch_time_seconds']:.2f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)
    records = load_metric_records(metrics_path)
    best_epoch_index = best_epoch(records)

    plot_loss_curves(records, output_dir / "loss_curves.png", best_epoch_index)
    plot_representation_curves(
        records, output_dir / "representation_curves.png", best_epoch_index
    )
    plot_optimization_curves(
        records, output_dir / "optimization_curves.png", best_epoch_index
    )
    plot_run_health(records, output_dir / "run_health.png", best_epoch_index)

    summary = summarize_run(run_dir, output_dir, config, records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(output_dir / "summary.txt", summary)

    print(f"Saved training-run plots to {output_dir}")


if __name__ == "__main__":
    main()
