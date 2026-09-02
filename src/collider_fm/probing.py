"""Linear probes on frozen backbone features (Panda-style pretraining evaluation).

Reproduces the Panda paper's linear-probing protocol (arXiv 2512.01324): freeze the
pretrained backbone, extract per-point features, and train a single ``nn.Linear`` head
on top. Good representations should be linearly separable/decodable, so probe quality
tracks pretraining quality without finetuning.

Two probes over the deterministic EMA **teacher** backbone's full-up-cast features
(voxel resolution, augmentation-free base view -- the same feature space as the
"full" t-SNE in :mod:`collider_fm.visualization`):

- **Semantic segmentation** (per point): classify each voxel's dominant-particle
  calorimetry-role bucket (:data:`collider_fm.evaluation_labels.PDG_BUCKET_NAMES`).
  Labels follow the full-up-cast labeling chain: output point -> coord-matched voxel ->
  representative raw hit -> dominant contributor's pdg -> bucket. Reported as overall
  accuracy, macro-F1, and mean IoU over classes present in the split.
- **Per-event class energy** (per event): regress the event's total deposited energy
  per class (neutral hadron / photon / charged hadron, from ``contrib_energies``
  bucketed by the contributing particle's pdg) from the mean-pooled per-point features.
  Trained on ``log1p`` energies; reported as per-class R^2 (log1p space) and MAE (raw
  energy units).

Probe train/val events are disjoint slices of the held-out project ``val`` window, so
neither split was seen by pretraining.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ._panda.structure import Point
from .evaluation_labels import PDG_BUCKET_NAMES, event_dominant_particles, pdg_bucket
from .views import build_point_view_from_event
from .visualization import _full_space_labels

__all__ = [
    "ENERGY_PROBE_CLASSES",
    "ProbeDataCollection",
    "event_class_energies",
    "collect_probe_data",
    "train_segmentation_probe",
    "train_energy_probe",
    "plot_confusion_matrix",
    "plot_energy_scatter",
    "plot_loss_curves",
    "format_probes_report",
]

# (metric key, PDG bucket name) pairs defining the energy-regression targets.
ENERGY_PROBE_CLASSES: list[tuple[str, str]] = [
    ("neutral_hadron", "neutral hadron"),
    ("photon", "γ"),
    ("charged_hadron", "charged hadron"),
]
_ENERGY_BUCKET_INDICES = [PDG_BUCKET_NAMES.index(bucket) for _, bucket in ENERGY_PROBE_CLASSES]


def event_class_energies(event: Mapping[str, Any], pid_to_pdg: Mapping[int, int] | None) -> np.ndarray:
    """Total deposited energy per probe class for one raw calo event.

    Flattens `contrib_particle_ids` / `contrib_energies` across all hits, maps each
    contributing particle to its coarse pdg bucket (`pdg_bucket`), and sums the
    contribution energies of the buckets in `ENERGY_PROBE_CLASSES`. Contributions
    from unknown particles (missing from `pid_to_pdg`) or other buckets are ignored.

    Args:
        event (Mapping[str, Any]): Raw calo event with `contrib_particle_ids` and
            `contrib_energies` fields.
        pid_to_pdg (Mapping[int, int] | None): `particle_id -> pdg_id` map. If None,
            all contributions are unknown and the result is all zeros.

    Returns:
        np.ndarray: Class energies of shape `[len(ENERGY_PROBE_CLASSES)]` (float64),
        ordered as `ENERGY_PROBE_CLASSES`.
    """
    out = np.zeros(len(ENERGY_PROBE_CLASSES), dtype=np.float64)
    pids = list(chain.from_iterable(event["contrib_particle_ids"]))
    if not pids or pid_to_pdg is None:
        return out
    energies = np.fromiter(chain.from_iterable(event["contrib_energies"]), dtype=np.float64, count=len(pids))
    pdgs = np.array([pid_to_pdg.get(int(p), -1) for p in pids], dtype=np.int64)
    buckets = pdg_bucket(pdgs)
    for k, bucket_index in enumerate(_ENERGY_BUCKET_INDICES):
        out[k] = float(energies[buckets == bucket_index].sum())
    return out


@dataclass
class ProbeDataCollection:
    """Frozen-feature probe data for one split.

    - ``point_features`` / ``point_labels``: ``[M, D]`` full-up-cast per-point features
      (raw, not normalized) and their ``[M]`` pdg-bucket labels (``-1`` = unknown),
      subsampled across events to ``max_points``.
    - ``event_features`` / ``event_targets``: ``[N, D]`` mean-pooled per-event features
      (over **all** points, before subsampling) and ``[N, C]`` raw class energies
      ordered as :data:`ENERGY_PROBE_CLASSES`.
    """

    point_features: torch.Tensor
    point_labels: torch.Tensor
    event_features: torch.Tensor
    event_targets: torch.Tensor
    num_events: int


@torch.no_grad()
def collect_probe_data(
    model,
    calo_truth_dataset: Any,
    pid_to_pdg: Mapping[int, int] | None,
    device: torch.device,
    *,
    view_kwargs: Mapping[str, Any],
    max_events: int,
    max_points: int = 1_000_000,
    seed: int = 0,
    desc: str = "probe",
) -> ProbeDataCollection:
    """Collect frozen teacher features + probe labels/targets for one split.

    For each event: build one augmentation-free base view, run the deterministic
    teacher backbone, full-up-cast to voxel resolution, then record (a) the per-point
    features with dominant-pdg bucket labels (subsampled proportionally across events
    to `max_points`) and (b) the mean-pooled event feature with the per-class energy
    target from `event_class_energies`.

    Args:
        model: Sonata model with a teacher backbone.
        calo_truth_dataset (Any): Raw calo truth dataset (from `load_calo_truth`),
            already sliced to this split's events.
        pid_to_pdg (Mapping[int, int] | None): `particle_id -> pdg_id` map.
        device (torch.device): Compute device.
        view_kwargs (Mapping[str, Any]): Keyword arguments for
            `build_point_view_from_event`.
        max_events (int): Maximum events to process.
        max_points (int, optional): Cap on per-point rows collected. Defaults to
            1_000_000.
        seed (int, optional): RNG seed for the per-point subsampling. Defaults to 0.
        desc (str, optional): Progress-bar label. Defaults to `"probe"`.

    Returns:
        ProbeDataCollection: Frozen features, labels, and targets for the split.
    """
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    point_feat_parts: list[torch.Tensor] = []
    point_label_parts: list[torch.Tensor] = []
    event_feat_parts: list[torch.Tensor] = []
    event_target_parts: list[torch.Tensor] = []
    total_points = 0
    events_done = 0

    from tqdm import tqdm

    pbar = tqdm(calo_truth_dataset, total=max_events, desc=desc, unit="event", mininterval=2.0)
    for event_index, event in enumerate(pbar):
        if events_done >= max_events:
            break

        wrapped = {
            "calo_hits": {
                "x": torch.as_tensor(event["x"], dtype=torch.float32, device=device),
                "y": torch.as_tensor(event["y"], dtype=torch.float32, device=device),
                "z": torch.as_tensor(event["z"], dtype=torch.float32, device=device),
                "total_energy": torch.as_tensor(event["total_energy"], dtype=torch.float32, device=device),
            }
        }
        view = build_point_view_from_event(wrapped, device=device, **dict(view_kwargs))
        n_in = view["coord"].shape[0]
        if n_in == 0:
            continue

        _, pdg_per_hit, _, _ = event_dominant_particles(event, pid_to_pdg)

        point = Point(
            feat=view["feat"].float(),
            coord=view["coord"].float(),
            origin_coord=view["origin_coord"].float(),
            offset=torch.tensor([n_in], dtype=torch.long, device=device),
            grid_size=view["grid_size"],
            mask=torch.zeros(n_in, dtype=torch.bool, device=device),
        )
        point = model.teacher["backbone"](point)
        feats, pdg_lab = _full_space_labels(point, view, pdg_per_hit)
        labels = torch.as_tensor(pdg_bucket(pdg_lab.numpy()), dtype=torch.long)

        # Per-event pooled feature over all points, before the per-point subsampling.
        event_feat_parts.append(feats.mean(dim=0).float().cpu())
        event_target_parts.append(torch.from_numpy(event_class_energies(event, pid_to_pdg)).float())
        # Note: unlike collect_tsne_points, the early-exit below stops once the per-point
        # budget fills, so the event-level rows may be shorter than max_events when the
        # per-point cap is tight. Use a loose probe_max_points for the energy probe.

        n_out = feats.shape[0]
        remaining_budget = max_points - total_points
        remaining_events = max(1, max_events - events_done)
        if remaining_budget > 0:
            quota = max(1, int(round(remaining_budget / remaining_events)))
            k = min(n_out, quota, remaining_budget)
            idx = torch.randperm(n_out, generator=generator)[:k]
            point_feat_parts.append(feats[idx.to(feats.device)].float().cpu())
            point_label_parts.append(labels[idx])
            total_points += k

        events_done += 1
        pbar.set_postfix(points=total_points, refresh=False)

        if total_points >= max_points:
            break
    pbar.close()

    return ProbeDataCollection(
        point_features=torch.cat(point_feat_parts) if point_feat_parts else torch.empty(0, 0),
        point_labels=torch.cat(point_label_parts) if point_label_parts else torch.empty(0, dtype=torch.long),
        event_features=torch.stack(event_feat_parts) if event_feat_parts else torch.empty(0, 0),
        event_targets=torch.stack(event_target_parts) if event_target_parts else torch.empty(0, len(ENERGY_PROBE_CLASSES)),
        num_events=events_done,
    )


def _standardize(train: torch.Tensor, *others: torch.Tensor, eps: float = 1e-6) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Standardize tensors column-wise by the train tensor's mean/std.

    An affine reparameterization, so it does not change what is linearly decodable --
    it only conditions the optimization uniformly across checkpoints with different
    feature scales.

    Args:
        train (torch.Tensor): `[N, D]` tensor whose statistics are used.
        *others (torch.Tensor): Additional tensors standardized with the same stats.
        eps (float, optional): Minimum std to avoid division by zero. Defaults to 1e-6.

    Returns:
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        `([train_std, *others_std], mean, std)`.
    """
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(eps)
    return [(t - mean) / std for t in (train, *others)], mean, std


def _fit_linear_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    out_features: int,
    loss: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[nn.Linear, float]:
    """Fit one linear head on frozen features with AdamW (Panda-style probe training).

    Data stays on CPU; minibatches are moved to `device` per step.

    Args:
        features (torch.Tensor): `[N, D]` training features.
        targets (torch.Tensor): `[N]` long class ids (`loss="ce"`) or `[N, C]` float
            regression targets (`loss="mse"`).
        out_features (int): Head output width.
        loss (str): `"ce"` or `"mse"`.
        epochs (int): Passes over the data.
        batch_size (int): Rows per optimizer step.
        lr (float): AdamW learning rate.
        weight_decay (float): AdamW weight decay.
        device (torch.device): Compute device for the head.
        seed (int): Seed for head init and shuffling.

    Returns:
        tuple[nn.Linear, list[float]]: The trained head (on `device`) and the mean
        training loss per epoch.

    Raises:
        ValueError: If `loss` is not `"ce"` or `"mse"`.
    """
    if loss not in {"ce", "mse"}:
        raise ValueError(f"loss must be 'ce' or 'mse', got {loss!r}")
    head = nn.Linear(features.shape[1], out_features).to(device)
    # Re-init deterministically from a dedicated generator instead of the RNG-seeding
    # torch.manual_seed, so running a probe does not reseed the shared global RNG
    # mid-eval. kaiming_uniform is PyTorch's default nn.Linear init.
    init_generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        head.weight.data = nn.init.kaiming_uniform_(head.weight, a=5**0.5, generator=init_generator)
        head.bias.data = nn.init.uniform_(head.bias, a=-(1.0 / features.shape[1]) ** 0.5, b=(1.0 / features.shape[1]) ** 0.5, generator=init_generator)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion: nn.Module = nn.CrossEntropyLoss() if loss == "ce" else nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    n = features.shape[0]
    epoch_losses: list[float] = []
    head.train()
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x = features[idx].to(device, non_blocking=True)
            y = targets[idx].to(device, non_blocking=True)
            batch_loss = criterion(head(x), y)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item()) * x.shape[0]
        epoch_losses.append(epoch_loss / max(n, 1))
    return head, epoch_losses


@torch.no_grad()
def _predict(head: nn.Linear, features: torch.Tensor, device: torch.device, batch_size: int = 65536) -> torch.Tensor:
    """Run the head over CPU features in chunks; returns CPU outputs `[N, out_features]`."""
    head.eval()
    parts = [head(features[start : start + batch_size].to(device)).cpu() for start in range(0, features.shape[0], batch_size)]
    return torch.cat(parts) if parts else torch.empty(0, head.out_features)


def _classification_metrics(pred: np.ndarray, target: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Confusion-matrix classification metrics.

    Macro-F1 and mean IoU average only over classes present in `target` (support > 0),
    so absent rare classes do not dilute the averages.

    Args:
        pred (np.ndarray): Predicted class ids, shape `[N]`.
        target (np.ndarray): True class ids, shape `[N]`.
        class_names (list[str]): Names indexing the class ids.

    Returns:
        dict[str, Any]: `accuracy`, `macro_f1`, `mean_iou`, `per_class`
        `{name: {f1, iou, support}}`, and `confusion` -- the raw `[C, C]` count
        matrix indexed `[true, predicted]` (rows sum to the per-class support).
    """
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (target, pred), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros(num_classes), where=denom > 0)
    union = support + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(num_classes), where=union > 0)
    present = support > 0
    return {
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "per_class": {name: {"f1": float(f1[i]), "iou": float(iou[i]), "support": int(support[i])} for i, name in enumerate(class_names)},
        "confusion": confusion.tolist(),
    }


def train_segmentation_probe(
    train_data: ProbeDataCollection,
    val_data: ProbeDataCollection,
    *,
    epochs: int = 20,
    batch_size: int = 65536,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> dict[str, Any]:
    """Train + evaluate the per-point semantic-segmentation linear probe.

    A single `nn.Linear` from frozen per-point features to the pdg buckets
    (`PDG_BUCKET_NAMES`), cross-entropy trained; points with unknown label (`-1`)
    are dropped from both splits. Features are standardized by train statistics.

    Args:
        train_data (ProbeDataCollection): Probe-training split.
        val_data (ProbeDataCollection): Probe-evaluation split.
        epochs (int, optional): Training epochs. Defaults to 20.
        batch_size (int, optional): Rows per optimizer step. Defaults to 65536.
        lr (float, optional): AdamW learning rate. Defaults to 1e-3.
        weight_decay (float, optional): AdamW weight decay. Defaults to 0.0.
        device (torch.device, optional): Compute device. Defaults to CPU.
        seed (int, optional): Seed for head init and shuffling. Defaults to 0.

    Returns:
        dict[str, Any]: `num_train_points`, `num_val_points`, `epoch_losses`
        (mean train loss per epoch), `final_train_loss`, and `train` / `val` metric
        dicts from `_classification_metrics` (each including the raw `confusion`
        count matrix).

    Raises:
        ValueError: If either split has no labeled points.
    """
    train_keep = train_data.point_labels >= 0
    val_keep = val_data.point_labels >= 0
    x_train = train_data.point_features[train_keep].float()
    y_train = train_data.point_labels[train_keep]
    x_val = val_data.point_features[val_keep].float()
    y_val = val_data.point_labels[val_keep]
    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise ValueError("Segmentation probe needs labeled points in both splits.")

    (x_train, x_val), _, _ = _standardize(x_train, x_val)
    head, epoch_losses = _fit_linear_head(
        x_train,
        y_train,
        out_features=len(PDG_BUCKET_NAMES),
        loss="ce",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
    )
    train_pred = _predict(head, x_train, device).argmax(dim=1).numpy()
    val_pred = _predict(head, x_val, device).argmax(dim=1).numpy()
    return {
        "num_train_points": int(x_train.shape[0]),
        "num_val_points": int(x_val.shape[0]),
        "epoch_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "train": _classification_metrics(train_pred, y_train.numpy(), PDG_BUCKET_NAMES),
        "val": _classification_metrics(val_pred, y_val.numpy(), PDG_BUCKET_NAMES),
    }


def _regression_metrics(pred_log: torch.Tensor, target_raw: torch.Tensor) -> dict[str, Any]:
    """Per-class regression metrics in log1p and raw energy space.

    Args:
        pred_log (torch.Tensor): Predicted `log1p` energies, shape `[N, C]`.
        target_raw (torch.Tensor): True raw energies, shape `[N, C]`.

    Returns:
        dict[str, Any]: `mean_r2` and `per_class` `{name: {r2, mae, target_mean}}`,
        where `r2` is in `log1p` space (0.0 when the target is constant) and `mae` /
        `target_mean` are in raw energy units.
    """
    target_log = torch.log1p(target_raw)
    pred_raw = torch.expm1(pred_log)
    per_class: dict[str, dict[str, float]] = {}
    r2_values: list[float] = []
    for k, (name, _) in enumerate(ENERGY_PROBE_CLASSES):
        ss_res = float(((pred_log[:, k] - target_log[:, k]) ** 2).sum())
        ss_tot = float(((target_log[:, k] - target_log[:, k].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        mae = float((pred_raw[:, k] - target_raw[:, k]).abs().mean())
        per_class[name] = {"r2": r2, "mae": mae, "target_mean": float(target_raw[:, k].mean())}
        r2_values.append(r2)
    return {"mean_r2": float(np.mean(r2_values)), "per_class": per_class}


def train_energy_probe(
    train_data: ProbeDataCollection,
    val_data: ProbeDataCollection,
    *,
    epochs: int = 300,
    batch_size: int = 65536,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> dict[str, Any]:
    """Train + evaluate the per-event class-energy linear probe.

    A single `nn.Linear` from mean-pooled frozen event features to the
    `ENERGY_PROBE_CLASSES` energies, MSE-trained on standardized `log1p` targets.
    Features and targets are standardized by train statistics; predictions are
    un-standardized before computing metrics.

    Args:
        train_data (ProbeDataCollection): Probe-training split.
        val_data (ProbeDataCollection): Probe-evaluation split.
        epochs (int, optional): Training epochs (the event-level row count is tiny, so
            more epochs than the segmentation probe). Defaults to 300.
        batch_size (int, optional): Rows per optimizer step. Defaults to 65536.
        lr (float, optional): AdamW learning rate. Defaults to 1e-3.
        weight_decay (float, optional): AdamW weight decay. Defaults to 0.0.
        device (torch.device, optional): Compute device. Defaults to CPU.
        seed (int, optional): Seed for head init and shuffling. Defaults to 0.

    Returns:
        dict[str, Any]: `num_train_events`, `num_val_events`, `epoch_losses` (mean
        train loss per epoch), `final_train_loss`, `train` / `val` metric dicts from
        `_regression_metrics`, and `scatter` -- per-event raw-space predictions and
        targets (`{train_pred, train_target, val_pred, val_target}`, each
        `[N, C]` CPU tensors) for the predicted-vs-true plot. `scatter` holds
        tensors, so callers must pop it before serializing the metrics to JSON.

    Raises:
        ValueError: If either split has fewer than 2 events.
    """
    if train_data.event_features.shape[0] < 2 or val_data.event_features.shape[0] < 2:
        raise ValueError("Energy probe needs at least 2 events in both splits.")
    x_train = train_data.event_features.float()
    x_val = val_data.event_features.float()
    y_train_raw = train_data.event_targets.float()
    y_val_raw = val_data.event_targets.float()

    (x_train, x_val), _, _ = _standardize(x_train, x_val)
    (y_train_std,), y_mean, y_std = _standardize(torch.log1p(y_train_raw))
    head, epoch_losses = _fit_linear_head(
        x_train,
        y_train_std,
        out_features=len(ENERGY_PROBE_CLASSES),
        loss="mse",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
    )
    train_pred_log = _predict(head, x_train, device) * y_std + y_mean
    val_pred_log = _predict(head, x_val, device) * y_std + y_mean
    return {
        "num_train_events": int(x_train.shape[0]),
        "num_val_events": int(x_val.shape[0]),
        "epoch_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "train": _regression_metrics(train_pred_log, y_train_raw),
        "val": _regression_metrics(val_pred_log, y_val_raw),
        "scatter": {
            "train_pred": torch.expm1(train_pred_log),
            "train_target": y_train_raw,
            "val_pred": torch.expm1(val_pred_log),
            "val_target": y_val_raw,
        },
    }


# ---------------------------------------------------------------------------
# Human-reviewable artifacts (PNG plots + plain-text report)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    train_confusion: Sequence[Sequence[int]],
    val_confusion: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
) -> None:
    """Row-normalized confusion heatmaps (train | val) for the segmentation probe.

    Rows are true classes (labeled with their support), columns predicted; cell
    values are percentages of the true class's points. Written to `path` as PNG.

    Args:
        train_confusion (Sequence[Sequence[int]]): `[C, C]` count matrix indexed
            `[true, predicted]` (from `_classification_metrics`).
        val_confusion (Sequence[Sequence[int]]): Same, for the val split.
        class_names (Sequence[str]): Class names indexing rows/columns.
        path (str | Path): Output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    image = None
    for ax, confusion, split in zip(axes, (train_confusion, val_confusion), ("train", "val")):
        counts = np.asarray(confusion, dtype=np.float64)
        support = counts.sum(axis=1, keepdims=True)
        norm = np.divide(counts, support, out=np.zeros_like(counts), where=support > 0)
        image = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels([f"{name} ({int(support[i, 0])})" for i, name in enumerate(class_names)], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true (support)")
        ax.set_title(split)
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.85, label="fraction of true class")
    fig.suptitle("semantic segmentation probe -- confusion matrix")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_energy_scatter(scatter: Mapping[str, torch.Tensor], energy_metrics: Mapping[str, Any], path: str | Path) -> None:
    """Predicted-vs-true per-event class-energy scatters (rows train/val, one column per class).

    Log-log axes with a diagonal reference; points where true or predicted energy is
    <= 0 are dropped (count annotated per panel) since log axes cannot show them.
    Each panel's title carries the class's R^2 (log1p space) and MAE (raw units)
    from `energy_metrics`.

    Args:
        scatter (Mapping[str, torch.Tensor]): The `scatter` dict from
            `train_energy_probe` (`train_pred` / `train_target` / `val_pred` /
            `val_target`, each `[N, C]` raw-energy tensors).
        energy_metrics (Mapping[str, Any]): The metrics dict from
            `train_energy_probe` (for the R^2 / MAE annotations).
        path (str | Path): Output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(ENERGY_PROBE_CLASSES)
    fig, axes = plt.subplots(2, num_classes, figsize=(4.2 * num_classes, 3.8 * 2), squeeze=False)
    for row, split in enumerate(("train", "val")):
        target = scatter[f"{split}_target"]
        pred = scatter[f"{split}_pred"]
        for k, (name, _) in enumerate(ENERGY_PROBE_CLASSES):
            ax = axes[row][k]
            x = target[:, k].numpy()
            y = pred[:, k].numpy()
            keep = (x > 0) & (y > 0)
            if keep.any():
                ax.scatter(x[keep], y[keep], s=4, alpha=0.4, edgecolors="none")
                low = min(x[keep].min(), y[keep].min())
                high = max(x[keep].max(), y[keep].max())
                ax.plot([low, high], [low, high], "k--", lw=0.8)
                ax.set_xscale("log")
                ax.set_yscale("log")
            cls_metrics = energy_metrics[split]["per_class"][name]
            ax.set_title(f"{name} ({split})\nR²={cls_metrics['r2']:.3f}  MAE={cls_metrics['mae']:.2f}", fontsize=9)
            ax.set_xlabel("true energy")
            ax.set_ylabel("predicted energy")
            dropped = int((~keep).sum())
            if dropped:
                ax.text(0.02, 0.98, f"{dropped} pts ≤0", transform=ax.transAxes, fontsize=7, va="top", color="0.35")
    fig.suptitle("energy probe -- predicted vs true per-event class energy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss_curves(seg_losses: Sequence[float], energy_losses: Sequence[float], path: str | Path) -> None:
    """Per-epoch probe training-loss curves (segmentation CE | energy MSE).

    Diagnoses whether the probe heads converged or were still improving when
    training stopped (an underfit probe understates feature quality).

    Args:
        seg_losses (Sequence[float]): Mean train loss per epoch from
            `train_segmentation_probe`.
        energy_losses (Sequence[float]): Mean train loss per epoch from
            `train_energy_probe` (standardized log1p space).
        path (str | Path): Output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, losses, label in zip(
        axes,
        (seg_losses, energy_losses),
        ("segmentation (cross-entropy)", "energy (MSE, standardized log1p)"),
    ):
        ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("linear probe training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _loss_convergence(losses: Sequence[float]) -> str:
    """One-line convergence summary: final loss and the change over the last 5 epochs."""
    if not losses:
        return "n/a"
    reference = losses[-6] if len(losses) > 5 else losses[0]
    return f"{losses[-1]:.4f} ({losses[-1] - reference:+.4f} over last 5 epochs)"


def format_probes_report(
    main_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None = None,
    *,
    main_label: str = "trained",
    baseline_label: str = "random init",
) -> str:
    """Render the probe results (optionally side-by-side with a baseline) as plain text.

    Sections: segmentation per-class table (F1 / IoU / support), energy per-class
    table (R^2 / MAE / target mean), overall summaries, and probe convergence.

    Args:
        main_metrics (Mapping[str, Any]): `{"semantic_segmentation": ..., "energy": ...}`
            from the main probe run (as returned by `train_*_probe`, `scatter` popped).
        baseline_metrics (Mapping[str, Any] | None): Same shape, from the baseline
            probe run. If None, single-column tables are rendered.
        main_label (str, optional): Label for the main columns. Defaults to "trained".
        baseline_label (str, optional): Label for the baseline columns. Defaults to
            "random init".

    Returns:
        str: The plain-text report.
    """

    def _col(value: Any, fmt: str = ".4f") -> str:
        if value is None:
            return " " * 17
        return f"{value:>17{fmt}}"

    def _seg_value(metrics: Mapping[str, Any] | None, name: str, key: str) -> float | None:
        if metrics is None:
            return None
        return metrics["semantic_segmentation"]["val"]["per_class"][name][key]

    def _energy_value(metrics: Mapping[str, Any] | None, name: str, key: str) -> float | None:
        if metrics is None:
            return None
        return metrics["energy"]["val"]["per_class"][name][key]

    lines: list[str] = [
        "Linear probes on frozen teacher features",
        "=" * 60,
        f"weights         : {main_label}",
        f"baseline        : {baseline_label if baseline_metrics is not None else 'none'}",
    ]
    seg = main_metrics["semantic_segmentation"]
    energy = main_metrics["energy"]
    lines += [
        f"probe data      : {seg['num_train_points']} / {seg['num_val_points']} points (segmentation), "
        f"{energy['num_train_events']} / {energy['num_val_events']} events (energy)",
        "",
        "Semantic segmentation (per-point dominant-particle bucket)",
        "-" * 60,
        f"  {'class':<18}{'support':>10}   {'F1 [' + main_label + ']':>17}   {'F1 [' + baseline_label + ']':>17}   "
        f"{'IoU [' + main_label + ']':>17}   {'IoU [' + baseline_label + ']':>17}",
    ]
    for name in PDG_BUCKET_NAMES:
        support = seg["val"]["per_class"][name]["support"]
        lines.append(
            f"  {name:<18}{support:>10}   "
            f"{_col(_seg_value(main_metrics, name, 'f1'))}   {_col(_seg_value(baseline_metrics, name, 'f1'))}   "
            f"{_col(_seg_value(main_metrics, name, 'iou'))}   {_col(_seg_value(baseline_metrics, name, 'iou'))}"
        )
    for key, label in (("accuracy", "accuracy"), ("macro_f1", "macro F1"), ("mean_iou", "mean IoU")):
        baseline_value = baseline_metrics["semantic_segmentation"]["val"][key] if baseline_metrics is not None else None
        lines.append(
            f"  {label:<18}{'':>10}   {_col(seg['val'][key])}   {_col(baseline_value)}".rstrip()
        )
    lines += [
        "",
        "Energy regression (per-event class energy; R2 in log1p space, MAE in raw units)",
        "-" * 60,
        f"  {'class':<18}{'target mean':>12}   {'R2 [' + main_label + ']':>17}   {'R2 [' + baseline_label + ']':>17}   "
        f"{'MAE [' + main_label + ']':>17}   {'MAE [' + baseline_label + ']':>17}",
    ]
    for name, _ in ENERGY_PROBE_CLASSES:
        cls_metrics = energy["val"]["per_class"][name]
        lines.append(
            f"  {name:<18}{cls_metrics['target_mean']:>12.2f}   "
            f"{_col(cls_metrics['r2'])}   {_col(_energy_value(baseline_metrics, name, 'r2'))}   "
            f"{_col(cls_metrics['mae'], fmt='.3f')}   {_col(_energy_value(baseline_metrics, name, 'mae'), fmt='.3f')}"
        )
    baseline_mean_r2 = baseline_metrics["energy"]["val"]["mean_r2"] if baseline_metrics is not None else None
    lines.append(f"  {'mean R2':<18}{'':>12}   {_col(energy['val']['mean_r2'])}   {_col(baseline_mean_r2)}".rstrip())
    lines += [
        "",
        "Probe convergence (final-epoch train loss)",
        "-" * 60,
        f"  {f'segmentation [{main_label}]':<28}: {_loss_convergence(seg['epoch_losses'])}",
        f"  {f'energy [{main_label}]':<28}: {_loss_convergence(energy['epoch_losses'])}",
    ]
    if baseline_metrics is not None:
        lines += [
            f"  {f'segmentation [{baseline_label}]':<28}: {_loss_convergence(baseline_metrics['semantic_segmentation']['epoch_losses'])}",
            f"  {f'energy [{baseline_label}]':<28}: {_loss_convergence(baseline_metrics['energy']['epoch_losses'])}",
        ]
    return "\n".join(lines) + "\n"
