from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import comet_ml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
from collider_fm.model import create_training_model
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    model_factory_kwargs,
    select_model_config,
    to_plain_container,
)
from collider_fm.views import build_sonata_batch


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Train the ColliderFM Sonata self-distillation model.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train.py\n"
            "  uv run python scripts/train.py training.batch_size=16 training.num_epochs=10\n"
            "  uv run python scripts/train.py data.local_files_only=true training.log_backend=jsonl\n"
            "  uv run python scripts/train.py training.run_dir=runs training.run_name=my_run"
        ),
        config_sections=(
            "data",
            "views",
            "model.training",
            "training",
        ),
    )


def create_dataloader(config: DictConfig, split: str, shuffle: bool) -> DataLoader:
    """Create the calo-only dataloader used by train and validation."""

    data_config = config.data
    training_config = config.training
    dataset = ColliderMLDataset(
        dataset_name=data_config.dataset_name,
        split=split,
        dataset_type=data_config.dataset_type,
        pu_config=data_config.pu_config,
        object_types=["calo_hits"],
        cache_dir=data_config.cache_dir,
        dataset_revision=data_config.dataset_revision,
        local_files_only=data_config.local_files_only,
    )
    dataloader_kwargs = {
        "batch_size": training_config.batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": training_config.num_workers,
        "pin_memory": bool(training_config.pin_memory and torch.cuda.is_available()),
    }
    if training_config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = training_config.prefetch_factor

    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def resolve_mixed_precision_dtype(
    training_config: DictConfig, device: torch.device
) -> torch.dtype | None:
    """Resolve the requested mixed-precision mode for this device."""

    mode = str(training_config.get("mixed_precision", "none")).lower()
    if mode == "none":
        return None
    if device.type != "cuda":
        print("Mixed precision requested without CUDA; disabling mixed precision.")
        return None
    if mode in {"bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print(
            "BF16 mixed precision is not supported on this GPU; disabling mixed precision."
        )
        return None
    if mode in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(
        f"Unsupported training.mixed_precision value: {training_config.mixed_precision}. "
        "Use one of: none, bf16, fp16."
    )


def mixed_precision_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "none"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return str(dtype)


def resolve_epoch_batch_limit(
    dataloader: DataLoader, requested_max_batches: int | None, phase: str
) -> int:
    """Resolve an epoch batch limit, using the full dataloader when unset."""

    total_batches = len(dataloader)
    if total_batches <= 0:
        raise ValueError(f"The {phase} dataloader produced zero batches.")
    if requested_max_batches is None:
        return total_batches
    if requested_max_batches <= 0:
        raise ValueError(
            f"{phase} max_batches must be a positive integer or None, got {requested_max_batches}."
        )
    return min(total_batches, requested_max_batches)


def learning_rate(optimizer: AdamW) -> float:
    """Return the optimizer learning rate from the first parameter group."""

    return float(optimizer.param_groups[0]["lr"])


def prototype_usage(logits: torch.Tensor, num_prototypes: int) -> torch.Tensor:
    """Estimate prototype occupancy from the student point assignments."""

    assignments = logits.argmax(dim=-1)
    counts = torch.bincount(assignments, minlength=num_prototypes).to(
        dtype=torch.float32
    )
    total = counts.sum().clamp_min(1.0)
    return counts / total


def prototype_entropy(probabilities: torch.Tensor) -> float:
    """Compute entropy for a normalized prototype-usage distribution."""

    probabilities = probabilities.clamp_min(1.0e-8)
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.item())


def embedding_norm(embeddings: torch.Tensor | None) -> float:
    """Average norm of the monitored student embeddings."""

    if embeddings is None or embeddings.numel() == 0:
        return 0.0
    return float(embeddings.norm(dim=-1).mean().item())


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib Figure to an HxWx3 uint8 numpy array (RGB)."""

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf[:, :, :3].copy()


def plot_prototype_usage(
    logits: torch.Tensor, num_prototypes: int, step: int
) -> np.ndarray:
    """Create a sorted rank-frequency bar chart of prototype assignments.

    Each point is assigned to its highest-logit prototype.  The bar chart
    shows how many points fall into each prototype, sorted from most to
    least occupied.  A log-scale y-axis makes it easy to spot dead
    prototypes (flat tail at zero) and collapse (single dominant bar).
    """

    assignments = logits.argmax(dim=-1)
    counts = torch.bincount(assignments, minlength=num_prototypes).cpu().numpy()
    sorted_counts = np.sort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(sorted_counts)), sorted_counts, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Prototype rank (sorted by occupancy)")
    ax.set_ylabel("Number of assigned points")
    ax.set_title(f"Prototype usage — step {step}")
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_cosine_similarity_histogram(
    cosine_similarities: torch.Tensor, step: int
) -> np.ndarray:
    """Histogram of per-point teacher-student cosine similarity.

    Shows how well the student EMA model tracks the teacher on matched
    point pairs.  A distribution peaked near 1.0 means the student is
    closely following the teacher; a broad or left-shifted distribution
    indicates misalignment.
    """

    sims = cosine_similarities.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(sims, bins=50, range=(-1.0, 1.0), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Cosine similarity (teacher vs student)")
    ax.set_ylabel("Number of matched points")
    ax.set_title(f"Teacher-student alignment — step {step}")
    ax.axvline(
        sims.mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"mean={sims.mean():.3f}",
    )
    ax.legend()
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_views_and_mask(
    global_origin_coord: torch.Tensor,
    global_offset: torch.Tensor,
    global_mask: torch.Tensor,
    local_origin_coord: torch.Tensor,
    local_offset: torch.Tensor,
    step: int,
) -> np.ndarray:
    """3D scatter plot of the first event's global view, mask, and local views.

    - Global view points are shown in light grey.
    - Masked (held-out) points are overlaid in red.
    - Local view points are shown in blue.

    Uses origin_coord (un-augmented positions) so the plot reflects the
    true detector geometry.  A fixed viewing angle makes images comparable
    across steps.
    """

    g_end = global_offset[0].item()
    l_end = local_offset[0].item()
    g_coords = global_origin_coord[:g_end].cpu().numpy()
    l_coords = local_origin_coord[:l_end].cpu().numpy()
    g_mask = global_mask[:g_end].cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        g_coords[:, 0],
        g_coords[:, 1],
        g_coords[:, 2],
        c="lightgrey",
        s=1,
        alpha=0.3,
        label="Global view",
    )
    if g_mask.any():
        ax.scatter(
            g_coords[g_mask, 0],
            g_coords[g_mask, 1],
            g_coords[g_mask, 2],
            c="red",
            s=2,
            alpha=0.5,
            label="Masked",
        )
    ax.scatter(
        l_coords[:, 0],
        l_coords[:, 1],
        l_coords[:, 2],
        c="blue",
        s=3,
        alpha=0.6,
        label="Local views",
    )

    ax.view_init(elev=20, azim=45)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Views + mask — step {step}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig_to_numpy(fig)


def save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    is_best: bool = False,
    step_tag: str | None = None,
) -> Path:
    """Write epoch, optimizer, and model state to the run directory."""

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if step_tag is not None:
        checkpoint_path = checkpoints_dir / f"step_{global_step:08d}.pt"
    else:
        checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_path)
    if is_best:
        torch.save(payload, checkpoints_dir / "best.pt")
    torch.save(payload, checkpoints_dir / "latest.pt")
    return checkpoint_path


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    lr_scheduler: CosineAnnealingLR | None,
    grad_scaler: Any,
    mixed_precision_dtype: torch.dtype | None,
    max_batches: int,
    view_config: DictConfig,
    phase: str,
    logger: Any = None,
    log_every_n_steps: int | None = None,
    checkpoint_every_n_steps: int | None = None,
    viz_every_n_steps: int | None = None,
    run_dir: Path | None = None,
    epoch_index: int = 0,
    global_step_offset: int = 0,
) -> tuple[dict[str, float], int]:
    """Run one train or validation epoch over a bounded number of batches.

    When *logger* and *log_every_n_steps* are provided, running metrics are
    emitted every N training batches so that long epochs are visible in
    comet / jsonl logs without waiting for the epoch to finish.

    When *checkpoint_every_n_steps*, *run_dir*, *optimizer* are all provided,
    a step checkpoint is saved every N training batches.

    When *viz_every_n_steps* and *logger* are provided, diagnostic images
    (prototype usage histogram, teacher-student cosine similarity, 3D
    view+mask plot) are logged every N training batches.
    """

    is_training = optimizer is not None
    autocast_enabled = mixed_precision_dtype is not None and device.type == "cuda"
    model.train(mode=is_training)
    totals = {
        "loss": 0.0,
        "prototype_entropy": 0.0,
        "embedding_norm": 0.0,
        "masked_fraction": 0.0,
        "data_wait_seconds": 0.0,
        "view_build_seconds": 0.0,
        "model_step_seconds": 0.0,
    }
    processed_batches = 0
    processed_events = 0
    last_logged_step = global_step_offset
    last_checkpoint_step = global_step_offset
    last_viz_step = global_step_offset

    progress_bar = tqdm(
        range(max_batches),
        total=max_batches,
        desc=phase,
        leave=True,
        dynamic_ncols=False,
        ascii=True,
    )

    data_iter = iter(dataloader)

    for batch_index in progress_bar:
        data_wait_start = time.perf_counter()
        try:
            events = next(data_iter)
        except StopIteration:
            break
        data_wait_seconds = time.perf_counter() - data_wait_start

        view_start = time.perf_counter()
        model_inputs = build_sonata_batch(
            events,
            device=device,
            max_calo_hits=view_config.max_calo_hits,
            grid_size=float(getattr(model, "grid_size", 0.002)),
            coord_noise_scale=view_config.coord_noise_scale,
            feat_noise_scale=view_config.energy_jitter_scale,
            point_dropout=view_config.point_dropout,
            num_global_views=view_config.num_global_views,
            num_local_views=view_config.num_local_views,
            global_crop_min_ratio=view_config.global_crop_min_ratio,
            global_crop_max_ratio=view_config.global_crop_max_ratio,
            local_crop_min_ratio=view_config.local_crop_min_ratio,
            local_crop_max_ratio=view_config.local_crop_max_ratio,
            coord_center=view_config.coord_center,
            coord_scale=view_config.coord_scale,
            energy_transform=view_config.energy_transform,
            energy_min=view_config.energy_min,
            energy_max=view_config.energy_max,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        view_build_seconds = time.perf_counter() - view_start

        model_step_start = time.perf_counter()

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=mixed_precision_dtype or torch.float32,
                enabled=autocast_enabled,
            ):
                if is_training:
                    model.step_schedules()
                result_dict = model(model_inputs)
                loss = result_dict["loss"]
                monitor_state = getattr(model, "last_monitoring_state", {})
                monitor_logits = monitor_state.get("student_logits")
                monitor_embeddings = monitor_state.get("point_features")
                masked_fraction = float(monitor_state.get("masked_fraction", 0.0))

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None and grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            model.update_teacher(momentum=None)

        if device.type == "cuda":
            torch.cuda.synchronize()
        model_step_seconds = time.perf_counter() - model_step_start

        if monitor_logits is None:
            monitor_logits = loss.new_zeros(
                (1, int(getattr(model, "num_prototypes", 1)))
            )
        usage = prototype_usage(
            monitor_logits, num_prototypes=int(getattr(model, "num_prototypes", 1))
        )
        totals["loss"] += float(loss.item())
        totals["prototype_entropy"] += prototype_entropy(usage)
        totals["embedding_norm"] += embedding_norm(monitor_embeddings)
        totals["masked_fraction"] += float(masked_fraction)
        totals["data_wait_seconds"] += data_wait_seconds
        totals["view_build_seconds"] += view_build_seconds
        totals["model_step_seconds"] += model_step_seconds
        processed_batches += 1
        processed_events += len(events)

        progress_bar.set_postfix(
            data=f"{data_wait_seconds:.1f}s",
            view=f"{view_build_seconds:.1f}s",
            model=f"{model_step_seconds:.1f}s",
            loss=f"{loss.item():.4f}",
            masked=f"{masked_fraction:.3f}",
        )

        current_absolute_step = global_step_offset + processed_batches
        just_logged_scalars = False

        if (
            is_training
            and logger is not None
            and log_every_n_steps is not None
            and log_every_n_steps > 0
            and current_absolute_step - last_logged_step >= log_every_n_steps
        ):
            running = {
                f"{phase}_loss_running": totals["loss"] / processed_batches,
                f"{phase}_prototype_entropy_running": totals["prototype_entropy"]
                / processed_batches,
                f"{phase}_embedding_norm_running": totals["embedding_norm"]
                / processed_batches,
                f"{phase}_masked_fraction_running": totals["masked_fraction"]
                / processed_batches,
                "learning_rate": learning_rate(optimizer),
                "epoch": epoch_index + 1,
                "mask_size": float(getattr(model, "mask_size", 0.0)),
                "mask_ratio": float(getattr(model, "mask_ratio", 0.0)),
                "teacher_temperature": float(getattr(model, "teacher_temp", 0.07)),
                "teacher_momentum": float(getattr(model, "momentum", 0.994)),
            }
            logger.log_metrics(running, step=current_absolute_step)
            last_logged_step = current_absolute_step
            just_logged_scalars = True

        if (
            is_training
            and checkpoint_every_n_steps is not None
            and checkpoint_every_n_steps > 0
            and run_dir is not None
            and current_absolute_step - last_checkpoint_step >= checkpoint_every_n_steps
        ):
            running_metrics = {
                key: value / processed_batches for key, value in totals.items()
            }
            checkpoint_path = save_checkpoint(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch_index + 1,
                global_step=current_absolute_step,
                metrics=running_metrics,
                step_tag="step",
            )
            print(
                f"Saved step checkpoint (step {current_absolute_step}): {checkpoint_path}"
            )
            last_checkpoint_step = current_absolute_step

        should_log_images = is_training and logger is not None and just_logged_scalars
        should_viz = (
            is_training
            and logger is not None
            and viz_every_n_steps is not None
            and viz_every_n_steps > 0
            and current_absolute_step - last_viz_step >= viz_every_n_steps
        )

        if should_log_images or should_viz:
            num_protos = int(getattr(model, "num_prototypes", 1))
            image = plot_prototype_usage(
                monitor_logits, num_protos, current_absolute_step
            )
            logger.log_image("prototype_usage", image, step=current_absolute_step)

            monitor_state = getattr(model, "last_monitoring_state", {})
            cos_sims = monitor_state.get("cosine_similarities")
            if cos_sims is not None and cos_sims.numel() > 0:
                image = plot_cosine_similarity_histogram(
                    cos_sims, current_absolute_step
                )
                logger.log_image("cosine_similarity", image, step=current_absolute_step)

            if should_viz:
                global_mask = monitor_state.get("global_mask")
                if (
                    global_mask is not None
                    and model_inputs is not None
                    and "global_origin_coord" in model_inputs
                ):
                    image = plot_views_and_mask(
                        global_origin_coord=model_inputs["global_origin_coord"],
                        global_offset=model_inputs["global_offset"],
                        global_mask=global_mask,
                        local_origin_coord=model_inputs["local_origin_coord"],
                        local_offset=model_inputs["local_offset"],
                        step=current_absolute_step,
                    )
                    logger.log_image("views_mask", image, step=current_absolute_step)
                last_viz_step = current_absolute_step

    progress_bar.close()

    if processed_batches == 0:
        raise ValueError(
            f"No {phase} batches were processed. Check the chosen split and max batch count."
        )

    averaged_metrics = {key: value / processed_batches for key, value in totals.items()}
    averaged_metrics["events_per_second"] = processed_events / max(
        1.0e-6,
        totals["data_wait_seconds"]
        + totals["view_build_seconds"]
        + totals["model_step_seconds"],
    )
    return averaged_metrics, processed_batches


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)
    training_config = config.training
    view_config = config.views
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print(
            "Training requires a CUDA-enabled environment because the current PTv3/spconv stack is GPU-only."
        )
        print("Run this script on a GPU node, for example through a SLURM job.")
        return

    train_loader = create_dataloader(
        config, training_config.train_split, shuffle=training_config.train_shuffle
    )
    val_loader = create_dataloader(config, training_config.val_split, shuffle=False)
    max_train_batches = resolve_epoch_batch_limit(
        train_loader, training_config.max_train_batches, "train"
    )
    max_val_batches = resolve_epoch_batch_limit(
        val_loader, training_config.max_val_batches, "val"
    )

    run_dir, run_name = ensure_run_directory(
        PROJECT_ROOT,
        run_dir=training_config.get("run_dir"),
        run_name=training_config.get("run_name"),
    )
    logger = create_experiment_logger(
        training_config.log_backend, run_dir=run_dir, run_name=run_name
    )
    run_config = to_plain_container(config) | {
        "device": str(device),
        "run_dir": str(run_dir),
        "run_name": run_name,
    }
    print(f"Run directory: {run_dir}")

    mixed_precision_dtype = resolve_mixed_precision_dtype(training_config, device)

    model = create_training_model(
        device=device,
        **model_factory_kwargs(select_model_config(config, "training")),
    )
    model.setup_schedulers(
        total_steps=max(1, training_config.num_epochs * max_train_batches)
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=mixed_precision_dtype is torch.float16
    )
    print(f"Mixed precision: {mixed_precision_name(mixed_precision_dtype)}")
    print(
        f"Flash attention: {model.flash_attention_enabled} ({model.flash_attention_backend})"
    )
    run_config["resolved_mixed_precision"] = mixed_precision_name(mixed_precision_dtype)
    run_config["resolved_flash_attention"] = bool(model.flash_attention_enabled)
    run_config["resolved_flash_attention_backend"] = model.flash_attention_backend
    config_path = write_run_config(run_dir, run_config)
    logger.log_params(run_config)
    print(f"Run config: {config_path}")
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_config.num_epochs * max_train_batches),
        eta_min=training_config.min_learning_rate,
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    global_step = 0
    best_val_loss = float("inf")
    log_every_n_steps = int(training_config.get("log_every_n_steps", 500))
    checkpoint_every_n_steps = (
        int(training_config.get("checkpoint_every_n_steps", 0)) or None
    )
    viz_every_n_steps = int(training_config.get("viz_every_n_steps", 0)) or None

    try:
        for epoch in range(training_config.num_epochs):
            print(f"Epoch {epoch + 1}/{training_config.num_epochs}")
            epoch_start = time.perf_counter()

            current_momentum = float(getattr(model, "momentum", 0.994))
            current_temperature = float(getattr(model, "teacher_temp", 0.07))

            train_metrics, train_batches = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_scaler=grad_scaler,
                mixed_precision_dtype=mixed_precision_dtype,
                max_batches=max_train_batches,
                view_config=view_config,
                phase="train",
                logger=logger,
                log_every_n_steps=log_every_n_steps,
                checkpoint_every_n_steps=checkpoint_every_n_steps,
                viz_every_n_steps=viz_every_n_steps,
                run_dir=run_dir,
                epoch_index=epoch,
                global_step_offset=global_step,
            )
            global_step += train_batches

            val_metrics, _ = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                lr_scheduler=None,
                grad_scaler=None,
                mixed_precision_dtype=mixed_precision_dtype,
                max_batches=max_val_batches,
                view_config=view_config,
                phase="val",
            )

            epoch_time_seconds = time.perf_counter() - epoch_start
            current_momentum = float(getattr(model, "momentum", current_momentum))
            current_temperature = float(
                getattr(model, "teacher_temp", current_temperature)
            )
            epoch_metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_prototype_entropy": train_metrics["prototype_entropy"],
                "val_prototype_entropy": val_metrics["prototype_entropy"],
                "train_embedding_norm": train_metrics["embedding_norm"],
                "val_embedding_norm": val_metrics["embedding_norm"],
                "train_masked_fraction": train_metrics["masked_fraction"],
                "val_masked_fraction": val_metrics["masked_fraction"],
                "train_data_wait_seconds": train_metrics["data_wait_seconds"],
                "val_data_wait_seconds": val_metrics["data_wait_seconds"],
                "train_view_build_seconds": train_metrics["view_build_seconds"],
                "val_view_build_seconds": val_metrics["view_build_seconds"],
                "train_model_step_seconds": train_metrics["model_step_seconds"],
                "val_model_step_seconds": val_metrics["model_step_seconds"],
                "train_events_per_second": train_metrics["events_per_second"],
                "val_events_per_second": val_metrics["events_per_second"],
                "learning_rate": learning_rate(optimizer),
                "teacher_momentum": current_momentum,
                "teacher_temperature": current_temperature,
                "epoch_time_seconds": epoch_time_seconds,
            }
            logger.log_metrics(epoch_metrics, step=global_step)
            print("epoch summary: " + json.dumps(epoch_metrics, sort_keys=True))

            is_best = epoch_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = epoch_metrics["val_loss"]
            checkpoint_path = save_checkpoint(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                metrics=epoch_metrics,
                is_best=is_best,
            )
            print(f"Saved checkpoint: {checkpoint_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
