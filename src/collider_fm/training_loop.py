"""Ray Train worker loop, checkpoint I/O, and epoch runner for Sonata training."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ray.train
import ray.train.torch
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ColliderMLDataset, collate_fn
from .experiment_logging import (
    NullLogger,
    create_experiment_logger,
    write_run_config,
)
from .model import create_training_model
from .project_config import (
    model_factory_kwargs,
    select_model_config,
    sonata_batch_kwargs,
    to_plain_container,
)
from .sonata_model import CosineScheduler
from .views import build_sonata_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a DDP-wrapped model to access the underlying module."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def is_main_process() -> bool:
    """True on rank 0 (or in non-distributed mode)."""
    return ray.train.get_context().get_world_rank() == 0


def learning_rate(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def mixed_precision_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "none"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return str(dtype)


def resolve_mixed_precision_dtype(
    training_config: DictConfig, device: torch.device
) -> torch.dtype | None:
    mode = str(training_config.get("mixed_precision", "none")).lower()
    if mode == "none":
        return None
    if device.type != "cuda":
        return None
    if mode in {"bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    if mode in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported mixed_precision: {mode}")


def resolve_epoch_batch_limit(
    dataloader: DataLoader, requested_max_batches: int | None, phase: str
) -> int:
    total_batches = len(dataloader)
    if total_batches <= 0:
        raise ValueError(f"The {phase} dataloader produced zero batches.")
    if requested_max_batches is None:
        return total_batches
    if requested_max_batches <= 0:
        raise ValueError(
            f"{phase} max_batches must be positive or None, got {requested_max_batches}."
        )
    return min(total_batches, requested_max_batches)


def build_optimizer_param_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Build optimizer param groups with WD exclusion for bias/norm/1D params.

    Matches pimm's WeightDecayExclusion: bias, norm, gamma, token, and 1D
    parameters are excluded from weight decay.
    """
    base = unwrap_model(model)
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    for name, param in base.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "gamma" in name.lower()
            or "token" in name.lower()
            or param.ndim == 1
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay, "apply_wd": True},
        {"params": no_decay_params, "weight_decay": 0.0, "apply_wd": False},
    ]


def step_weight_decay(optimizer: AdamW, wd_scheduler: CosineScheduler) -> float:
    """Advance the WD scheduler and update applicable param groups."""
    wd = wd_scheduler.step()
    for group in optimizer.param_groups:
        if group.get("apply_wd", False):
            group["weight_decay"] = wd
    return wd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def prototype_usage(logits: torch.Tensor, num_prototypes: int) -> torch.Tensor:
    assignments = logits.argmax(dim=-1)
    counts = torch.bincount(assignments, minlength=num_prototypes).to(dtype=torch.float32)
    return counts / counts.sum().clamp_min(1.0)


def prototype_entropy(probabilities: torch.Tensor) -> float:
    p = probabilities.clamp_min(1.0e-8)
    return float(-(p * p.log()).sum().item())


def embedding_norm(embeddings: torch.Tensor | None) -> float:
    if embeddings is None or embeddings.numel() == 0:
        return 0.0
    return float(embeddings.norm(dim=-1).mean().item())


def reduce_scalar(value: float, device: torch.device) -> float:
    """All-reduce a scalar across ranks and return the sum."""
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor)
        return float(tensor.item())
    return value


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf[:, :, :3].copy()


def plot_prototype_usage(logits: torch.Tensor, num_prototypes: int, step: int) -> np.ndarray:
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


def plot_cosine_similarity_histogram(sims: torch.Tensor, step: int) -> np.ndarray:
    sims_np = sims.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(sims_np, bins=50, range=(-1.0, 1.0), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Cosine similarity (teacher vs student)")
    ax.set_ylabel("Number of matched points")
    ax.set_title(f"Teacher-student alignment — step {step}")
    ax.axvline(sims_np.mean(), color="red", linestyle="--", linewidth=1,
               label=f"mean={sims_np.mean():.3f}")
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
    g_end = global_offset[0].item()
    l_end = local_offset[0].item()
    g_coords = global_origin_coord[:g_end].cpu().numpy()
    l_coords = local_origin_coord[:l_end].cpu().numpy()
    g_mask = global_mask[:g_end].cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(g_coords[:, 0], g_coords[:, 1], g_coords[:, 2],
               c="lightgrey", s=1, alpha=0.3, label="Global view")
    if g_mask.any():
        ax.scatter(g_coords[g_mask, 0], g_coords[g_mask, 1], g_coords[g_mask, 2],
                   c="red", s=2, alpha=0.5, label="Masked")
    ax.scatter(l_coords[:, 0], l_coords[:, 1], l_coords[:, 2],
               c="blue", s=3, alpha=0.6, label="Local views")
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Views + mask — step {step}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig_to_numpy(fig)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

CHECKPOINT_FILES = {
    "model": "model.pt",
    "optimizer": "optimizer.pt",
    "lr_scheduler": "scheduler.pt",
    "scaler": "scaler.pt",
    "training_state": "training_state.pt",
}


def save_checkpoint_to_dir(
    directory: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    lr_scheduler: OneCycleLR,
    grad_scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    wd_scheduler_iter: int | None = None,
) -> None:
    """Write the canonical checkpoint payload into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    base_model = unwrap_model(model)
    torch.save(base_model.state_dict(), directory / CHECKPOINT_FILES["model"])
    torch.save(optimizer.state_dict(), directory / CHECKPOINT_FILES["optimizer"])
    torch.save(lr_scheduler.state_dict(), directory / CHECKPOINT_FILES["lr_scheduler"])
    if grad_scaler is not None:
        torch.save(grad_scaler.state_dict(), directory / CHECKPOINT_FILES["scaler"])
    sonata_state = {}
    for attr in ("mask_size", "mask_ratio", "teacher_temp", "momentum", "mask_jitter"):
        val = getattr(base_model, attr, None)
        if val is not None:
            sonata_state[attr] = float(val)
    for sched_name in ("mask_size_scheduler", "mask_ratio_scheduler",
                       "teacher_temp_scheduler", "momentum_scheduler",
                       "mask_jitter_scheduler"):
        sched = getattr(base_model, sched_name, None)
        if sched is not None:
            sonata_state[f"{sched_name}_iter"] = sched.iter
    training_state: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "sonata_state": sonata_state,
    }
    if wd_scheduler_iter is not None:
        training_state["wd_scheduler_iter"] = wd_scheduler_iter
    torch.save(
        training_state,
        directory / CHECKPOINT_FILES["training_state"],
    )


def load_checkpoint_from_dir(
    directory: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    lr_scheduler: OneCycleLR,
    grad_scaler: torch.amp.GradScaler | None,
) -> tuple[int, int, float, int | None]:
    """Restore model/optimizer/scheduler/scaler from *directory*.

    Returns (epoch, global_step, best_val_loss, wd_scheduler_iter).
    """
    base_model = unwrap_model(model)
    base_model.load_state_dict(
        torch.load(directory / CHECKPOINT_FILES["model"], map_location="cpu", weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(directory / CHECKPOINT_FILES["optimizer"], map_location="cpu", weights_only=True)
    )
    lr_scheduler.load_state_dict(
        torch.load(directory / CHECKPOINT_FILES["lr_scheduler"], map_location="cpu", weights_only=True)
    )
    scaler_path = directory / CHECKPOINT_FILES["scaler"]
    if grad_scaler is not None and scaler_path.exists():
        grad_scaler.load_state_dict(
            torch.load(scaler_path, map_location="cpu", weights_only=True)
        )
    training_state = torch.load(
        directory / CHECKPOINT_FILES["training_state"], map_location="cpu", weights_only=True
    )
    sonata_state = training_state.get("sonata_state", {})
    base_model = unwrap_model(model)
    for attr in ("mask_size", "mask_ratio", "teacher_temp", "momentum", "mask_jitter"):
        if attr in sonata_state:
            setattr(base_model, attr, sonata_state[attr])
    for sched_name in ("mask_size_scheduler", "mask_ratio_scheduler",
                       "teacher_temp_scheduler", "momentum_scheduler",
                       "mask_jitter_scheduler"):
        key = f"{sched_name}_iter"
        if key in sonata_state:
            sched = getattr(base_model, sched_name, None)
            if sched is not None:
                sched.iter = sonata_state[key]
    return (
        training_state["epoch"],
        training_state["global_step"],
        training_state["best_val_loss"],
        training_state.get("wd_scheduler_iter"),
    )


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_dataloader(config: DictConfig, split: str, shuffle: bool) -> DataLoader:
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
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": training_config.batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": training_config.num_workers,
        "pin_memory": bool(training_config.pin_memory and torch.cuda.is_available()),
    }
    if training_config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = training_config.prefetch_factor
    return DataLoader(dataset, **dataloader_kwargs)


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    lr_scheduler: OneCycleLR | None,
    grad_scaler: Any,
    mixed_precision_dtype: torch.dtype | None,
    max_batches: int,
    batch_kwargs: dict[str, Any],
    phase: str,
    logger: Any = None,
    log_every_n_steps: int | None = None,
    viz_every_n_steps: int | None = None,
    epoch_index: int = 0,
    global_step_offset: int = 0,
    clip_grad: float = float("inf"),
    wd_scheduler: CosineScheduler | None = None,
) -> tuple[dict[str, float], int]:
    """Run one train or validation epoch, rank-safe under DDP."""

    world_size = ray.train.get_context().get_world_size()
    is_training = optimizer is not None
    autocast_enabled = mixed_precision_dtype is not None and device.type == "cuda"
    base_model = unwrap_model(model)
    base_model.train(mode=is_training)

    totals = {
        "loss": 0.0,
        "prototype_entropy": 0.0,
        "embedding_norm": 0.0,
        "masked_fraction": 0.0,
    }
    processed_batches = 0
    last_logged_step = global_step_offset
    last_viz_step = global_step_offset

    progress_bar = (
        tqdm(range(max_batches), total=max_batches, desc=phase,
             leave=True, dynamic_ncols=False, ascii=True)
        if is_main_process()
        else range(max_batches)
    )

    data_iter = iter(dataloader)
    for _batch_index in progress_bar:
        try:
            events = next(data_iter)
        except StopIteration:
            break

        model_inputs = build_sonata_batch(events, device=device, **batch_kwargs)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=mixed_precision_dtype or torch.float32,
                enabled=autocast_enabled,
            ):
                if is_training:
                    base_model.step_schedules()
                result_dict = model(model_inputs)
                loss = result_dict["loss"]
                monitor_state = getattr(base_model, "last_monitoring_state", {})
                monitor_logits = monitor_state.get("student_logits")
                monitor_embeddings = monitor_state.get("point_features")
                masked_fraction = float(monitor_state.get("masked_fraction", 0.0))
                mask_match_fraction = float(monitor_state.get("mask_match_fraction", 0.0))
                roll_match_fraction = float(monitor_state.get("roll_match_fraction", 0.0))
                unmask_match_fraction = float(monitor_state.get("unmask_match_fraction", 0.0))
                batch_mask_loss = float(result_dict.get("mask_loss", 0.0))
                batch_roll_mask_loss = float(result_dict.get("roll_mask_loss", 0.0))
                batch_unmask_loss = float(result_dict.get("unmask_loss", 0.0))
                cos_sims = monitor_state.get("cosine_similarities")
                batch_cosine_sim = float(cos_sims.mean().item()) if cos_sims is not None and cos_sims.numel() > 0 else 0.0

        batch_grad_norm = 0.0
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if wd_scheduler is not None:
                step_weight_decay(optimizer, wd_scheduler)
            if grad_scaler is not None and grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                batch_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad,
                ).item())
                grad_scaler.step(optimizer)
                old_scale = grad_scaler.get_scale()
                grad_scaler.update()
                if old_scale <= grad_scaler.get_scale() and lr_scheduler is not None:
                    lr_scheduler.step()
            else:
                loss.backward()
                batch_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad,
                ).item())
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            base_model.update_teacher(momentum=None)

        num_prototypes = int(getattr(base_model, "num_prototypes",
                                     monitor_logits.shape[-1]
                                     if monitor_logits is not None and monitor_logits.ndim == 2
                                     else 1))
        if monitor_logits is None:
            monitor_logits = loss.new_zeros((1, num_prototypes))
        usage = prototype_usage(monitor_logits, num_prototypes=num_prototypes)
        totals["loss"] += float(loss.item())
        totals["prototype_entropy"] += prototype_entropy(usage)
        totals["embedding_norm"] += embedding_norm(monitor_embeddings)
        totals["masked_fraction"] += float(masked_fraction)
        processed_batches += 1

        if is_main_process() and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", masked=f"{masked_fraction:.3f}")

        current_absolute_step = global_step_offset + processed_batches

        if (
            is_training
            and logger is not None
            and log_every_n_steps is not None
            and log_every_n_steps > 0
            and current_absolute_step - last_logged_step >= log_every_n_steps
        ):
            step_metrics = {
                "record_type": "step_metrics",
                "train_loss": float(loss.item()),
                "train_prototype_entropy": prototype_entropy(usage),
                "train_embedding_norm": embedding_norm(monitor_embeddings),
                "train_masked_fraction": masked_fraction,
                "train_mask_match_fraction": mask_match_fraction,
                "train_roll_match_fraction": roll_match_fraction,
                "train_unmask_match_fraction": unmask_match_fraction,
                "train_mask_loss": batch_mask_loss,
                "train_roll_mask_loss": batch_roll_mask_loss,
                "train_unmask_loss": batch_unmask_loss,
                "train_cosine_similarity": batch_cosine_sim,
                "train_gradient_norm": batch_grad_norm,
                "learning_rate": learning_rate(optimizer),
                "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
                "epoch": epoch_index,
                "mask_size": float(getattr(base_model, "mask_size", 0.0)),
                "mask_ratio": float(getattr(base_model, "mask_ratio", 0.0)),
                "teacher_temperature": float(getattr(base_model, "teacher_temp", 0.07)),
                "teacher_momentum": float(getattr(base_model, "momentum", 0.994)),
            }
            logger.log_metrics(step_metrics, step=current_absolute_step)
            last_logged_step = current_absolute_step

        # Periodic visualization (rank 0 only)
        should_viz = (
            is_training
            and logger is not None
            and viz_every_n_steps is not None
            and viz_every_n_steps > 0
            and current_absolute_step - last_viz_step >= viz_every_n_steps
        )
        if should_viz:
            image = plot_prototype_usage(monitor_logits, num_prototypes, current_absolute_step)
            logger.log_image("prototype_usage", image, step=current_absolute_step)
            monitor_state = getattr(base_model, "last_monitoring_state", {})
            cos_sims = monitor_state.get("cosine_similarities")
            if cos_sims is not None and cos_sims.numel() > 0:
                image = plot_cosine_similarity_histogram(cos_sims, current_absolute_step)
                logger.log_image("cosine_similarity", image, step=current_absolute_step)
            global_mask = monitor_state.get("global_mask")
            if global_mask is not None and "global_origin_coord" in model_inputs:
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

    if isinstance(progress_bar, tqdm):
        progress_bar.close()

    if processed_batches == 0:
        raise ValueError(f"No {phase} batches were processed.")

    # Reduce epoch metrics across ranks
    if world_size > 1:
        for key in totals:
            totals[key] = reduce_scalar(totals[key], device) / world_size
        processed_batches = int(reduce_scalar(float(processed_batches), device) / world_size)

    averaged_metrics = {key: value / max(1, processed_batches) for key, value in totals.items()}
    return averaged_metrics, processed_batches


# ---------------------------------------------------------------------------
# Ray Train worker entrypoint
# ---------------------------------------------------------------------------

def train_loop_per_worker(train_loop_config: dict) -> None:
    """Ray Train worker function — runs on each GPU rank."""
    config = OmegaConf.create(train_loop_config)
    training_config = config.training

    # Device & distributed context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    is_rank0 = world_rank == 0

    # Seed
    seed = int(training_config.get("seed", 42))
    torch.manual_seed(seed + world_rank)

    # Dataloaders — prepare_data_loader adds DistributedSampler and
    # sets move_to_device=False because our collate returns raw event lists.
    train_loader = create_dataloader(config, training_config.train_split, shuffle=True)
    val_loader = create_dataloader(config, training_config.val_split, shuffle=False)
    train_loader = ray.train.torch.prepare_data_loader(train_loader, move_to_device=False)
    val_loader = ray.train.torch.prepare_data_loader(val_loader, move_to_device=False)

    batch_kwargs = sonata_batch_kwargs(
        config, "training", max_calo_hits=config.views.max_calo_hits,
    )

    max_train_batches = resolve_epoch_batch_limit(
        train_loader, training_config.max_train_batches, "train"
    )
    max_val_batches = resolve_epoch_batch_limit(
        val_loader, training_config.max_val_batches, "val"
    )

    # Run directory & logging — only rank 0 creates real artifacts
    run_dir: Path | None = None
    logger: Any = NullLogger()
    if is_rank0:
        run_dir_value = training_config.get("run_dir")
        if run_dir_value is None:
            raise ValueError("training.run_dir must be resolved before entering train_loop_per_worker.")
        run_dir = Path(run_dir_value)
        run_name = run_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = create_experiment_logger(training_config.log_backend, run_dir=run_dir)
        run_config_dict = to_plain_container(config) | {
            "device": str(device),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "world_size": world_size,
        }
        write_run_config(run_dir, run_config_dict)
        logger.log_params(run_config_dict)
        # Write a hint so downstream tools can find the Ray checkpoint directory
        storage_path = training_config.get(
            "ray_storage_path", "/mnt/ceph/users/ewulff/raytrain_results/"
        )
        (run_dir / "checkpoint_path.txt").write_text(
            str(Path(storage_path) / run_name) + "\n"
        )

    mixed_precision_dtype = resolve_mixed_precision_dtype(training_config, device)

    # Model
    model = create_training_model(
        device=device,
        **model_factory_kwargs(select_model_config(config, "training")),
    )
    model = ray.train.torch.prepare_model(
        model,
        parallel_strategy="ddp",
        parallel_strategy_kwargs={"find_unused_parameters": True},
    )
    base_model = unwrap_model(model)

    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=mixed_precision_dtype is torch.float16
    )
    if is_rank0:
        print(f"Mixed precision: {mixed_precision_name(mixed_precision_dtype)}")
        print(f"Flash attention: {base_model.flash_attention_enabled} ({base_model.flash_attention_backend})")
        print(f"World size: {world_size}")

    # Optimizer & LR scheduler
    param_groups = build_optimizer_param_groups(model, float(training_config.weight_decay))
    optimizer = AdamW(
        param_groups,
        lr=float(training_config.learning_rate),
    )
    total_steps = max(1, int(training_config.num_epochs) * max_train_batches)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=float(training_config.learning_rate),
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
    )
    clip_grad = float(training_config.get("clip_grad", float("inf")))

    # Resume from Ray checkpoint if available
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    wd_scheduler_iter = 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            start_epoch, global_step, best_val_loss, wd_scheduler_iter = load_checkpoint_from_dir(
                Path(checkpoint_dir), model, optimizer, lr_scheduler, grad_scaler,
            )
        if is_rank0:
            print(f"Resumed from checkpoint: epoch={start_epoch}, step={global_step}")

    # Setup Sonata schedulers (always from scratch with correct total_steps)
    base_model.setup_schedulers(total_steps=total_steps, current_step=global_step)

    # Weight decay scheduler (cosine ramp 0.04 -> 0.2 over full training)
    wd_scheduler = CosineScheduler(
        base_value=float(training_config.weight_decay),
        final_value=float(training_config.final_weight_decay),
        total_iters=total_steps,
    )
    wd_scheduler.iter = wd_scheduler_iter or 0

    # Re-apply sonata state from checkpoint on top of the fresh schedulers
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            ts = torch.load(
                Path(checkpoint_dir) / CHECKPOINT_FILES["training_state"],
                map_location="cpu", weights_only=True,
            )
            sonata_state = ts.get("sonata_state", {})
            for attr in ("mask_size", "mask_ratio", "teacher_temp", "momentum", "mask_jitter"):
                if attr in sonata_state:
                    setattr(base_model, attr, sonata_state[attr])
            for sched_name in ("mask_size_scheduler", "mask_ratio_scheduler",
                               "teacher_temp_scheduler", "momentum_scheduler",
                               "mask_jitter_scheduler"):
                key = f"{sched_name}_iter"
                if key in sonata_state:
                    sched = getattr(base_model, sched_name, None)
                    if sched is not None:
                        sched.iter = sonata_state[key]

    log_every_n_steps = int(training_config.get("log_every_n_steps", 500))
    viz_every_n_steps = int(training_config.get("viz_every_n_steps", 0)) or None

    # ---- Training loop ----
    try:
        for epoch in range(start_epoch, training_config.num_epochs):
            if is_rank0:
                print(f"Epoch {epoch}/{max(0, training_config.num_epochs - 1)}")

            # Set epoch on the DistributedSampler so shuffling differs per epoch
            if world_size > 1 and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_metrics, train_batches = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_scaler=grad_scaler,
                mixed_precision_dtype=mixed_precision_dtype,
                max_batches=max_train_batches,
                batch_kwargs=batch_kwargs,
                phase="train",
                logger=logger,
                log_every_n_steps=log_every_n_steps,
                viz_every_n_steps=viz_every_n_steps,
                epoch_index=epoch,
                global_step_offset=global_step,
                clip_grad=clip_grad,
                wd_scheduler=wd_scheduler,
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
                batch_kwargs=batch_kwargs,
                phase="val",
            )

            current_momentum = float(getattr(base_model, "momentum", 0.994))
            current_temperature = float(getattr(base_model, "teacher_temp", 0.07))
            epoch_metrics = {
                "record_type": "epoch_metrics",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_prototype_entropy": train_metrics["prototype_entropy"],
                "val_prototype_entropy": val_metrics["prototype_entropy"],
                "train_embedding_norm": train_metrics["embedding_norm"],
                "val_embedding_norm": val_metrics["embedding_norm"],
                "train_masked_fraction": train_metrics["masked_fraction"],
                "val_masked_fraction": val_metrics["masked_fraction"],
                "learning_rate": learning_rate(optimizer),
                "teacher_momentum": current_momentum,
                "teacher_temperature": current_temperature,
            }

            if is_rank0:
                logger.log_metrics(epoch_metrics, step=global_step, epoch=epoch)
                print("epoch summary: " + json.dumps(epoch_metrics, sort_keys=True))

            # Checkpoint: rank 0 writes, all workers report
            is_best = epoch_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = epoch_metrics["val_loss"]

            with tempfile.TemporaryDirectory() as tmpdir:
                ray_checkpoint = None
                if is_rank0:
                    save_checkpoint_to_dir(
                        Path(tmpdir), model, optimizer, lr_scheduler, grad_scaler,
                        epoch=epoch + 1,  # + 1 because training_state.epoch is the next epoch to run
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        wd_scheduler_iter=wd_scheduler.iter,
                    )
                    ray_checkpoint = ray.train.Checkpoint.from_directory(tmpdir)

                ray.train.report(
                    metrics={"val_loss": epoch_metrics["val_loss"]} if not is_rank0 else epoch_metrics,
                    checkpoint=ray_checkpoint,
                )

    finally:
        if is_rank0:
            logger.finish()
