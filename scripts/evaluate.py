"""Evaluate ColliderFM pretraining quality with label-free collapse metrics.

Loads a checkpoint (or runs a random-init baseline), encodes held-out validation
events through the deterministic EMA teacher backbone, and reports:

  * per-point participation ratio (effective rank) + singular-value spectrum  (collapse headline)
  * per-point prototype usage / entropy + dead-prototype count
  * per-event NN view-retrieval R@1/R@5 + alignment / uniformity (secondary lens)

Metrics are written to ``<training_run_dir>/eval/<checkpoint>/metrics_step.jsonl``
when evaluating a training checkpoint (placed inside the training run dir, in a
subfolder named after the checkpoint dir so the source is obvious), or to
``runs/eval_<stem>_<timestamp>/metrics_step.jsonl`` for random-init / raw ``.pt``
baselines. Pass ``evaluation.run_name=...`` to override.

Examples:
  uv run python scripts/evaluate.py evaluation.checkpoint=runs/myrun   # resolves latest checkpoint + matches the run's backbone
  uv run python scripts/evaluate.py evaluation.checkpoint=runs/myrun/model.pt   # raw .pt (no auto backbone match)
  uv run python scripts/evaluate.py                                   # random-init baseline
  uv run python scripts/evaluate.py evaluation.max_events=500 evaluation.val_split=val[:500]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.diagnostics import create_dataloader, load_checkpoint
from collider_fm.evaluation import collect_embeddings, summarize
from collider_fm.experiment_logging import JsonlLogger, write_run_config
from collider_fm.model import create_training_model
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    model_factory_kwargs,
    point_view_kwargs,
    select_model_config,
    sonata_batch_kwargs,
    to_plain_container,
)
from collider_fm._panda.utils import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the evaluate CLI.

    Returns:
        argparse.ArgumentParser: Configured parser with `--config` and dotlist
        override support.
    """
    return build_config_arg_parser(
        description=(
            "Evaluate ColliderFM pretraining with label-free collapse metrics " "(participation ratio, prototype usage, NN view-retrieval, alignment/uniformity)."
        ),
        epilog=(
            "Examples:\n"
            "  uv run python scripts/evaluate.py evaluation.checkpoint=runs/myrun/model.pt\n"
            "  uv run python scripts/evaluate.py evaluation.checkpoint=runs/myrun\n"
            "  uv run python scripts/evaluate.py  # random-init baseline\n"
            "  uv run python scripts/evaluate.py evaluation.max_events=500"
        ),
        config_sections=("data", "views", "model.training", "evaluation"),
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve a torch device, falling back to CPU if CUDA is unavailable.

    Args:
        device_name (str): Requested device name (e.g. `"cuda"` or `"cpu"`).

    Returns:
        torch.device: The resolved device.
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def _config_alongside(path: Path) -> Path | None:
    """Return ``config.json`` next to a training run dir (marked by checkpoint_path.txt)."""
    for cand in (path, *path.parents):
        if (cand / "checkpoint_path.txt").is_file() and (cand / "config.json").is_file():
            return cand / "config.json"
    return None


def _config_by_run_name(ray_dir: Path) -> Path | None:
    """Recover ``runs/<run_name>/config.json`` from a Ray storage directory name."""
    cfg = PROJECT_ROOT / "runs" / ray_dir.name / "config.json"
    return cfg if cfg.is_file() else None


def _best_checkpoint_dir(root: Path) -> Path | None:
    """Find the best checkpoint directory under ``root`` that holds a ``model.pt``.

    Prefers ``checkpoint_manager_snapshot.json`` (Ray's canonical, naming-scheme-
    agnostic record) and picks the entry with the lowest ``val_loss``. Falls
    back to lexicographic sort of ``checkpoint_*`` dirs (i.e. the latest), which
    is the best guess available without per-checkpoint metrics.

    Args:
        root (Path): Directory containing ``checkpoint_*`` subdirs and
            (optionally) ``checkpoint_manager_snapshot.json``.

    Returns:
        Path | None: Path to the best checkpoint dir containing ``model.pt``,
        or ``None`` if none is found.
    """
    snapshot = root / "checkpoint_manager_snapshot.json"
    if snapshot.is_file():
        try:
            data = json.loads(snapshot.read_text())
            candidates: list[tuple[float, Path]] = []
            for result in data.get("checkpoint_results", []):
                name = result.get("checkpoint_dir_name")
                val_loss = result.get("metrics", {}).get("val_loss")
                if name and val_loss is not None and (root / name / "model.pt").is_file():
                    candidates.append((float(val_loss), root / name))
            if candidates:
                return min(candidates)[1]
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    ckpts = sorted(p for p in root.glob("checkpoint_*") if p.is_dir() and (p / "model.pt").is_file())
    return ckpts[-1] if ckpts else None


def resolve_checkpoint(spec: Any) -> tuple[Path | None, Path | None]:
    """Resolve a checkpoint spec to `(model.pt path, run config.json path)`.

    Accepts `None`/empty (random-init baseline -> `(None, None)`), a `model.pt`
    file, a run directory (`runs/<run>` -> reads `checkpoint_path.txt` -> best
    `checkpoint_*/model.pt` by lowest val_loss), or a Ray storage directory
    containing `checkpoint_*` subdirs. The run `config.json` is recovered when
    the spec is a training run dir (`runs/<run>`) or a Ray storage dir whose
    name matches a run dir, so the model can be built to match the checkpoint's
    architecture exactly.

    Args:
        spec (Any): Checkpoint specification (None, path string, or Path).

    Returns:
        tuple[Path | None, Path | None]: `(model_pt_path, run_config_path)`.
        Both are None for a random-init baseline.
    """
    if spec is None or str(spec).strip() == "":
        return None, None
    path = Path(str(spec))

    if path.is_file():
        return path, _config_alongside(path)

    if path.is_dir():
        run_config = _config_alongside(path)
        pointer = path / "checkpoint_path.txt"
        search_roots = []
        if pointer.exists():
            ray_dir = Path(pointer.read_text().strip())
            if ray_dir.exists():
                search_roots.append(ray_dir)
        search_roots.append(path)
        for root in search_roots:
            best = _best_checkpoint_dir(root)
            if best is not None:
                if run_config is None:
                    run_config = _config_alongside(root) or _config_by_run_name(root)
                return best / "model.pt", run_config
    # Fall through; let load_checkpoint surface a clear error.
    return path, None


def main() -> None:
    """CLI entry point: loads checkpoint, collects embeddings, reports metrics."""
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)

    data_config = config.data
    eval_config = config.evaluation
    set_seed(int(eval_config.seed))
    device = resolve_device(str(eval_config.device))
    print(f"Using device: {device}")

    checkpoint, run_config_path = resolve_checkpoint(eval_config.get("checkpoint"))
    weights_source = f"checkpoint: {checkpoint}" if checkpoint is not None else "random initialization"
    print(f"Weights: {weights_source}")

    # Build the model to match the checkpoint's architecture. When a run config.json is
    # available (e.g. evaluation.checkpoint=runs/<run>), use its model.training block so a
    # different backbone (e.g. enc_channels) loads cleanly. Otherwise fall back to the
    # default config -- but warn, since a mismatch leaves checkpoint weights partly random.
    if run_config_path is not None:
        run_cfg = OmegaConf.create(json.loads(run_config_path.read_text()))
        model_config = select_model_config(run_cfg, "training")
        print(f"Model config from run: {run_config_path}")
    else:
        model_config = select_model_config(config, "training")
        if checkpoint is not None:
            print(
                "WARNING: no run config.json found for this checkpoint; building the model "
                "from the default config. If the checkpoint's backbone differs, weights will "
                "be partly random. Pass a runs/<run> directory to auto-match its architecture."
            )

    model = create_training_model(
        device=device,
        **model_factory_kwargs(model_config),
    )
    checkpoint_report = None
    if checkpoint is not None:
        checkpoint_report = load_checkpoint(model, str(checkpoint))
        missing = checkpoint_report.get("missing_keys", [])
        unexpected = checkpoint_report.get("unexpected_keys", [])
        if missing or unexpected:
            print(f"Checkpoint key report -- missing: {len(missing)}, " f"unexpected: {len(unexpected)}")
    model.eval()

    dataloader = create_dataloader(
        split=str(eval_config.val_split),
        batch_size=int(eval_config.batch_size),
        dataset_name=str(data_config.dataset_name),
        dataset_type=str(data_config.dataset_type),
        pu_config=str(data_config.pu_config),
        cache_dir=str(data_config.cache_dir),
        dataset_revision=str(data_config.dataset_revision) if data_config.get("dataset_revision") is not None else None,
        local_files_only=bool(data_config.get("local_files_only", False)),
    )

    batch_kwargs = sonata_batch_kwargs(config, "training", max_calo_hits=config.views.max_calo_hits)

    num_prototypes = int(model.num_prototypes)
    max_events = eval_config.get("max_events")
    max_events = int(max_events) if max_events is not None else None

    print(
        f"Collecting embeddings: val_split={eval_config.val_split}, "
        f"max_events={max_events}, subsample_budget={eval_config.point_subsample_budget}, "
        f"num_prototypes={num_prototypes}"
    )
    collection = collect_embeddings(
        model,
        dataloader,
        device,
        num_prototypes=num_prototypes,
        batch_kwargs=batch_kwargs,
        point_subsample_budget=int(eval_config.point_subsample_budget),
        max_events=max_events,
        seed=int(eval_config.seed),
    )
    metrics = summarize(collection)
    metrics["record_type"] = "eval"
    metrics["checkpoint"] = str(checkpoint) if checkpoint is not None else None
    metrics["weights_source"] = "trained" if checkpoint is not None else "random_init"
    metrics["val_split"] = str(eval_config.val_split)

    run_name = str(eval_config.get("run_name") or "").strip()
    if run_name:
        # Explicit override -> path under runs/ (user opt-out).
        run_dir = PROJECT_ROOT / "runs" / run_name
    elif checkpoint is not None and run_config_path is not None:
        # Evaluating a training checkpoint -> place inside the training run dir,
        # in a subfolder named after the checkpoint dir so the source is obvious.
        training_run_dir = run_config_path.parent
        ckpt_name = checkpoint.parent.name
        run_dir = training_run_dir / "eval" / ckpt_name
        if run_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = training_run_dir / "eval" / f"{ckpt_name}_{timestamp}"
    else:
        # Random-init or raw .pt without a run config -> legacy fallback.
        stem = checkpoint.stem if checkpoint is not None else "random_init"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{stem}_{timestamp}"
        run_dir = PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional phases that load the raw calo truth (preserving contrib_* fields that
    # ColliderMLDataset drops via select_columns). With data.local_files_only=true, set
    # HF_HUB_OFFLINE=1 to skip the slow 1000-shard hub verification.
    enable_dominance = bool(eval_config.get("enable_dominance_report", False))
    enable_tsne = bool(eval_config.get("enable_tsne", False))
    calo_truth = None
    pid_to_pdg = None
    if enable_dominance or enable_tsne:
        from collider_fm.evaluation_labels import load_calo_truth

        calo_truth = load_calo_truth(
            split=str(eval_config.val_split),
            dataset_name=str(data_config.dataset_name),
            dataset_type=str(data_config.dataset_type),
            pu_config=str(data_config.pu_config),
            cache_dir=str(data_config.cache_dir),
            dataset_revision=str(data_config.dataset_revision) if data_config.get("dataset_revision") is not None else None,
            local_files_only=bool(data_config.get("local_files_only", False)),
        )

    # Dominance report: pure-data characterization of the held-out subset's label-noise
    # floor (shared calorimeter cells have many contributors). Independent of the checkpoint.
    if enable_dominance:
        from collider_fm.evaluation_labels import compute_dominance_report, format_dominance_report

        dominance_max_events = int(eval_config.get("dominance_max_events", 200))
        print(f"\nDominance report: scanning up to {dominance_max_events} events " f"from {eval_config.val_split}...")
        dominance, n_scanned = compute_dominance_report(calo_truth, dominance_max_events)
        dominance["num_events_scanned"] = n_scanned
        metrics["dominance"] = dominance
        (run_dir / "dominance_report.txt").write_text(format_dominance_report(dominance))
        print(f"  wrote {run_dir / 'dominance_report.txt'}")

    # t-SNE: Panda-style per-point visualization of the backbone features, colored by
    # dominant-particle type + event id. Needs the particle_id->pdg_id join from the
    # sibling particles config. Two feature spaces are supported: full-up-cast (always)
    # and up_cast(2) (the pretraining space, when enabled).
    if enable_tsne:
        from collider_fm.evaluation_labels import load_particle_pdg
        from collider_fm.visualization import collect_tsne_points, make_2d_embedding_plots

        tsne_max_events = int(eval_config.get("tsne_max_events", 100))
        tsne_max_points = int(eval_config.get("tsne_max_points", 20000))
        enable_upcast2 = bool(eval_config.get("tsne_upcast2", False))
        up_cast_level = int(config.model.training.get("up_cast_level", 2))
        print(f"\nt-SNE: collecting up to {tsne_max_points} points over " f"{tsne_max_events} events from {eval_config.val_split}...")
        pid_to_pdg = load_particle_pdg(
            split=str(eval_config.val_split),
            dataset_name=str(data_config.dataset_name),
            dataset_type=str(data_config.dataset_type),
            pu_config=str(data_config.pu_config),
            cache_dir=str(data_config.cache_dir),
            dataset_revision=str(data_config.dataset_revision) if data_config.get("dataset_revision") is not None else None,
            local_files_only=bool(data_config.get("local_files_only", False)),
        )
        tsne_view_kwargs = point_view_kwargs(
            config,
            "training",
            max_calo_hits=int(eval_config.get("tsne_max_calo_hits", 8000)),
        )
        tsne_dir = run_dir / "viz"
        tsne_metrics: dict[str, Any] = {}

        # Full up-cast (Panda-style): input-resolution features, exact per-hit labels.
        print(f"\nt-SNE [full]: collecting full-up-cast features...")
        full_collection = collect_tsne_points(
            model,
            calo_truth,
            pid_to_pdg,
            device,
            view_kwargs=tsne_view_kwargs,
            max_events=tsne_max_events,
            max_points=tsne_max_points,
            seed=int(eval_config.seed),
            feature_space="full",
        )
        tsne_metrics["full"] = {
            "num_events": full_collection.num_events,
            "num_points": int(full_collection.features.shape[0]),
            "feature_dim": int(full_collection.features.shape[1]) if full_collection.features.ndim == 2 else 0,
            "plots": make_2d_embedding_plots(full_collection, tsne_dir, seed=int(eval_config.seed))
            + make_2d_embedding_plots(full_collection, tsne_dir, method="pca", seed=int(eval_config.seed)),
        }

        # up_cast(2): the pretraining feature space (the one the prototype loss shapes).
        # Downsampled vs input; labels are energy-dominant per cluster. A second forward
        # pass, so only run when requested.
        if enable_upcast2:
            print(f"\nt-SNE [upcast2]: collecting up_cast(2) features...")
            upcast2_collection = collect_tsne_points(
                model,
                calo_truth,
                pid_to_pdg,
                device,
                view_kwargs=tsne_view_kwargs,
                max_events=tsne_max_events,
                max_points=tsne_max_points,
                seed=int(eval_config.seed),
                feature_space="upcast2",
                up_cast_level=up_cast_level,
            )
            tsne_metrics["upcast2"] = {
                "num_events": upcast2_collection.num_events,
                "num_points": int(upcast2_collection.features.shape[0]),
                "feature_dim": int(upcast2_collection.features.shape[1]) if upcast2_collection.features.ndim == 2 else 0,
                "plots": make_2d_embedding_plots(upcast2_collection, tsne_dir, seed=int(eval_config.seed), subdir="upcast2")
                + make_2d_embedding_plots(upcast2_collection, tsne_dir, method="pca", seed=int(eval_config.seed), subdir="upcast2"),
            }
        metrics["tsne"] = tsne_metrics

    write_run_config(run_dir, to_plain_container(config))
    logger = JsonlLogger(run_dir)
    logger.log_metrics(metrics, step=0)

    report_path = run_dir / "summary.json"
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"\nWrote metrics to {run_dir / 'metrics_step.jsonl'}")
    print(f"Wrote summary to {report_path}\n")
    print("--- metrics ---")
    headline = [
        "participation_ratio",
        "participation_ratio_dim",
        "point_subsample_size",
        "prototype_entropy",
        "num_dead_prototypes",
        "num_active_prototypes",
        "num_empty_prototypes",
        "num_prototypes",
        "alignment",
        "uniformity",
        "r_at_1",
        "r_at_5",
        "num_events",
    ]
    for key in headline:
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print(
        f"  participation_ratio_spectrum (first 8 of {len(metrics.get('participation_ratio_spectrum', []))}): "
        f"{[round(v, 4) for v in metrics.get('participation_ratio_spectrum', [])[:8]]}"
    )

    if "dominance" in metrics:
        dom = metrics["dominance"]
        print("\n--- dominance report ---")
        print(f"  num_events_scanned: {dom.get('num_events_scanned')}")
        print(f"  num_hits: {dom.get('num_hits')}")
        print(
            f"  contributor_count: mean={dom.get('contributor_count_mean'):.2f} "
            f"median={dom.get('contributor_count_median')} p99={dom.get('contributor_count_p99')} "
            f"max={dom.get('contributor_count_max')}"
        )
        print(f"  pct_single_contributor={dom.get('pct_single_contributor'):.1f}% " f"pct_shared={dom.get('pct_shared'):.1f}%")
        print(
            f"  dominant_frac: median={dom.get('dominant_frac_median')} "
            f"pct>=0.9={dom.get('pct_dominant_ge_0.9'):.1f}% "
            f"pct>=0.5={dom.get('pct_dominant_ge_0.5'):.1f}%"
        )
        if "shared_num_hits" in dom:
            print(
                f"  shared-only ({dom.get('shared_num_hits')} hits): "
                f"frac_median={dom.get('shared_dominant_frac_median')} "
                f"pct>=0.9={dom.get('shared_pct_dominant_ge_0.9'):.1f}%"
            )

    if "tsne" in metrics:
        print("\n--- t-SNE ---")
        for space, tsne in metrics["tsne"].items():
            print(f"  [{space}] num_events: {tsne.get('num_events')}  " f"num_points: {tsne.get('num_points')}  " f"feature_dim: {tsne.get('feature_dim')}")
            for plot in tsne.get("plots", []):
                print(f"    plot: {plot}")


if __name__ == "__main__":
    main()
