"""Evaluate ColliderFM pretraining quality with label-free collapse metrics.

Loads a checkpoint (or runs a random-init baseline), encodes held-out validation
events through the deterministic EMA teacher backbone, and reports:

  * per-point stable rank + singular-value spectrum  (collapse headline)
  * per-point prototype usage / entropy + dead-prototype count
  * per-event NN view-retrieval R@1/R@5 + alignment / uniformity (secondary lens)

Metrics are written to ``runs/eval_<run_name>/metrics_step.jsonl``.

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
    select_model_config,
    sonata_batch_kwargs,
    to_plain_container,
)
from collider_fm._panda.utils import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description=(
            "Evaluate ColliderFM pretraining with label-free collapse metrics "
            "(stable rank, prototype usage, NN view-retrieval, alignment/uniformity)."
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


def resolve_checkpoint(spec: Any) -> tuple[Path | None, Path | None]:
    """Resolve a checkpoint spec to ``(model.pt path, run config.json path or None)``.

    Accepts ``None``/empty (random-init baseline -> ``(None, None)``), a ``model.pt``
    file, a run directory (``runs/<run>`` -> reads ``checkpoint_path.txt`` -> latest
    ``checkpoint_*/model.pt``), or a Ray storage directory containing ``checkpoint_*``
    subdirs. The run ``config.json`` is recovered when the spec is a training run dir
    (``runs/<run>``) or a Ray storage dir whose name matches a run dir, so the model can
    be built to match the checkpoint's architecture exactly.
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
            # Only true checkpoint directories (skip checkpoint_manager_snapshot.json etc.).
            ckpts = sorted(p for p in root.glob("checkpoint_*") if p.is_dir())
            if ckpts:
                model_pt = ckpts[-1] / "model.pt"
                if model_pt.is_file():
                    if run_config is None:
                        run_config = _config_alongside(root) or _config_by_run_name(root)
                    return model_pt, run_config
    # Fall through; let load_checkpoint surface a clear error.
    return path, None


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)

    data_config = config.data
    eval_config = config.evaluation
    set_seed(int(eval_config.seed))
    device = resolve_device(str(eval_config.device))
    print(f"Using device: {device}")

    checkpoint, run_config_path = resolve_checkpoint(eval_config.get("checkpoint"))
    weights_source = (
        f"checkpoint: {checkpoint}" if checkpoint is not None else "random initialization"
    )
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
            print(
                f"Checkpoint key report -- missing: {len(missing)}, "
                f"unexpected: {len(unexpected)}"
            )
    model.eval()

    dataloader = create_dataloader(
        split=str(eval_config.val_split),
        batch_size=int(eval_config.batch_size),
        dataset_name=str(data_config.dataset_name),
        dataset_type=str(data_config.dataset_type),
        pu_config=str(data_config.pu_config),
        cache_dir=str(data_config.cache_dir),
        dataset_revision=str(data_config.dataset_revision)
        if data_config.get("dataset_revision") is not None
        else None,
        local_files_only=bool(data_config.get("local_files_only", False)),
    )

    batch_kwargs = sonata_batch_kwargs(
        config, "training", max_calo_hits=config.views.max_calo_hits
    )

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
    if not run_name:
        stem = (
            checkpoint.stem if checkpoint is not None else "random_init"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{stem}_{timestamp}"
    run_dir = PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, to_plain_container(config))
    logger = JsonlLogger(run_dir)
    logger.log_metrics(metrics, step=0)

    report_path = run_dir / "summary.json"
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"\nWrote metrics to {run_dir / 'metrics_step.jsonl'}")
    print(f"Wrote summary to {report_path}\n")
    print("--- v1 metrics ---")
    headline = [
        "stable_rank",
        "stable_rank_dim",
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
        f"  stable_rank_spectrum (first 8 of {len(metrics.get('stable_rank_spectrum', []))}): "
        f"{[round(v, 4) for v in metrics.get('stable_rank_spectrum', [])[:8]]}"
    )


if __name__ == "__main__":
    main()
