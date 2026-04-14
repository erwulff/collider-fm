# AGENTS.md

Operational guide for coding agents. Prefer current code over stale docs.

## Architecture

Two self-distillation recipes on ColliderML Release 1 (calo-only, `CERN/ColliderML-Release-1`):

| | Legacy (`recipe=legacy`) | Sonata (`recipe=sonata`) |
|---|---|---|
| Model | `PandaSelfDistillation` (`model.py`) | `SonataSelfDistillation` (`sonata_model.py`) |
| Views | `build_distillation_views` (`views.py`) | `build_sonata_batch` (`views.py`) |
| Heads | Single prototype head + center | Dual `OnlineCluster` heads (mask/unmask) + Sinkhorn-Knopp |
| Scheduling | Per-epoch teacher temp/momentum | Per-step cosine schedulers (mask_size, mask_ratio, temp, momentum) |
| Monitoring | `last_monitoring_state` | `last_monitoring_state` (+ `global_mask`, `cosine_similarities`) |

**Active recipe**: Sonata. Legacy is maintained but not the focus.

**Backbone** (both recipes): Vendored PTv3 in `src/collider_fm/_panda/` — do not modify.

**Data flow**: `ColliderMLDataset` → `collate_fn` → `build_sonata_batch` → `SonataBatch` → `SonataSelfDistillation.forward()`

**Config**: `config/default.yaml` — single file, overridden via dotlist CLI args. Sections: `data`, `views`, `sonata_views`, `model.*`, `training`, `diagnostics`.

**Training loop**: `scripts/train.py` — epoch loop with within-epoch logging/checkpointing/viz.

**Coordinates**: Normalized (÷5000.0, ~[-1,1]). All spatial params (grid_size, mask_size, match_max_r) are in normalized space.

## Key files

- `src/collider_fm/sonata_model.py` — Sonata model, schedulers, mask generation, matching
- `src/collider_fm/model.py` — Factory functions, PandaSelfDistillation
- `src/collider_fm/views.py` — View construction, augmentation, batching
- `src/collider_fm/data.py` — Dataset and collate
- `src/collider_fm/experiment_logging.py` — Null/Jsonl/Comet loggers + `log_image`
- `src/collider_fm/project_config.py` — OmegaConf config loading
- `scripts/train.py` — Training loop, checkpointing, diagnostic images
- `config/default.yaml` — All defaults

## Operating principles

1. **Think before coding** — Read relevant code first. Trace data flow end-to-end before changing it. State assumptions explicitly.
2. **Simplicity first** — No new abstractions, no new files, no new dependencies unless the user asks. Prefer config changes over code changes.
3. **Surgical changes** — Change only what's needed. Do not refactor adjacent code. Do not touch `Panda_repo/` or `_panda/` unless asked.
4. **Goal-driven execution** — Verify changes with `pytest` before asking to commit. Define success criteria before implementing.

## Hard rules

- Keep the runtime path calo-only unless explicitly asked otherwise.
- Use `Panda_repo/` and `particle-imaging-models/` as reference only, not as runtime code.
- Do not run long downloads or GPU jobs interactively. Use SLURM.
- Leave `runs/` and `diagnostics/` uncommitted unless asked.
- Always pass `data.local_files_only=true` in SLURM jobs.
- Ask the user before committing — let them review first.
- Remove `set -euo pipefail` from SLURM scripts.
- Define all SLURM parameters in the sbatch script, not on the command line.

## Cluster

- Load `uv` once: `module load uv`
- HF cache: `/mnt/ceph/users/ewulff/data/hf`
- Dataset revision: `e28a24cc9c1641a478ae4e5bc3b376eb624b7283`
- SLURM jobs source `slurm/load_env.sh`
- Single-GPU → `a100-80gb`. Multi-GPU (4+) → `h100`/`h200`.
- `torch_scatter` requires A100/H100 (no CUDA kernels for RTX Ada / compute 8.9).

## Docs

- `README.md` — project overview
- `markdown/WORKFLOWS.md` — runtime commands
- `markdown/HPC.md` — cluster specifics
- `markdown/PLAN.md` — roadmap
- Keep `AGENTS.md` short and operational.
- Keep all project markdown files except `README.md` and `AGENTS.md` under `markdown/`.
