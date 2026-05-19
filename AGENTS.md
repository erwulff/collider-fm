# AGENTS.md

Operational guide for coding agents. Prefer current code over stale docs.

## Behavioural guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project specific guidelines

### Architecture

Self-distillation pretraining on ColliderML Release 1 (calo-only, `CERN/ColliderML-Release-1`):

| Component | Implementation |
|---|---|
| Model | `SonataSelfDistillation` (`sonata_model.py`) |
| Views | `build_sonata_batch` (`views.py`) |
| Heads | Dual `OnlineCluster` student heads (mask/unmask) + Sinkhorn-Knopp; teacher uses unified mask_head for both branches |
| Scheduling | Per-step cosine schedulers (mask_size, mask_ratio, temp, momentum) |
| Monitoring | `last_monitoring_state` (+ `global_mask`, `cosine_similarities`) |
| Multi-GPU | Ray Train (`training_loop.py`) with PyTorch DDP; future Ray Tune HPO compatible |

**Backbone**: Vendored PTv3 in `src/collider_fm/_panda/` — do not modify.

**Data flow**: `ColliderMLDataset` → `collate_fn` → `build_sonata_batch` → `SonataBatch` → `SonataSelfDistillation.forward()`

**Config**: `config/default.yaml` — single file, overridden via dotlist CLI args. Sections: `data`, `views`, `model.*`, `training`, `diagnostics`.

**Training loop**: `src/collider_fm/training_loop.py` — Ray Train worker function, epoch loop, checkpoint I/O. `scripts/train.py` is a thin CLI driver that launches a `TorchTrainer`.

**Multi-GPU**: `training.num_gpus` controls GPU count per run. `training.batch_size` is per-GPU; global batch = `batch_size * num_gpus`. Checkpoints are persisted by Ray at `training.ray_storage_path`; local `runs/` holds metrics, config, and viz.

**Coordinates**: Normalized (÷5000.0, ~[-1,1]). All spatial params (grid_size, mask_size, match_max_r) are in normalized space.

### Key files

- `src/collider_fm/sonata_model.py` — Sonata model, schedulers, mask generation, matching
- `src/collider_fm/model.py` — Factory functions (`create_small_model`, `create_training_model`)
- `src/collider_fm/views.py` — View construction, augmentation, batching
- `src/collider_fm/data.py` — Dataset and collate
- `src/collider_fm/training_loop.py` — Ray Train worker function, epoch loop, checkpoint save/load
- `src/collider_fm/experiment_logging.py` — Null/Jsonl/Comet loggers + `log_image`
- `src/collider_fm/project_config.py` — OmegaConf config loading
- `scripts/train.py` — Ray Train CLI driver
- `config/default.yaml` — All defaults

### Operating principles

1. **Think before coding** — Read relevant code first. Trace data flow end-to-end before changing it. State assumptions explicitly.
2. **Simplicity first** — No new abstractions, no new files, no new dependencies unless the user asks. Prefer config changes over code changes.
3. **Surgical changes** — Change only what's needed. Do not refactor adjacent code. Do not touch `Panda_repo/` or `_panda/` unless asked.
4. **Goal-driven execution** — Verify changes with `pytest` before asking to commit. Define success criteria before implementing.

### Hard rules

- Keep the runtime path calo-only unless explicitly asked otherwise.
- Use `Panda_repo/` and `particle-imaging-models/` as reference only, not as runtime code.
- Do not run long downloads or GPU jobs interactively. Use SLURM.
- Leave `runs/` and `diagnostics/` uncommitted unless asked.
- Always pass `data.local_files_only=true` in SLURM jobs.
- Ask the user before committing — let them review first.
- Remove `set -euo pipefail` from SLURM scripts.
- Define all SLURM parameters in the sbatch script, not on the command line.

### Cluster

- Load `uv` once: `module load uv`
- HF cache: `/mnt/ceph/users/ewulff/data/hf`
- Dataset revision: `e28a24cc9c1641a478ae4e5bc3b376eb624b7283`
- SLURM jobs source `slurm/load_env.sh`
- Single-GPU → `a100-80gb`. Multi-GPU (2 for debug, 8 for full) → `h100`/`h200`.
- Ray Train checkpoints: `/mnt/ceph/users/ewulff/raytrain_results/`
- Ray Tune checkpoints (future HPO): `/mnt/ceph/users/ewulff/raytune_results/`
- Resume: re-run `scripts/train.py` with the same `training.run_name`.
- `torch_scatter` requires A100/H100 (no CUDA kernels for RTX Ada / compute 8.9).

### Docs

- `README.md` — project overview
- `markdown/WORKFLOWS.md` — runtime commands
- `markdown/HPC.md` — cluster specifics
- `markdown/PLAN.md` — roadmap
- Keep `AGENTS.md` short and operational.
- Keep all project markdown files except `README.md` and `AGENTS.md` under `markdown/`.
