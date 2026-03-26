# AGENTS.md

This is the short operational guide for coding agents working in this repository. Prefer the current filesystem and code over stale planning text.

## Project snapshot

This repo currently implements a calo-only Panda-style self-distillation pipeline on ColliderML Release 1.

Implemented today:

- `src/collider_fm/data.py`: ColliderML loading and calorimeter energy alias normalization
- `src/collider_fm/views.py`: point-view construction and masked-global view batching
- `src/collider_fm/model.py`: point-level student/teacher scaffold
- `src/collider_fm/diagnostics.py`: shared plotting and checkpoint helpers
- `scripts/train.py`: checkpointed training loop with JSONL logging
- `scripts/smoke_test_model.py`: GPU smoke test
- `scripts/plot_diagnostics.py`: raw/view/model diagnostics
- `scripts/plot_training_run.py`: run-level metric plotting
- `notebooks/plot_diagnostics_explorer.ipynb`: tutorial notebook
- `slurm/`: batch jobs for setup, download, smoke test, and short/medium training runs

## Source-of-truth docs

- `README.md`: project overview and common workflows
- `markdown/PLAN.md`: current roadmap
- `markdown/HPC.md`: cluster-specific notes and SLURM usage
- `markdown/IMPLEMENTATION_GUIDELINES.md`: current design target and implementation notes
- `markdown/PANDA_SUMMARY.md`: short Panda background note for this repo

## Operating rules

- Read relevant files before editing.
- Prefer current repo state over aspirational or stale docs.
- Keep the current runtime path calo-only unless the user explicitly asks for a different scope.
- Use `Panda_repo/` as a reference implementation, not as the main runtime path.
- Do not run long downloads or long GPU jobs interactively.
- Prefer the checked-in SLURM jobs for heavyweight work.
- Leave generated outputs in `runs/` and `diagnostics/` uncommitted unless the user explicitly asks otherwise.
- Do not touch unrelated submodule changes in `Panda_repo/`.

## Cluster notes

- Load `uv` with `module load uv`.
- The documented HF cache is `/mnt/ceph/users/ewulff/data/hf`.
- Runtime SLURM jobs source `slurm/load_env.sh`.
- Current training jobs target single-GPU `a100-40gb` nodes.

## Documentation maintenance

- Keep `AGENTS.md` short and operational.
- Keep all project markdown files except `README.md` and `AGENTS.md` under `markdown/`.
- Put user-facing workflows in `README.md`.
- Put cluster specifics in `markdown/HPC.md`.
- Put roadmap items in `markdown/PLAN.md`.
