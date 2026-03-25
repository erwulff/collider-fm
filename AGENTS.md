# AGENTS.md

This file is the short operational guide for coding agents in this repository.
When docs disagree, prefer the current filesystem and current code.

## Project snapshot

This repository currently implements a simple, calo-only self-supervised learning baseline on ColliderML Release 1.

Main implemented pieces:

- `src/collider_fm/data.py`: ColliderML loading and simple collation
- `src/collider_fm/views.py`: calo-only point views plus global, local, and masked variants
- `src/collider_fm/model.py`: small Panda-style student/teacher model
- `scripts/train.py`: checkpointed training with tqdm progress
- `scripts/plot_training_run.py`: loss and held-out similarity plots for finished runs
- `scripts/plot_diagnostics.py`: richer checkpoint diagnostics
- `scripts/export_embeddings.py`: frozen embedding export
- `notebooks/calo_pipeline_tutorial.ipynb`: teaching notebook

## Source-of-truth docs

- `README.md`: project overview and common workflows
- `PLAN.md`: short roadmap
- `IMPLEMENTATION_GUIDELINES.md`: technical target and current contracts
- `HPC.md`: cluster-specific workflow

## Repo rules

- Keep the code simple and easy to teach.
- Prefer explicit data flow over abstraction-heavy designs.
- Keep the project calo-only unless the user asks for a broader scope.
- Preserve the current point-view contract unless there is a strong reason to change it.
- Keep generated outputs out of git.
- Prefer SLURM for heavy GPU work.

## Execution notes

- Hugging Face cache: `/mnt/ceph/users/ewulff/data/hf`
- Generated outputs live in `runs/` and `diagnostics/`
- SLURM logs live in `logs_slurm/`

## Documentation maintenance

- Keep `AGENTS.md` short and operational.
- Update markdown files when workflows or filenames change.
- Remove stale statements that describe planned behavior as if it already exists.
