# AGENTS.md

This file is the primary agent-facing guide for this repository. Use it as the operational source of truth for repo workflow and constraints. When other docs conflict with the current filesystem, prefer the current repo state.

## Project Snapshot

This repository is an early-stage Python project for foundation-model work on particle collider data, with an initial focus on adapting Panda-style self-distillation workflows to ColliderML-style data.

Current implemented areas:

- `src/collider_fm/data.py`: dataset loading and simple collate logic.
- `src/collider_fm/model.py`: early Panda-style model scaffold with a mock test entrypoint.
- `scripts/download_data.py`: dataset download helper.
- `scripts/inspect_data.py`: basic event visualization.
- `slurm/`: job scripts for environment setup, dependency installation, data download, and model smoke tests.
- `apptainer/build_gemini_container.sh`: container build helper.

This repo is still in a scaffold/prototype phase. Some docs describe intended future structure that is not present yet.

## Source Of Truth Docs

- `README.md`: human-oriented project overview and setup notes.
- `PLAN.md`: roadmap and task checklist.
- `HPC.md`: HPC and SLURM usage constraints.
- `GEMINI.md`: legacy project context. Use it for background, not as the primary operational guide.

If a document mentions files or workflows that do not exist, treat them as planned or stale unless the filesystem confirms them.

## Repo Map

Top-level layout at time of writing:

- `src/collider_fm/`: Python package code.
- `scripts/`: standalone utilities.
- `slurm/`: batch jobs intended for the cluster.
- `apptainer/`: container-related helper scripts.
- `Panda_repo/`: git submodule checkout of the Panda reference implementation.
- `main.py`: trivial placeholder entrypoint.
- `requirements.txt`: Python dependency list.

Known doc/repo mismatches:

- `.venv` is referenced in docs and SLURM scripts, but is not present in the repo.
- `WORKING_NOTES.md` is referenced in docs, but is not present in the repo.

Agents should not assume those paths exist unless they are created later.

## Agent Operating Rules

- Read relevant files before editing them.
- Prefer current repository state over stale documentation.
- Keep changes aligned with the existing project phase; avoid inventing full production infrastructure where only scaffolding exists.
- Use the `Panda_repo/` submodule as the reference implementation when working on Panda-related model code.
- When touching planning docs, preserve the distinction between implemented code and planned work.
- Do not run compute-heavy training, large downloads, or long GPU jobs from the interactive terminal.
- Prefer SLURM scripts for heavyweight tasks. If SLURM submission is required and unavailable to the agent, prepare the script changes and tell the user what to run.

## Execution Notes

- On this cluster, load `uv` with `module load uv` so `UV_CACHE_DIR=$HOME/.cache/uv` is set. The module appends its own binary to `PATH`, so a user-installed `uv` still takes precedence.
- The Hugging Face cache location documented for this project is `/mnt/ceph/users/ewulff/data/hf`.
- Heavy dataset download and GPU validation workflows are intended to run through files in `slurm/`.
- `logs_slurm/` is tracked as a placeholder directory for SLURM stdout and stderr files.
- CUDA versions and cluster details belong in `HPC.md`; do not duplicate large hardware reference tables in agent-facing docs.

## Documentation Maintenance

When updating repo docs:

- Keep `AGENTS.md` short and operational.
- Put project background and narrative in `README.md`.
- Put cluster policy and example batch directives in `HPC.md`.
- Put roadmap and task tracking in `PLAN.md`.
- Remove or downgrade stale aspirational statements that read like current facts.
