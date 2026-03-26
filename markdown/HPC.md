# HPC resources and SLURM

This repository is developed and tested on an HPC cluster. Use this document for cluster-specific setup, recommended job order, and the SLURM jobs that match the current codebase.

## General guidance

- Prefer SLURM for heavy downloads, dependency builds, smoke tests, and real training runs.
- Keep SLURM stdout and stderr under `logs_slurm/`.
- Avoid over-requesting CPUs or system memory, especially on shared A100 and H100 nodes.
- The current PTv3 and `spconv` runtime path is GPU-only for real model execution.

## uv on the cluster

Load the cluster `uv` module first:

```bash
module load uv
```

The module sets `UV_CACHE_DIR=$HOME/.cache/uv` and appends its own binary to `PATH`, so a user-installed `uv` can still remain first in `PATH`.

## Environment setup jobs

The repository includes these setup jobs:

- `slurm/create_uv_venv.slurm`
  - creates `.venv`
  - runs `uv sync`
- `slurm/install_deps.slurm`
  - rebuilds selected packages on a compute node
- `slurm/download.slurm`
  - downloads the documented calo-hit subset into the shared HF cache

The shared runtime bootstrap lives in `slurm/load_env.sh` and is sourced by the runtime jobs. It now fails early if `.venv` has not been created yet.

## Recommended job order

For a fresh environment and first validation pass, use:

```bash
sbatch slurm/create_uv_venv.slurm
sbatch slurm/install_deps.slurm
sbatch slurm/download.slurm
sbatch slurm/test_model.slurm
sbatch slurm/train_small.slurm
```

For a more informative debug training run after the short pass succeeds:

```bash
sbatch slurm/train_medium.slurm
```

## Current training jobs

### `slurm/test_model.slurm`

- runs `scripts/smoke_test_model.py`
- checks the real calo-only data path on GPU when cached data is available

### `slurm/train_small.slurm`

- short debug run
- 1 GPU on `a100-40gb`
- useful for validating the trainer, checkpointing, and metric logging quickly

### `slurm/train_medium.slurm`

- medium-length debug run
- 1 GPU on `a100-40gb`
- intended to produce more informative curves and checkpoints than the short recipe

## Dataset cache

The documented Hugging Face cache for this project is:

```text
/mnt/ceph/users/ewulff/data/hf
```

Current scripts and SLURM jobs default to that path.

## Logging and outputs

- training runs are written under `runs/<run_name>/`
- SLURM logs go to `logs_slurm/log_<jobname>_<jobid>.out` and `.err`
- diagnostics are typically written under `diagnostics/`

## Comet logging

Training always writes JSONL metrics locally. Optional Comet logging can be enabled by either:

- a saved `~/.comet.config`, or
- exported `COMET_API_KEY`, `COMET_PROJECT_NAME`, and `COMET_WORKSPACE`

## Current node guidance

For the current single-GPU training/debug jobs in this repo, the active recommendation is:

- `-p gpu`
- `--gpus-per-node=1`
- `--cpus-per-task=16`
- `-C a100-40gb`

That matches the short and medium training jobs presently checked into `slurm/`.
