# HPC and SLURM

This document covers cluster-specific setup, cache paths, and the checked-in SLURM jobs. For the project overview, use `README.md`. For runtime details and local commands, use `markdown/WORKFLOWS.md`.

## Cluster assumptions

- prefer SLURM for dependency builds, heavy downloads, smoke tests, and real training runs
- keep SLURM stdout and stderr under `logs_slurm/`
- the PTv3 plus `spconv` runtime path is GPU-only for real model execution
- the documented Hugging Face cache is `/mnt/ceph/users/ewulff/data/hf`

## Interactive setup

Load `uv` once at the start of the session:

```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync --locked --dev
```

The shared batch bootstrap lives in `slurm/load_env.sh`. Runtime jobs source that file and fail early if `.venv` has not been created yet.
The checked-in runtime jobs execute with the activated environment's `python`; `uv` is reserved for environment creation and dependency mutation jobs.

## Environment bootstrap jobs

- `slurm/create_uv_venv.slurm`
  - creates `.venv`
  - runs `uv sync`
- `slurm/download.slurm`
  - generic bulk-cache download job
  - currently requests `tracker_hits` and `particles`, so edit it if you want a calo-only cache warmup for the current runtime path

## Recommended first-pass job order

For a fresh cluster environment:

```bash
sbatch slurm/create_uv_venv.slurm
sbatch slurm/test_model.slurm
sbatch slurm/train_gpu1_debug.slurm
```

If you want to prewarm the dataset cache first, either edit `slurm/download.slurm` for `calo_hits` or use `scripts/download_data.py` directly with the desired object types.

## Checked-in runtime jobs

### `slurm/test_model.slurm`

- runs `scripts/smoke_test_model.py`
- 1 GPU on `a100-80gb`
- uses the shared `slurm/load_env.sh` bootstrap

### `slurm/train_gpu1_debug.slurm`

- short Sonata debug run
- 1 GPU on `a100`
- 4 epochs, batch_size=4, 200 train / 40 val events
- useful for validating the Sonata pipeline end-to-end before a longer run
- run names are suffixed with a microsecond-resolution timestamp, for example `test_x_sonata_debug_20260610_043244_123456`
- optional positional prefix: `sbatch slurm/train_gpu1_debug.slurm test_x` -> `test_x_sonata_debug_<timestamp>`

### `slurm/train_gpu1.slurm`

- longer Sonata training run
- 1 GPU on `a100-80gb`
- 5 epochs, batch_size=8, full train and val splits
- uses `data.local_files_only=true` and logs to Comet
- run names are suffixed with a microsecond-resolution timestamp, for example `test_x_sonata_full_20260610_043244_123456`
- optional positional prefix: `sbatch slurm/train_gpu1.slurm test_x` -> `test_x_sonata_full_<timestamp>`

### `slurm/train_multigpu_debug.slurm`

- 2-GPU Ray Train debug run
- 2 GPUs on `a100-80gb`, 16 CPUs
- 5 epochs, batch_size=4 per GPU, 20 train / 5 val batches
- uses `training.num_gpus=2`, `training.num_workers=4`
- uses `training.log_backend=comet`
- run names are suffixed with a microsecond-resolution timestamp, for example `test_x_train_multigpu_debug_20260610_043244_123456`
- optional positional prefix: `sbatch slurm/train_multigpu_debug.slurm test_x` -> `test_x_train_multigpu_debug_<timestamp>`

### `slurm/train_multigpu.slurm`

- 8-GPU Ray Train full run
- 8 GPUs on `h100`, 64 CPUs
- 5 epochs, batch_size=12 per GPU (global batch=96), full splits
- uses `training.num_gpus=8`, `training.num_workers=4`, `data.local_files_only=true`
- checkpoints persisted to `/mnt/ceph/users/ewulff/raytrain_results/`
- run names are suffixed with a microsecond-resolution timestamp, for example `test_x_train_multigpu_20260610_043244_123456`
- optional positional prefix: `sbatch slurm/train_multigpu.slurm test_x` -> `test_x_train_multigpu_<timestamp>`

## Reproducibility and cache usage

The checked-in default dataset revision is:

```text
64c3d2f112df3d5d20979d22da7cfdff13e10c4b
```

For reproducible training, prefer pinning that revision and using `data.local_files_only=true` once the cache is populated.

## Node guidance

The checked-in jobs currently use these hardware patterns:

- smoke test and single-GPU training jobs: `a100` or `a100-80gb`
- multi-GPU training jobs (2 GPUs for debug, 8 GPUs for full): `h100`/`h200`
- environment bootstrap jobs: `h100`

When requesting fewer than 4 GPUs, always use `a100` (not `h100` or `h200`). Only use `h100`/`h200` for multi-GPU jobs (4+ GPUs).

## Outputs and logging

- SLURM logs go to `logs_slurm/log_<jobname>_<jobid>.out` and `.err`
- training outputs and local logging conventions are documented in `markdown/WORKFLOWS.md`
