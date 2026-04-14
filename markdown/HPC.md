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
uv sync --dev
```

The shared batch bootstrap lives in `slurm/load_env.sh`. Runtime jobs source that file and fail early if `.venv` has not been created yet.

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
sbatch slurm/train_small.slurm
```

If you want to prewarm the dataset cache first, either edit `slurm/download.slurm` for `calo_hits` or use `scripts/download_data.py` directly with the desired object types.

## Checked-in runtime jobs

### `slurm/test_model.slurm`

- runs `scripts/smoke_test_model.py`
- currently targets H100 nodes
- the checked-in resource request is larger than the script strictly needs, so treat it as a cluster-specific example

### `slurm/train_small.slurm`

- short debug run
- 1 GPU on `a100-40gb`
- tiny train and validation slices
- useful for validating trainer, checkpointing, and metric logging quickly

### `slurm/train_medium.slurm`

- medium debug run
- 1 GPU on `a100-40gb`
- still intentionally small enough for quick turnaround and curve inspection

### `slurm/train.slurm`

- longer example training job (legacy recipe)
- currently targets 1 GPU on `h100`
- overrides `training.batch_size=32`, uses `data.local_files_only=true`, and logs to Comet
- best treated as a starting point rather than the tuned project default, which now lives in `config/default.yaml`

### `slurm/train_sonata_debug.slurm`

- short Sonata debug run
- 1 GPU on `a100-40gb`
- 4 epochs, batch_size=4, 200 train / 40 val events
- useful for validating the Sonata pipeline end-to-end before a longer run

### `slurm/train_sonata.slurm`

- longer Sonata training run
- 1 GPU on `a100`
- 20 epochs, batch_size=8, 5k train / 200 val events
- uses `data.local_files_only=true` and logs to Comet

## Reproducibility and cache usage

The checked-in default dataset revision is:

```text
e28a24cc9c1641a478ae4e5bc3b376eb624b7283
```

For reproducible training, prefer pinning that revision and using `data.local_files_only=true` once the cache is populated.

## Node guidance

The checked-in jobs split across two hardware patterns:

- `slurm/train_small.slurm`, `slurm/train_medium.slurm`, `slurm/train_sonata_debug.slurm`, `slurm/train_sonata.slurm`: single-GPU `a100` or `a100-40gb`
- setup, smoke-test, and long-train jobs: `h100`

When requesting fewer than 4 GPUs, always use `a100` (not `h100` or `h200`). Only use `h100`/`h200` for multi-GPU jobs (4+ GPUs).

## Outputs and logging

- SLURM logs go to `logs_slurm/log_<jobname>_<jobid>.out` and `.err`
- training outputs and local logging conventions are documented in `markdown/WORKFLOWS.md`
