# HPC resources and SLURM

This project is developed on an HPC cluster.
Use this document for cluster-specific setup notes and the recommended SLURM workflow.

## General guidance

- Prefer SLURM jobs for heavy downloads, dependency builds, GPU validation, and training runs.
- Write SLURM logs to `logs_slurm/`.
- Keep interactive terminal work lightweight.

## uv on the cluster

The cluster provides a `uv` module:

```bash
module load uv
```

The documented setup flow is:

```bash
sbatch slurm/create_uv_venv.slurm
sbatch slurm/install_deps.slurm
sbatch slurm/download.slurm
sbatch slurm/test_model.slurm
sbatch slurm/train_small.slurm
```

Runtime jobs source `slurm/load_env.sh`, which loads modules and activates `.venv`.

## Project-specific notes

- The Hugging Face cache location is `/mnt/ceph/users/ewulff/data/hf`.
- The current PTv3 plus spconv stack is effectively GPU-only for model execution.
- Optional Comet ML logging can be enabled through `~/.comet.config` or the standard `COMET_*` environment variables.

## Main SLURM jobs

- `slurm/create_uv_venv.slurm`
  - creates the `.venv`
  - runs `uv sync`
- `slurm/install_deps.slurm`
  - reinstalls GPU-sensitive packages on a compute node
- `slurm/download.slurm`
  - downloads ColliderML calorimeter tables
- `slurm/test_model.slurm`
  - runs the smoke test on a GPU node
- `slurm/train_small.slurm`
  - runs the current SSL baseline

Both runtime jobs now source `slurm/load_env.sh` so the environment setup is shared in one place.

## Current Monday baseline job

`slurm/train_small.slurm` runs the current reference baseline:

- 20 epochs
- 100 train batches per epoch
- 20 validation batches per epoch
- `max_calo_hits = 256`
- local views enabled with `local_fraction = 0.5`
- masked views enabled with `mask_fraction = 0.3`
- tqdm progress reporting in the SLURM logs
- default run name `monday_ssl_baseline`

Submit it with:

```bash
sbatch slurm/train_small.slurm
```

Monitor it with:

```bash
squeue --me
tail -f logs_slurm/log_train_ssl_baseline_<jobid>.out
tail -f logs_slurm/log_train_ssl_baseline_<jobid>.err
```

## Output locations

- training outputs: `runs/<run-name>/`
- detailed diagnostics: `diagnostics/<name>/`
- SLURM logs: `logs_slurm/`

These output directories are local generated artifacts and are ignored by git.

## Practical note

If you rerun the same named training job, `metrics.jsonl` may accumulate multiple runs in the same directory.
`scripts/plot_training_run.py` already handles this by plotting only the last monotonic run segment, but unique run names are still cleaner when comparing experiments.
