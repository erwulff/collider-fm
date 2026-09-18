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
  - generic bulk-cache download job; currently requests `tracker_hits` and `particles`, so edit it if you want a calo-only cache warmup

## Standard SLURM scripts

Four training and four evaluation scripts are kept; each has an "EDIT" block at the top for the per-run knobs (run name prefix, epochs, batch size, checkpoint, etc.), and everything else is fixed to sensible defaults. Add new experiments by editing a script before submitting rather than by creating a new one; run-specific runs are distinguished by `RUN_NAME_PREFIX` (first positional arg), not by separate files.

### Training

| script | purpose | GPUs / node | defaults |
|---|---|---|---|
| `slurm/train_smoke.slurm` | end-to-end training wiring sanity check; *not* for experiments | 1 × a100 | 1 epoch, batch 4, tiny `train[:32]` / `val[:8]`, auto-eval off |
| `slurm/train.slurm` | the training script — modify the EDIT block per run | 8 × h100 | full paper-config backbone, 100 epochs, batch 8/GPU |

Usage:

```bash
sbatch slurm/train.slurm                # run name "train_<ts>"
sbatch slurm/train.slurm mytest         # run name "mytest_train_<ts>"
```

### Evaluation

| script | purpose | defaults |
|---|---|---|
| `slurm/eval.slurm` | label-free quality metrics (harness); set `TSNE=true` to add t-SNE/PCA | probes and t-SNE off |
| `slurm/eval_tsne.slurm` | t-SNE/PCA plots, per-point features colored by particle / event id | 100 events × 20k points caps |
| `slurm/eval_probes.slurm` | Panda-style linear probes (seg + energy, random-init and raw-input baselines) | 2000 train / 500 val probe events |

`eval.slurm` is the "evaluate this checkpoint" job; `eval_probes.slurm` re-runs probes when probe settings change (cached backbone features make the second run fast); `eval_tsne.slurm` adds visualization.

### First-pass job order on a fresh cluster

```bash
sbatch slurm/create_uv_venv.slurm
sbatch slurm/test_model.slurm
sbatch slurm/train_smoke.slurm
```

If you want to prewarm the dataset cache first, either edit `slurm/download.slurm` for `calo_hits` or use `scripts/download_data.py` directly with the desired object types.

## Reproducibility and cache usage

The checked-in default dataset revision is:

```text
64c3d2f112df3d5d20979d22da7cfdff13e10c4b
```

For reproducible training, prefer pinning that revision and using `data.local_files_only=true` once the cache is populated.

## Node guidance

The checked-in jobs currently use these hardware patterns:

- smoke test and single-GPU training jobs: `a100` or `a100-80gb`
- multi-GPU training jobs (8 GPUs for full): `h100`/`h200`
- environment bootstrap jobs: `h100`

When requesting fewer than 4 GPUs, always use `a100` (not `h100` or `h200`). Only use `h100`/`h200` for multi-GPU jobs (4+ GPUs).

## Outputs and logging

- SLURM logs go to `logs_slurm/log_<jobname>_<jobid>.out` and `.err`
- training outputs and local logging conventions are documented in `markdown/WORKFLOWS.md`
