# Collider Foundation Model

Collider Foundation Model is an early-stage Python project for learning reusable representations from particle collider data. The current focus is a calo-only first stage: adapting Panda-style self-distillation workflows to ColliderML calorimeter hits before adding tracker or downstream reconstruction heads.

## Project goal

The project explores self-distillation methods for high-energy physics data, with Panda-style sensor-level representation learning as the starting point. The near-term goal is to get a practical training and evaluation workflow running on ColliderML before scaling to larger experiments.

## Current status

- `src/collider_fm/data.py` loads ColliderML calorimeter parquet tables through Hugging Face `datasets` and normalizes the energy field aliases.
- `src/collider_fm/views.py` converts one event into calo-only point clouds with features `[x, y, z, energy]` plus bookkeeping fields used by the SSL pipeline.
- `src/collider_fm/model.py` contains the current Panda-inspired self-distillation scaffold and the shared small-model factory used by train, smoke-test, and diagnostics scripts.
- `scripts/download_data.py`, `scripts/inspect_data.py`, and `scripts/plot_diagnostics.py` are aligned with the calo-only phase.
- `Panda_repo/` is kept as a reference implementation, while the runtime code vendors the PTv3 pieces currently needed under `src/collider_fm/_panda/`.

## Repository layout

```text
/
├── src/collider_fm/            # Python package code
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts for downloads, inspection, and smoke tests
├── slurm/                      # Batch jobs for cluster workflows
├── apptainer/                  # Container helper scripts
├── Panda_repo/                 # Panda reference submodule
├── logs_slurm/                 # Placeholder directory for SLURM logs
├── pyproject.toml              # Project metadata and tool configuration
├── uv.lock                     # Locked dependency resolution for uv
├── PLAN.md                     # Roadmap and task tracking
└── HPC.md                      # Cluster and SLURM notes
```

## Environment setup

The project uses `uv` for dependency management and targets Python 3.12.

On this cluster, load the `uv` module first:

```bash
module load uv
```

Then create and activate a virtual environment and sync dependencies:

```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

Some dependencies such as `spconv` and `torch-scatter` may need to be installed on compute nodes rather than the login node. `torch-scatter` is configured in `pyproject.toml` to build from source.

If you prefer to create the environment through SLURM, use:

```bash
sbatch slurm/create_uv_venv.slurm
```

The intended compute-node setup order is:

```bash
sbatch slurm/create_uv_venv.slurm
sbatch slurm/test_model.slurm
sbatch slurm/train_small.slurm
```

The shared cluster environment bootstrap for the runtime jobs lives in `slurm/load_env.sh`.

## Dataset

The project uses the ColliderML dataset:

- Hugging Face: https://huggingface.co/datasets/ColliderML/ColliderML
- Project site: https://colliderml.github.io/

ColliderML configurations combine a physics process such as `ttbar`, `ggf`, or `dihiggs`, a pileup setting such as `pu0` or `pu200`, and an object type such as `calo_hits`, `tracker_hits`, `particles`, or `tracks`.

The current implementation intentionally uses only the `calo_hits` tables. Each event is treated as a sparse 3D point cloud with per-point features `[x, y, z, energy]`.

On this cluster, the documented Hugging Face cache location is `/mnt/ceph/users/ewulff/data/hf`.

## Common workflows

Download the calo-hit subset used by the current pipeline:

```bash
uv run python scripts/download_data.py --pu-config pu0 --num-proc 12
```

Inspect a sample event:

```bash
uv run python scripts/inspect_data.py
```

Run the model smoke test on a GPU allocation:

```bash
sbatch slurm/test_model.slurm
```

The smoke-test entrypoint lives in `scripts/smoke_test_model.py`.
By default it expects cached ColliderML data and fails loudly if the dataset is unavailable.
For a CUDA-only synthetic sanity check, run:

```bash
uv run python scripts/smoke_test_model.py --allow-synthetic-fallback
```

A minimal tracked training run looks like:

```bash
uv run python scripts/train.py --num-epochs 1 --max-train-batches 1 --max-val-batches 1
```

Generate saved diagnostics from cached ColliderML calo events:

```bash
uv run python scripts/plot_diagnostics.py --detail-split train[0:1] --representation-split train[:10]
```

## Development notes

Python formatting is handled with Black and configured in `pyproject.toml` with a line length of 160 characters.

```bash
uv run black .
```

Notebook commits are cleaned automatically with `pre-commit` and `nbstripout` so committed `.ipynb` files do not keep cell outputs.

```bash
uv run pre-commit install
```

If a notebook with outputs is staged, the hook will strip the outputs, stop the commit once, and ask you to re-stage the cleaned notebook before committing again.

Training is currently calo-only. By default `scripts/train.py` writes `config.json` and `metrics.jsonl` under a timestamped directory in `runs/`.

To opt into Comet ML as an additional backend, either log in so Comet saves credentials in `~/.comet.config`, or export:

```bash
export COMET_API_KEY=...
export COMET_PROJECT_NAME=collider-fm
export COMET_WORKSPACE=...
```

Then run training with the default `--log-backend auto` or force Comet with `--log-backend comet`.

For roadmap and cluster-specific guidance:

- `PLAN.md` tracks milestones and open work.
- `HPC.md` covers cluster usage, SLURM notes, and environment details.

## References

- Panda paper: https://arxiv.org/abs/2512.01324
- ColliderML dataset: https://huggingface.co/datasets/ColliderML/ColliderML
- ColliderML website: https://colliderml.github.io/
