# Collider Foundation Model

Collider Foundation Model is a calo-only self-supervised learning project built around ColliderML Release 1 and a Panda-style student-teacher training recipe. The current repository focus is a clean first-stage pipeline for calorimeter hits only: build point-cloud views from raw events, train a masked global self-distillation model, save checkpoints, and generate diagnostics and training-run plots.

## Current status

- `src/collider_fm/data.py` loads ColliderML `calo_hits` through Hugging Face `datasets` and normalizes calorimeter energy aliases.
- `src/collider_fm/views.py` converts events into point views with features `[x, y, z, energy]` plus `offset`, `source_index`, `patch_id`, `mask`, and `view_kind`.
- `src/collider_fm/model.py` contains the current point-level Panda-style self-distillation scaffold, with both a compact diagnostics model and a larger training model.
- `scripts/train.py` runs the current calo-only masked-global training loop with JSONL logging, checkpointing, and terminal/log-friendly `tqdm` progress bars.
- `scripts/plot_diagnostics.py` generates raw-data, view-level, and checkpoint-backed model diagnostics.
- `scripts/plot_training_run.py` plots saved training metrics directly from a completed run directory.
- `notebooks/plot_diagnostics_explorer.ipynb` is the tutorial notebook for the repo workflow.

This is still a research/prototype codebase, but the current calo-only training path now runs end to end on GPU and through SLURM.

## Repository layout

```text
/
├── src/collider_fm/            # Package code for data, views, model, and diagnostics
├── tests/                      # Unit tests for the current pipeline
├── scripts/                    # Download, inspection, training, smoke-test, and plotting scripts
├── notebooks/                  # Tutorial notebook for diagnostics and workflow exploration
├── slurm/                      # Cluster job scripts for setup, downloads, tests, and training
├── apptainer/                  # Container helper scripts
├── Panda_repo/                 # Panda reference submodule
├── logs_slurm/                 # Placeholder directory for SLURM logs
├── markdown/                   # Project markdown docs except AGENTS.md
├── AGENTS.md                   # Short operational guidance for coding agents
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Locked dependency resolution for uv
```

## Data and model contract

- Dataset source: `CERN/ColliderML-Release-1`
- Active modality: `calo_hits` only
- Current point feature contract: `[x, y, z, energy]`
- Current training recipe:
  - two teacher global views
  - two masked student global views
  - point-level distillation on shared `source_index`
  - masked pooled prototype loss per event

The repo intentionally does not preserve the older tracker+calo path in the current runtime scripts.

## Environment setup

The project targets Python 3.12 and uses `uv` for dependency management.

On this cluster, start with:

```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

Notes:

- some CUDA-adjacent packages are better installed on compute nodes than on the login node
- `torch-scatter` is configured to build from source through `uv`
- the documented Hugging Face cache is `/mnt/ceph/users/ewulff/data/hf`

If you prefer a batch setup path, use the SLURM helpers in `slurm/`.

## Common workflows

Download the calo-hit subset used by the current pipeline:

```bash
uv run python scripts/download_data.py --dataset-types ttbar --object-types calo_hits --pu-config pu0 --num-proc 12
```

Inspect a sample event:

```bash
uv run python scripts/inspect_data.py
```

Run the GPU smoke test interactively:

```bash
uv run python scripts/smoke_test_model.py
```

Or through SLURM:

```bash
sbatch slurm/test_model.slurm
```

Run a tiny tracked training pass:

```bash
uv run python scripts/train.py --num-epochs 1 --max-train-batches 1 --max-val-batches 1 --batch-size 1 --log-backend jsonl
```

Run the short SLURM training recipe:

```bash
sbatch slurm/train_small.slurm
```

Run the medium SLURM training recipe:

```bash
sbatch slurm/train_medium.slurm
```

Generate checkpoint-backed diagnostics:

```bash
uv run python scripts/plot_diagnostics.py --checkpoint runs/<run_name>/checkpoints/best.pt
```

Plot the saved metrics from a completed run into `<run_dir>/plots/`:

```bash
uv run python scripts/plot_training_run.py runs/<run_name>
```

Open the tutorial notebook:

```bash
uv run jupyter lab notebooks/plot_diagnostics_explorer.ipynb
```

## Run outputs

Training runs are written under `runs/<run_name>/` and currently include:

- `config.json`
- `metrics.jsonl`
- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- epoch checkpoints such as `checkpoints/epoch_001.pt`

Run-level metric plots can be generated into `runs/<run_name>/plots/`.

Checkpoint-backed diagnostics are typically written under `diagnostics/`.

## Diagnostics and plotting

There are two complementary plotting paths:

- `scripts/plot_diagnostics.py`
  - raw event plots
  - view construction plots
  - checkpoint-backed representation plots
  - optional training metric and schedule plots when `metrics.jsonl` is available
- `scripts/plot_training_run.py`
  - run-level loss curves
  - optimization schedules
  - representation metrics
  - train/val gap plots

The notebook in `notebooks/plot_diagnostics_explorer.ipynb` mirrors the same data-to-view-to-model flow and is intended to be the easiest place to understand the repo.

## Development notes

Format Python files with Black:

```bash
uv run black .
```

Run the current unit tests with:

```bash
uv run python -m unittest tests.test_views tests.test_model tests.test_diagnostics tests.test_experiment_logging
```

Notebook outputs are stripped on commit through `nbstripout` in pre-commit.

## Comet logging

Training always writes JSONL metrics locally. Optional Comet logging can be enabled by either logging in through `~/.comet.config` or exporting:

```bash
export COMET_API_KEY=...
export COMET_PROJECT_NAME=collider-fm
export COMET_WORKSPACE=...
```

Then run training with the default `--log-backend auto` or force Comet with `--log-backend comet`.

## Other docs

- `markdown/PLAN.md`: roadmap and next priorities
- `markdown/HPC.md`: cluster and SLURM guidance
- `markdown/IMPLEMENTATION_GUIDELINES.md`: current design target and gap notes
- `markdown/PANDA_SUMMARY.md`: short Panda-to-repo mapping note
- `AGENTS.md`: short operational instructions for coding agents

## References

- Panda paper: https://arxiv.org/abs/2512.01324
- ColliderML dataset: https://huggingface.co/datasets/ColliderML/ColliderML
- ColliderML website: https://colliderml.github.io/
