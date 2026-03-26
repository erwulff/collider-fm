# Collider Foundation Model

Collider Foundation Model is a calo-only self-supervised learning project built around ColliderML Release 1 and a Panda-style student-teacher training recipe. The current repository focus is a clean first-stage pipeline for calorimeter hits only: build point-cloud views from raw events, train a masked global self-distillation model, save checkpoints, and generate diagnostics and training-run plots.

The project now uses a shared OmegaConf config file at `config/default.yaml`. That file defines the project defaults for data loading, view construction, model setup, training, diagnostics, and download settings. Runtime scripts keep the merged config as a `DictConfig` during execution and only convert to plain containers at JSON or logging boundaries.

## Current status

- `src/collider_fm/data.py` loads ColliderML `calo_hits` through Hugging Face `datasets` and normalizes calorimeter energy aliases.
- `src/collider_fm/views.py` converts events into point views with features `[x, y, z, energy]` plus `offset`, `source_index`, `patch_id`, `mask`, and `view_kind`.
- `src/collider_fm/model.py` contains the current point-level Panda-style self-distillation scaffold, with both a compact diagnostics model and a larger training model.
- `scripts/train.py` runs the current calo-only masked-global training loop with JSONL logging, checkpointing, and terminal/log-friendly `tqdm` progress bars.
- `scripts/plot_diagnostics.py` generates raw-data, view-level, and checkpoint-backed model diagnostics.
- `scripts/plot_training_run.py` plots saved training metrics directly from a completed run directory.
- `notebooks/dataset_walkthrough.ipynb` explains the Hugging Face dataset and dataloader path.
- `notebooks/model_walkthrough.ipynb` explains point views, augmentations, model structure, training, and SSL validation plots.

This is still a research/prototype codebase, but the current calo-only training path now runs end to end on GPU and through SLURM.

## Repository layout

```text
/
├── src/collider_fm/            # Package code for data, views, model, and diagnostics
├── tests/                      # Unit tests for the current pipeline
├── scripts/                    # Download, inspection, training, smoke-test, and plotting scripts
├── notebooks/                  # Newcomer walkthrough notebooks
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
- Project split convention on the single HF `train` split:
  - `train` means `train[:950000]`
  - `val` means `train[950000:1000000]`
  - `val[:100]` means `train[950000:950100]`
  - requests outside those windows, such as `train[:960000]`, raise an error
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
uv run python scripts/download_data.py download.dataset_types=[ttbar] download.object_types=[calo_hits] download.pu_config=pu0 download.num_proc=12
```

For reproducible runs, you can pin the dataset revision and then load only from the local cache:

```bash
uv run python scripts/download_data.py download.dataset_types=[ttbar] download.object_types=[calo_hits] download.pu_config=pu0 download.dataset_revision=e28a24cc9c1641a478ae4e5bc3b376eb624b7283 download.num_proc=12
uv run python scripts/train.py data.dataset_revision=e28a24cc9c1641a478ae4e5bc3b376eb624b7283 data.local_files_only=true
```

This prevents training or diagnostics from silently following newer dataset commits on the Hub after you already cached a known-good version.

The dataset only exposes a Hugging Face `train` split, so the project reserves the final `50,000` events for validation. `ColliderMLDataset(split="val")` maps to `train[950000:1000000]`, and bounded slices like `split="val[:100]"` are translated inside that validation window.

## Shared config

The default project config lives at `config/default.yaml`.

Use a custom config file with:

```bash
uv run python scripts/train.py --config config/default.yaml
```

Any explicit OmegaConf dotlist overrides override values loaded from the config file.

When extending the runtime code, prefer keeping the merged config as a `DictConfig` and use dot access such as `cfg.data.dataset_revision` or `cfg.training.batch_size`.

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
uv run python scripts/train.py training.num_epochs=1 training.max_train_batches=1 training.max_val_batches=1 training.batch_size=1 training.log_backend=jsonl
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
uv run python scripts/plot_diagnostics.py diagnostics.checkpoint=runs/<run_name>_<timestamp>/checkpoints/best.pt
```

Plot the saved metrics from a completed run into `<run_dir>/plots/`:

```bash
uv run python scripts/plot_training_run.py runs/<run_name>_<timestamp>
```

Open the dataset walkthrough notebook:

```bash
uv run jupyter lab notebooks/dataset_walkthrough.ipynb
```

Open the model and training walkthrough notebook:

```bash
uv run jupyter lab notebooks/model_walkthrough.ipynb
```

## Run outputs

Training runs are written under `runs/<run_name>_<timestamp>/` and currently include:

- `config.json`
- `metrics.jsonl`
- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- epoch checkpoints such as `checkpoints/epoch_001.pt`

Run-level metric plots can be generated into `runs/<run_name>_<timestamp>/plots/`.

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

The two main newcomer notebooks are:

- `notebooks/dataset_walkthrough.ipynb` for the raw dataset and dataloader path
- `notebooks/model_walkthrough.ipynb` for point views, augmentations, model internals, training, and SSL validation

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

Then run training with the default `training.log_backend=auto` or force Comet with `training.log_backend=comet`.

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
