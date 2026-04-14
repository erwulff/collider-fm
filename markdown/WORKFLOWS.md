# Runtime and Workflows

This document collects the detailed runtime contract, configuration notes, and common local workflows. For the project overview, use `README.md`. For cluster-specific setup and SLURM jobs, use `markdown/HPC.md`.

## Runtime snapshot

- data source: `CERN/ColliderML-Release-1`
- active modality: `calo_hits` only
- point feature contract: `[x, y, z, energy]`
- project split convention on the single Hugging Face `train` split:
  - `train` means `train[:950000]`
  - `val` means `train[950000:1000000]`
  - bounded slices like `val[:100]` stay inside the validation window
- current training recipe: Sonata self-distillation
  - two global views (teacher + masked student) and four local views (student only)
  - Sinkhorn-Knott prototype assignment
  - separate mask/unmask heads
  - cosine schedulers for mask size, mask ratio, temperature, and EMA momentum
  - `match_neighbour` alignment via `origin_coord`
- current training defaults:
  - `training.batch_size=8`
  - `training.mixed_precision=bf16`
  - flash attention disabled by default
  - optional flash backend selection when enabled: `torch` or `flash_attn`

## Config and overrides

The default project config lives at `config/default.yaml`.

Use OmegaConf dotlist overrides such as:

```bash
uv run python scripts/train.py training.batch_size=8 training.num_epochs=10 training.run_name=my_run
```

Common overrides:

- `training.run_dir` and `training.run_name`
- `training.batch_size`, `training.num_epochs`, `training.max_train_batches`, `training.max_val_batches`
- `training.mixed_precision=none|bf16|fp16`
- `model.training.backbone.enable_flash=true`
- `model.training.backbone.flash_backend=torch|flash_attn`
- `data.dataset_revision=...` and `data.local_files_only=true`

To start a training run:

```bash
uv run python scripts/train.py training.num_epochs=20 training.batch_size=8 data.local_files_only=true
```

If you enable flash attention, start with PyTorch's built-in backend:

```bash
uv run python scripts/train.py model.training.backbone.enable_flash=true model.training.backbone.flash_backend=torch
```

## Data and caching

The dataset only exposes a Hugging Face `train` split, so the project reserves the final `50,000` events for validation. `ColliderMLDataset(split="val")` maps to `train[950000:1000000]`, and bounded slices like `split="val[:100]"` are translated inside that validation window.

For reproducible runs, pin the dataset revision and load only from the local cache:

```bash
uv run python scripts/download_data.py download.dataset_types=[ttbar] download.object_types=[calo_hits] download.dataset_revision=e28a24cc9c1641a478ae4e5bc3b376eb624b7283 download.num_proc=12
uv run python scripts/train.py data.dataset_revision=e28a24cc9c1641a478ae4e5bc3b376eb624b7283 data.local_files_only=true
```

## Common local workflows

Download the calo-hit subset used by the current runtime path:

```bash
uv run python scripts/download_data.py download.dataset_types=[ttbar] download.object_types=[calo_hits] download.pu_config=pu0 download.num_proc=12
```

Inspect one cached event:

```bash
uv run python scripts/inspect_data.py
```

Run the GPU smoke test:

```bash
uv run python scripts/smoke_test_model.py
```

Allow the smoke test to fall back to a synthetic CUDA-only check when cached data is unavailable:

```bash
uv run python scripts/smoke_test_model.py smoke_test.allow_synthetic_fallback=true
```

Run a tiny tracked training pass:

```bash
uv run python scripts/train.py training.num_epochs=1 training.max_train_batches=1 training.max_val_batches=1 training.batch_size=1 training.log_backend=jsonl
```

Generate checkpoint-backed diagnostics:

```bash
uv run python scripts/plot_diagnostics.py diagnostics.checkpoint=runs/<run_name>_<timestamp>/checkpoints/best.pt
```

Plot the saved metrics from a completed run:

```bash
uv run python scripts/plot_training_run.py runs/<run_name>_<timestamp>
```

Open the walkthrough notebooks:

```bash
uv run jupyter lab notebooks/dataset_walkthrough.ipynb
uv run jupyter lab notebooks/model_walkthrough.ipynb
uv run jupyter lab notebooks/sonata_views.ipynb
```

For SLURM workflows, see `markdown/HPC.md`.

## Outputs and logging

Training runs are written by default under `runs/<run_name>_<timestamp>/` and typically include:

- `config.json`
- `metrics.jsonl`
- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- per-epoch checkpoints such as `checkpoints/epoch_001.pt`

Run-level metric plots are written into `runs/<run_name>_<timestamp>/plots/`. Checkpoint-backed diagnostics are typically written under `diagnostics/`.

Training always writes local JSONL metrics. Optional Comet logging can be enabled by either a saved `~/.comet.config` or exported variables:

```bash
export COMET_API_KEY=...
export COMET_PROJECT_NAME=collider-fm
export COMET_WORKSPACE=...
```

Then use `training.log_backend=auto` or `training.log_backend=comet`.
