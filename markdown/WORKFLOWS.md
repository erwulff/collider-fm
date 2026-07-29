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
  - Sinkhorn-Knopp prototype assignment
  - separate student mask/unmask heads with matching teacher mask/unmask heads (`_head_for_diagnostics` prefers `unmask_head`)
  - cosine schedulers for mask size, mask ratio, temperature, and EMA momentum
  - `match_neighbour` alignment via `origin_coord`
- current training defaults:
  - `training.batch_size=8` (per-GPU; global batch = `batch_size * num_gpus`)
  - `training.num_gpus=1`
  - `training.mixed_precision=bf16`
  - flash attention enabled by default via `flash_attn`
  - `views.grid_sample_enabled=true`, `views.grid_sample_size=0.002`
  - `model.training.up_cast_level=2`
  - `model.training.head_embed_channels=256`

## Config and overrides

The default project config lives at `config/default.yaml`.

Use OmegaConf dotlist overrides such as:

```bash
uv run python scripts/train.py training.batch_size=8 training.num_epochs=10 training.run_name=my_run
```

To place the run under a custom experiment directory:

```bash
uv run python scripts/train.py training.experiment_dir=/tmp/collider-runs training.run_name=my_run
```

Common overrides:

- `training.experiment_dir` and `training.run_name` (local run directory resolves to `<experiment_dir>/<run_name>`)
- `training.resume=true|false`
- `training.num_gpus` (per-GPU batch size; global batch scales with GPU count)
- `training.batch_size`, `training.num_epochs`, `training.max_train_batches`, `training.max_val_batches`
- `training.mixed_precision=none|bf16|fp16`
- `model.training.backbone.enable_flash=true`
- `data.dataset_revision=...` and `data.local_files_only=true`
- `evaluation.checkpoint`, `evaluation.val_split`, `evaluation.max_events`, `evaluation.point_subsample_budget` (see `scripts/evaluate.py`)

To start a single-GPU training run:

```bash
uv run python scripts/train.py training.num_epochs=20 training.batch_size=8 data.local_files_only=true
```

To start a multi-GPU run:

```bash
uv run python scripts/train.py training.num_gpus=4 training.batch_size=8 data.local_files_only=true
```

To disable flash attention for comparison:

```bash
uv run python scripts/train.py model.training.backbone.enable_flash=false
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
uv run python scripts/plot_diagnostics.py diagnostics.checkpoint=/mnt/ceph/users/ewulff/raytrain_results/<run_name>/checkpoint_000015/
```

Plot the saved metrics from a completed run:

```bash
uv run python scripts/plot_training_run.py runs/<run_name>
```

Evaluate pretraining quality with the label-free collapse-detection harness. It encodes held-out events through the deterministic teacher backbone and reports per-point stable rank + singular-value spectrum, per-point prototype usage / entropy / effective count + dead-prototype count, and per-event NN view-retrieval R@1/R@5 + alignment / uniformity. Metrics go to `runs/eval_<run_name>/metrics_step.jsonl` and `summary.json`.

```bash
# trained checkpoint: a single model.pt, a runs/<run> dir, or a Ray storage dir all resolve
uv run python scripts/evaluate.py evaluation.checkpoint=runs/<run_name> data.local_files_only=true

# random-init baseline (omit evaluation.checkpoint) for the trained-vs-random sanity comparison
uv run python scripts/evaluate.py data.local_files_only=true

# smaller / faster eval
uv run python scripts/evaluate.py evaluation.checkpoint=runs/<run_name> evaluation.max_events=500 evaluation.val_split=val[:500]
```

The `evaluation.checkpoint` argument accepts a direct `model.pt` path, a local `runs/<run_name>` directory (resolved via `checkpoint_path.txt` to the latest `checkpoint_*/model.pt`), or a Ray storage directory containing `checkpoint_*` subdirs. The teacher backbone is used because it is deterministic (drop_path / attn_drop / proj_drop forced off) and is the deployment-target network.

Open the walkthrough notebooks:

```bash
uv run jupyter lab notebooks/dataset_walkthrough.ipynb
uv run jupyter lab notebooks/model_walkthrough.ipynb
uv run jupyter lab notebooks/sonata_views.ipynb
```

For SLURM workflows, see `markdown/HPC.md`.

## Outputs and logging

Training runs write two kinds of artifacts:

**Local run directory** (`runs/<run_name>/` by default):
- `config.json`
- `metrics_step.jsonl`
- `metrics_epoch.jsonl`
- `viz/` (diagnostic PNGs)
- `checkpoint_path.txt` (points to the Ray checkpoint directory)

**Evaluation run directory** (`runs/eval_<run_name>/` from `scripts/evaluate.py`):
- `config.json`
- `metrics_step.jsonl` (one `record_type: "eval"` record with all v1 metrics)
- `summary.json` (the same metrics, pretty-printed for quick inspection)

**Ray checkpoint storage** (`/mnt/ceph/users/ewulff/raytrain_results/<run_name>/`):
- `checkpoint_000000/`, `checkpoint_000001/`, ... (each contains `model.pt`, `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `training_state.pt`)
- Ray keeps the 3 best checkpoints by `val_loss` and prunes the rest

Fresh runs are explicit:

- If `training.experiment_dir` is unset, the local experiment directory defaults to `runs/` under the project root.
- If `training.run_name` is set, that value is used verbatim for both the local run directory name and the Ray storage folder name.
- If `training.run_name` is unset, a unique `run_<timestamp>` ID is generated for a fresh run.
- The local run directory is always resolved as `<experiment_dir>/<run_name>`.
- A fresh run fails if the target local run directory or Ray storage directory already exists.

Resume is also explicit:

- `training.resume=true` requires an explicit `training.run_name`.
- Resume fails unless the local run directory already exists and a valid checkpoint is present under `/mnt/ceph/users/ewulff/raytrain_results/<run_name>/`.

If you used a non-default `training.experiment_dir` for the original run, pass the same `training.experiment_dir` again when resuming.

To resume a run, re-run with the same `training.run_name` and set `training.resume=true`:

```bash
uv run python scripts/train.py training.run_name=my_run training.resume=true training.num_gpus=4 data.local_files_only=true
```

Resume from a custom experiment directory:

```bash
uv run python scripts/train.py training.experiment_dir=/tmp/collider-runs training.run_name=my_run training.resume=true training.num_gpus=4 data.local_files_only=true
```

Generate checkpoint-backed diagnostics by pointing at a Ray checkpoint directory:

```bash
uv run python scripts/plot_diagnostics.py diagnostics.checkpoint=/mnt/ceph/users/ewulff/raytrain_results/<run_name>/checkpoint_000015/
```

Plot the saved metrics from a local run directory:

```bash
uv run python scripts/plot_training_run.py runs/<run_name>
```

Training always writes local JSONL metrics. Optional Comet logging can be enabled by either a saved `~/.comet.config` or exported variables:

```bash
export COMET_API_KEY=...
export COMET_PROJECT_NAME=collider-fm
export COMET_WORKSPACE=...
```

Then use `training.log_backend=auto` or `training.log_backend=comet`.
