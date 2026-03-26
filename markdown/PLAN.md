# Project Plan: Panda-Style SSL on ColliderML

This document tracks the near-term roadmap for the current calo-only training pipeline. It is meant to reflect the code that actually exists today, plus the next concrete steps.

## What is done

### Data and views

- [x] Load ColliderML Release 1 through Hugging Face `datasets`
- [x] Default the runtime path to `calo_hits` only
- [x] Normalize calorimeter energy aliases to a stable `energy` field
- [x] Build model-ready point views with `[x, y, z, energy]`
- [x] Carry `offset`, `source_index`, `patch_id`, `mask`, and `view_kind` through the view pipeline

### Model and losses

- [x] Vendor the PTv3 pieces needed for the current runtime path under `src/collider_fm/_panda/`
- [x] Implement a readable student/teacher self-distillation model scaffold
- [x] Support both a small diagnostics model and a larger training model
- [x] Use point-level distillation on matched `source_index`
- [x] Add a masked pooled prototype loss per event
- [x] Normalize prototypes and maintain a running teacher center

### Training and diagnostics

- [x] Implement a CUDA training loop with AdamW, checkpointing, and JSONL metrics
- [x] Add train/validation metric logging for loss, prototype entropy, embedding norm, masked fraction, schedules, and center norm
- [x] Add terminal/log-friendly `tqdm` progress bars
- [x] Add a GPU smoke test for the current calo-only path
- [x] Add checkpoint-backed diagnostics plots in `scripts/plot_diagnostics.py`
- [x] Add run-level metric plotting in `scripts/plot_training_run.py`
- [x] Add newcomer walkthrough notebooks for the dataset/dataloader path and the model/training path
- [x] Submit and validate short SLURM training runs

## Current limitations

- [ ] The training recipe is still a simplified masked-global Panda-style setup rather than full paper parity.
- [ ] We only use two teacher global views and two masked student global views; local student crops are not implemented yet.
- [ ] The current objective and metrics are useful for debugging and early validation, but not yet enough for strong physics-facing claims.
- [ ] There is no downstream evaluation script yet.
- [ ] There is no frozen-embedding export utility yet.

## Next priorities

### Priority 1: stronger pretraining runs

- [ ] Run a longer medium-scale SLURM pretraining job and inspect the resulting curves
- [ ] Tune batch size, train/val slice size, and augmentation strengths for more informative validation behavior
- [ ] Compare multiple completed runs with the run-plotting script

### Priority 2: closer Panda-style views

- [ ] Add explicit local student crops in addition to masked global views
- [ ] Make the augmentation/view builder more closely mirror the intended global/local/masked Panda structure
- [ ] Add clearer per-view diagnostics for how many points survive crops, dropout, and masking

### Priority 3: evaluation and export

- [ ] Add a script to export frozen point embeddings from saved checkpoints
- [ ] Add a lightweight evaluation or probing path for held-out events
- [ ] Add comparison utilities for random-init vs trained checkpoints across multiple runs

### Priority 4: documentation and polish

- [ ] Keep README, PLAN, and HPC docs synchronized with the current scripts and SLURM jobs
- [ ] Add focused unit tests for any new plotting or export helpers that grow beyond simple scripts
- [ ] Decide which generated diagnostics and run summaries should be kept outside git by convention

## Later work

- [ ] Add tracker hits back as a second modality after the calo-only path is stable
- [ ] Explore multimodal fusion after the single-modality backbone is reliable
- [ ] Add downstream tasks such as probing, tagging, or reconstruction heads
- [ ] Revisit paper-parity details such as balancing variants and larger prototype counts if needed
