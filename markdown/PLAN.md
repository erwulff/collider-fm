# Project Plan

This document tracks the near-term roadmap for the current calo-only training pipeline. It is intentionally shorter than the README and only covers active priorities.

## Current baseline

- ColliderML Release 1 loading through Hugging Face `datasets`
- calo-only runtime path with point features `[x, y, z, energy]`
- point-view batching with masking, patch ids, and `source_index` alignment
- Panda-inspired PTv3 student-teacher training with:
  - point-level prototype matching
  - masked pooled prototype loss
  - EMA teacher updates and running center stabilization
- OmegaConf-based config shared across scripts in `config/default.yaml`
- mixed-precision training support and optional flash backends when explicitly enabled
- smoke test, diagnostics plotting, run-level metric plotting, and walkthrough notebooks

## Current limitations

- the training recipe is still a simplified masked-global Panda-style setup rather than full paper parity
- explicit local student crops are not implemented yet
- there is no downstream evaluation or frozen-embedding export path yet
- the checked-in SLURM jobs are partly examples and are not all aligned with the current tuned defaults

## Next priorities

### 1. Better long-run training evidence

- run longer SLURM pretraining jobs and inspect completed curves
- compare completed runs with `scripts/plot_training_run.py`
- confirm that the current smaller backbone and BF16 defaults are still the right tradeoff

### 2. Performance characterization

- profile the PTv3 path on realistic runs rather than just short debug slices
- compare `model.training.backbone.flash_backend=torch` against `flash_attn` when both are available
- identify whether the next bottleneck is view count, sparse PTv3 kernels, or the remaining loss and bookkeeping path

### 3. Closer Panda-style views

- add explicit local student crops in addition to masked global views
- make the view builder closer to the intended global/local/masked structure
- add clearer per-view diagnostics for surviving point counts after crop, dropout, and masking

### 4. Evaluation and export

- export frozen point embeddings from saved checkpoints
- add a lightweight probing or evaluation path for held-out events
- add utilities to compare random-init and trained checkpoints across multiple runs

### 5. Documentation and cleanup

- keep `README.md`, `markdown/HPC.md`, and this plan synchronized with the scripts and SLURM jobs that actually exist
- keep generated outputs in `runs/` and `diagnostics/` out of version control by convention
- add focused tests if new plotting, export, or profiling helpers stop being simple scripts

## Later work

- reintroduce tracker hits only after the calo-only path is stable and benchmarked
- explore multimodal fusion after the single-modality backbone is reliable
- add downstream tasks such as probing, tagging, or reconstruction heads
