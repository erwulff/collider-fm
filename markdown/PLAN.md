# Project Plan

This document tracks the near-term roadmap for the current calo-only training pipeline. It is intentionally shorter than the README and only covers active priorities.

## Current baseline

- ColliderML Release 1 loading through Hugging Face `datasets`
- calo-only runtime path with point features `[x, y, z, energy]`
- point-view batching with masking, patch ids, and `source_index` alignment
- Two training recipes selectable via `model.recipe`:
  - **legacy**: Panda-inspired PTv3 student-teacher with point-level prototype matching, masked pooled loss, EMA teacher, and running center stabilization
  - **sonata**: Sonata self-distillation with global/local view crops, Sinkhorn-Knott prototype assignment, separate mask/unmask heads, cosine schedulers for mask size, mask ratio, teacher temperature, and EMA momentum, and `match_neighbour` alignment via `origin_coord`
- OmegaConf-based config shared across scripts in `config/default.yaml`
- mixed-precision training support and optional flash backends when explicitly enabled
- smoke test, diagnostics plotting, run-level metric plotting, and walkthrough notebooks

## Current limitations

- the legacy training recipe is a simplified masked-global Panda-style setup rather than full paper parity
- there is no downstream evaluation or frozen-embedding export path yet
- the Sonata recipe uses a much smaller backbone and head than the reference pimm implementation (enc_channels=[16..128] vs [48..512], 256 vs 4096 prototypes)

## Next priorities

### 1. Better long-run training evidence

- run longer SLURM pretraining jobs with the Sonata recipe and inspect completed curves
- compare completed runs with `scripts/plot_training_run.py`
- confirm that the current smaller backbone and BF16 defaults are still the right tradeoff
- evaluate whether the Sonata spatial hyperparams (mask_size, match_max_r) produce reasonable patch counts and match rates

### 2. Performance characterization

- profile the Sonata forward path on realistic runs rather than just short debug slices
- compare `model.sonata_training.backbone.flash_backend=torch` against `flash_attn` when both are available
- identify whether the next bottleneck is view count, sparse PTv3 kernels, or the remaining loss and bookkeeping path

### 3. Evaluation and export

- export frozen point embeddings from saved checkpoints
- add a lightweight probing or evaluation path for held-out events
- add utilities to compare random-init and trained checkpoints across multiple runs

### 4. Documentation and cleanup

- keep `README.md`, `markdown/HPC.md`, and this plan synchronized with the scripts and SLURM jobs that actually exist
- keep generated outputs in `runs/` and `diagnostics/` out of version control by convention
- add focused tests if new plotting, export, or profiling helpers stop being simple scripts

## Later work

- reintroduce tracker hits only after the calo-only path is stable and benchmarked
- explore multimodal fusion after the single-modality backbone is reliable
- add downstream tasks such as probing, tagging, or reconstruction heads
- scale up the Sonata backbone and head to match the reference pimm configuration
