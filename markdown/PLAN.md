# Project Plan

This document tracks the near-term roadmap for the current calo-only training pipeline. It is intentionally shorter than the README and only covers active priorities.

## Current baseline

- ColliderML Release 1 loading through Hugging Face `datasets`
- calo-only runtime path with point features `[x, y, z, energy]`
- point-view batching with masking, patch ids, and `source_index` alignment
- Sonata self-distillation recipe with global/local view crops, Sinkhorn-Knopp prototype assignment, separate mask/unmask heads on both student and teacher (`_head_for_diagnostics` prefers `unmask_head`), cosine schedulers for mask size, mask ratio, teacher temperature, and EMA momentum, and `match_neighbour` alignment via `origin_coord`
- OmegaConf-based config shared across scripts in `config/default.yaml`
- mixed-precision training support and flash attention enabled by default via `flash_attn`
- smoke test, diagnostics plotting, run-level metric plotting, and walkthrough notebooks

## Current limitations

- the evaluation harness is label-free and collapse-focused (stable rank, prototype usage, NN view-retrieval, alignment/uniformity); usefulness probes requiring labels are not yet implemented
- the Sonata recipe uses a much smaller backbone and head than the reference pimm implementation (enc_channels=[16..128] vs [48..512], 4096 vs 4096 prototypes but much smaller head_embed_channels)

## Next priorities

### 1. Better long-run training evidence

- run longer SLURM pretraining jobs with the Sonata recipe and inspect completed curves
- compare completed runs with `scripts/plot_training_run.py`
- confirm that the current smaller backbone and BF16 defaults are still the right tradeoff
- evaluate whether the Sonata spatial hyperparams (mask_size, match_max_r) produce reasonable patch counts and match rates

### 2. Performance characterization

- profile the Sonata forward path on realistic runs rather than just short debug slices
- profile the remaining bottlenecks after restoring `up_cast_level=2`, `head_embed_channels=256`, and the simpler pimm-style head computation
- identify whether the next bottleneck is view count, sparse PTv3 kernels, or the remaining loss and bookkeeping path

### 3. Evaluation and export

- label-free collapse-detection harness shipped: `scripts/evaluate.py` + `src/collider_fm/evaluation.py` report per-point stable rank + singular-value spectrum, per-point prototype usage / entropy / effective count + dead-prototype count, and per-event NN view-retrieval R@1/R@5 + alignment / uniformity, on held-out events through the deterministic teacher backbone. Random-init vs trained comparison is the built-in sanity check (omit `evaluation.checkpoint` for the random baseline).
- follow-up: per-point correspondence retrieval (label-free, via `source_index`) and per-point `particle_id` probing + t-SNE (needs the `contrib_particle_ids` / `contrib_energies` argmax join; metric/clustering-based because `particle_id` is a non-recurring instance ID, so a cross-event linear probe is ill-posed)
- follow-up: export frozen point embeddings for downstream heads

### 4. Documentation and cleanup

- keep `README.md`, `markdown/HPC.md`, and this plan synchronized with the scripts and SLURM jobs that actually exist
- keep generated outputs in `runs/` and `diagnostics/` out of version control by convention
- add focused tests if new plotting, export, or profiling helpers stop being simple scripts

## Later work

- reintroduce tracker hits only after the calo-only path is stable and benchmarked
- explore multimodal fusion after the single-modality backbone is reliable
- add downstream tasks such as probing, tagging, or reconstruction heads
- scale up the Sonata backbone and head to match the reference pimm configuration
