# Implementation Guidelines

This document describes the next implementation phase for the existing `collider_fm` repository, currently in `collider-fm-yolo`.
It is an in-place migration plan, not a greenfield rewrite.
Keep the current package, scripts, and tests where practical, and refactor incrementally.

## Why this version exists

- The earlier spec assumed a brand-new `colliderml_panda/` package tree.
- The current repository already has useful working scaffolding under `src/collider_fm/`, `scripts/`, and `tests/`.
- The immediate priority is to correct the data scope and learning objective, not to spend a milestone on renaming or reorganizing the repo.

## Hard scope limits

- Use ColliderML Release 1 through the Hugging Face `datasets` interface.
- Use only `calo_hits` as model input in this phase.
- Ignore `tracker_hits`, `tracks`, `particles`, truth labels, and downstream task heads for the default pretraining path.
- Support both ECal and HCal hits from the start; detector identity must be preserved explicitly.
- Train only a reusable backbone plus pretraining heads.
- Keep interfaces open so tracker hits can be added later as a second modality, but do not build multimodal fusion now.

## Current baseline to keep and evolve

- `src/collider_fm/data.py` already loads ColliderML from Hugging Face.
- `src/collider_fm/model.py` already contains a PTv3-based Panda-style student/teacher scaffold.
- `src/collider_fm/views.py` already builds point-cloud views and batching utilities.
- `scripts/train.py`, `scripts/smoke_test_model.py`, `scripts/inspect_data.py`, and `scripts/plot_diagnostics.py` already provide a usable prototype workflow.
- `tests/` already provides a basic regression harness.

This baseline should be evolved, not discarded.

## Main mismatches to fix first

- The repo still assumes mixed tracker plus calorimeter inputs in several places; the training path must become calo-only.
- The current SSL path is event-pooled; the target is point-level Panda-style pretraining.
- The current augmentations are simple noise jitter; the target is global, local, and masked calorimeter views.
- The current detector feature is tracker-vs-calo oriented; the target is ECal/HCal-aware calo identity.
- Checkpoint save or resume, frozen-embedding export, and collapse-oriented diagnostics are still incomplete.

## Implementation principles

- Keep the package name `collider_fm` for this phase.
- Prefer small extractions and interface cleanups over a repo-wide rename.
- Keep the code runnable after each step; do not leave the project in a half-migrated state.
- Use the `Panda_repo/` submodule as the architecture reference when needed.
- Use SLURM jobs for heavy dataset or GPU work; keep interactive validation lightweight.

## Target repo shape for this phase

The target is still the current repo layout, with cleaner responsibilities.

- `src/collider_fm/data.py`
  - Keep Hugging Face loading.
  - Make `calo_hits` the default and primary training path.
  - Normalize calorimeter energy aliases and expose a stable calo-only event contract.
- `src/collider_fm/views.py`
  - Own calo event building, detector identity handling, batching, global/local view generation, and masking.
  - If this file becomes too large, split it later into smaller modules such as `schema.py`, `event_builder.py`, `augmentations.py`, and `masking.py`, but only after interfaces stabilize.
- `src/collider_fm/model.py`
  - Keep the PTv3 backbone and student/teacher wrapper.
  - Move the SSL objective from event-level pooled embeddings to point-level outputs.
  - Keep the projection and prototype heads separate from the reusable backbone.
- `src/collider_fm/diagnostics.py`
  - Add helpers for checkpoint loading, embedding export, prototype statistics, and collapse diagnostics.
- `scripts/`
  - Keep `train.py`, `smoke_test_model.py`, `inspect_data.py`, and `plot_diagnostics.py`.
  - Add `export_embeddings.py`.
  - Add `build_event_stats.py` if data-driven defaults are needed before finalizing crop and masking settings.
- `tests/`
  - Update the existing data and view tests to the calo-only contract.
  - Add tests for masking, multi-view batching, point-level loss wiring, and checkpoint reload.

## Detailed spec

### 1. Dataset and sample definition

- Load ColliderML Release 1 `calo_hits` through the Hugging Face `datasets` API.
- Treat each event as a sparse 3D point cloud with `coord = [x, y, z]` and per-point calorimeter features.
- Keep coordinates separate from features.
- The target feature contract should minimally preserve deposited energy and detector identity.
- If the current code temporarily keeps compatibility features such as duplicated coordinates in `feat`, treat that as a migration scaffold rather than the final design.
- Make ECal/HCal identity explicit from the dataset schema. If the raw table provides a finer-grained detector identifier, preserve that raw ID and expose a stable ECal/HCal mapping in one place.
- Reject empty events.
- Add configurable caps on point counts for memory control.
- Allow an optional stricter energy cut only as a memory-management tool, not as the default physics definition.
- Keep the released calorimeter energies by default; do not silently recalibrate the dataset in the default path.

### 2. Geometry and preprocessing

- Preserve detector geometry and cylindrical semantics.
- Use detector-safe transforms such as:
  - azimuthal rotation around the beam axis,
  - small coordinate jitter,
  - small multiplicative energy jitter,
  - contiguous local crops,
  - masked patch dropping,
  - random point dropout.
- Do not use arbitrary 3D rotations or mirror flips as the default, because those can break detector semantics.
- Keep a direct point-cloud path as the primary implementation.
- Optional sparse voxelization can be added later if memory requires it, but it is not the first milestone.
- If crop sizes, mask fractions, or point caps are uncertain, add `scripts/build_event_stats.py` and choose defaults from measured event statistics.

### 3. Backbone and representation

- Keep the Panda-inspired Point Transformer V3 backbone direction.
- Use a sparse stem plus 4 to 5 encoder stages with stride-2 downsampling between stages.
- Do not add a decoder in this phase.
- The target reusable representation is point-level, not event-level.
- Pooled event embeddings may still be used for diagnostics, but not as the main SSL target.
- Keep the projection head separate from the backbone.
- Target a 256-dimensional normalized latent space and a prototype layer with Panda-like scaling as the main baseline.
- A smaller debug configuration is acceptable for smoke tests, but the code path should cleanly scale to the intended baseline rather than baking in tiny-only assumptions.
- The long-term target is a multiscale point embedding path. If a simpler last-stage point-feature path is used briefly to land the point-level training loop, treat that as an intermediate step.

### 4. Panda-style pretraining objective

- Use a student encoder and an EMA teacher encoder with the same backbone architecture.
- The teacher should consume clean global views.
- The student should consume global, local, and masked views.
- Use two global views per event as the default starting point.
- Add several local views and masked global variants after the base global-view path is stable.
- Do not add a `[cls]` token or event summary token.
- Match teacher and student distributions at the point level.
- Implement two main losses:
  - local-view loss, where student local crops match the corresponding region in an unmasked teacher global view,
  - masked-view loss, where masked student points match the teacher on the masked points only.
- Keep stabilization close to Panda's recipe:
  - low student temperature,
  - teacher temperature warmup,
  - prototype normalization,
  - centering and/or Sinkhorn-style balancing,
  - EMA teacher momentum schedule,
  - AdamW with warmup and cosine decay as the default baseline.

### 5. Runtime and outputs

- Keep the current script-driven workflow for this phase.
- A full YAML config tree is not the first blocker; land the calo-only point-level path first, then extract configuration once interfaces settle.
- `scripts/train.py` must support:
  - calo-only training,
  - checkpoint save and resume,
  - clear separation between reusable backbone weights and pretraining-only heads.
- Add `scripts/export_embeddings.py` to write frozen point embeddings for held-out events.
- It is acceptable to keep using `runs/` and `diagnostics/` rather than introducing a brand-new top-level `outputs/` tree immediately.
- Store checkpoints and exported embeddings in a stable run-local structure so downstream probing can reuse them.

### 6. Validation and acceptance criteria

The first milestone is complete when all of the following are true:

- `scripts/inspect_data.py` can inspect a calo-only split without depending on tracker or particle tables.
- The default dataset and view code path uses `calo_hits` only.
- `scripts/smoke_test_model.py` can build calo-only views and run one forward or loss step.
- `scripts/train.py` can run a short calo-only SSL job, save checkpoints, and reload them.
- The training objective operates on point-level outputs rather than only pooled event embeddings.
- `scripts/export_embeddings.py` can export frozen embeddings from a saved checkpoint.
- Diagnostics include at least:
  - prototype usage statistics,
  - embedding norm statistics,
  - nearest-prototype entropy or an equivalent collapse signal,
  - a lightweight representation-geometry view such as PCA now, with UMAP or t-SNE optional once the pipeline is stable.
- Tests cover the calo-only data contract, view generation, loss wiring, and checkpoint reload.

## Recommended implementation order

1. Cut the runtime over to calo-only data loading and remove tracker, track, and particle dependencies from the default path.
2. Define the stable calo event contract, including explicit ECal/HCal identity.
3. Add event filtering and dataset-statistics helpers so point caps and crop settings are data-driven.
4. Replace the current noise-only augmentation path with global, local, and masked calo-safe view generation.
5. Move the SSL objective in `src/collider_fm/model.py` from pooled event embeddings to point-level outputs.
6. Add the missing stabilization pieces: prototype normalization, teacher temperature warmup, EMA scheduling, and balancing.
7. Add checkpoint save or resume and frozen-embedding export.
8. Expand diagnostics and tests to measure collapse resistance and representation quality.
9. Only after the single-GPU calo-only path is correct, add broader config cleanup and distributed runtime support.

## Non-goals for this phase

- Do not reintroduce tracker hits into the default pretraining path.
- Do not build multimodal fusion yet.
- Do not add downstream supervised heads yet.
- Do not spend the milestone on a package rename or exact tree rewrite.
- Do not prioritize distributed training over correctness of the single-GPU calo-only objective.

## Short brief for future coding agents

Use the existing `collider_fm` package and migrate it in place toward a calo-only Panda-style SSL pipeline on ColliderML Release 1.
Remove tracker-dependent assumptions from the default path, preserve ECal/HCal identity, replace event-pooled SSL with point-level global/local/masked training, add checkpoint and embedding export support, and extend diagnostics so representation collapse can be detected early.
