# Implementation Guidelines

This document describes the current implementation target for the existing `collider_fm` repository.
It is an in-place plan for a simple, calo-only self-supervised learning pipeline.

## Current target

Build and maintain a small, easy-to-understand Panda-style SSL baseline on ColliderML Release 1 using only `calo_hits`.

The code should be understandable by undergraduate students.
That means:

- keep the data flow explicit
- use simple view transformations
- avoid unnecessary abstraction
- prefer a clear baseline over a clever one

## What is implemented now

- `src/collider_fm/data.py`
  - loads ColliderML tables from Hugging Face
  - defaults to `calo_hits`
  - normalizes calorimeter energy aliases
- `src/collider_fm/views.py`
  - builds calo-only point views
  - provides global, local, and masked views
  - uses explicit `hidden_mask` and `loss_mask`
- `src/collider_fm/model.py`
  - contains a small Panda-style student/teacher model
  - matches teacher and student distributions point by point
- `scripts/train.py`
  - trains the model
  - saves checkpoints and run metrics
  - reports progress with `tqdm`
- `scripts/export_embeddings.py`
  - exports frozen embeddings from a checkpoint
- `scripts/plot_diagnostics.py`
  - saves checkpoint-level diagnostics
- `scripts/plot_training_run.py`
  - saves run-level loss and held-out similarity plots
- `notebooks/calo_pipeline_tutorial.ipynb`
  - explains the full pipeline step by step

## Stable contracts

### 1. Data contract

- use ColliderML Release 1
- use only `calo_hits` in the default path
- preserve raw coordinates `x`, `y`, `z`
- preserve deposited energy
- preserve detector identity through a simple ECal or HCal mapping

### 2. Point-view contract

Each point view should contain:

- `coord`
- `feat`
- `offset`
- `grid_size`
- `energy`
- `detector_id`
- `calo_type`
- `hidden_mask`
- `loss_mask`

Keep point order fixed across all views in this phase.
That design choice is deliberate because it keeps the loss logic simple.

### 3. View contract

- global views
  - student and teacher both use them
  - no points hidden
  - all points contribute to the loss
- local views
  - student-only
  - points outside one local neighborhood are hidden
  - only the kept neighborhood contributes to the loss
- masked views
  - student-only
  - randomly selected points are hidden
  - only masked points contribute to the loss

### 4. Training contract

The baseline training script should support:

- one-GPU training
- checkpoint save or reload
- `metrics.jsonl` output under `runs/<run-name>/`
- progress reporting in terminal and SLURM logs

## Runtime workflow

The intended workflow is:

1. download data with `scripts/download_data.py`
2. inspect data with `scripts/inspect_data.py`
3. validate the stack with `scripts/smoke_test_model.py`
4. run training with `scripts/train.py` or `slurm/train_small.slurm`
5. plot the run with `scripts/plot_training_run.py`
6. generate richer checkpoint diagnostics with `scripts/plot_diagnostics.py`
7. export embeddings with `scripts/export_embeddings.py`

## What is intentionally not in scope

- tracker hits in the default path
- multimodal fusion
- downstream supervised heads
- distributed training
- a large config system
- complex crop matching that makes the baseline hard to teach

## Next improvements to consider

- improve the held-out SSL validation signal
- improve prototype usage diversity if needed
- make repeated run naming safer so metrics are not appended accidentally
- split large files only if that makes the project easier to explain

## Short brief for future coding agents

Keep the project calo-only, simple, and readable.
Preserve the current point-view contract and fixed-order view design unless there is a strong reason to change it.
Prefer code and docs that make the student -> view -> loss path easy to explain.
