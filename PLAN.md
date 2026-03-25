# Project Plan: Simple Calo-Only Panda on ColliderML

This roadmap tracks the current in-place development of the `collider_fm` repository.
The main priority is still clarity: the code should stay simple, readable, and teachable.

## Current state

- [x] Use ColliderML through the Hugging Face `datasets` interface.
- [x] Keep `calo_hits` as the default and only training input.
- [x] Build a simple point-view contract around coordinates, energy, and ECal/HCal identity.
- [x] Support global, local, and masked student views while keeping point order fixed.
- [x] Train a small Panda-style student/teacher model with checkpoint save or reload.
- [x] Export frozen embeddings from a checkpoint.
- [x] Save diagnostics plots from a checkpoint.
- [x] Save training-run plots from `metrics.jsonl`.
- [x] Provide a step-by-step tutorial notebook.

## Working principles

- Prefer small, obvious functions over clever abstractions.
- Keep the data path easy to trace from dataset -> point view -> model -> loss -> diagnostics.
- Avoid unnecessary catches, indirection, and edge-case machinery.
- Keep tests focused on the main behavior.
- Keep docs aligned with the real repository state.

## What is already working

- [x] A stable SLURM training baseline in `slurm/train_small.slurm`.
- [x] A run-plotting script in `scripts/plot_training_run.py`.
- [x] A richer checkpoint diagnostics script in `scripts/plot_diagnostics.py`.
- [x] Progress reporting in training with `tqdm`.

## Next technical priorities

### 1. Improve SSL validation quality

- [ ] Make the same-event vs different-event similarity gap more informative.
- [ ] Add one or two extra non-collapse indicators that stay simple to explain.
- [ ] Review prototype usage and increase diversity if needed.

### 2. Improve training stability

- [ ] Decide whether the current baseline should stop earlier than 20 epochs.
- [ ] Tune view fractions, temperatures, or learning rate only if the plots suggest a clear issue.
- [ ] Consider a cleaner run-name policy so repeated runs do not append to the same metrics file.

### 3. Keep the repo easy to understand

- [ ] Split large files only if doing so makes the teaching story simpler.
- [ ] Keep notebooks and markdown files synced with the actual code.
- [ ] Keep generated outputs out of git.

## Non-goals right now

- [ ] Reintroducing tracker hits
- [ ] Multimodal fusion
- [ ] Distributed training
- [ ] Large configuration systems
- [ ] Complex Panda matching machinery that makes the code harder to teach
