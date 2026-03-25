# Project Plan: Simple Calo-Only Panda on ColliderML

This roadmap tracks the in-place migration of the existing `collider_fm` repository toward a clean calorimeter-only pretraining pipeline.
The top priority is clarity.
The code should stay simple, readable, and easy for undergraduate students to follow.

## Working principles

- Prefer small, obvious functions over clever abstractions.
- Keep the main data path easy to trace from dataset -> point view -> model -> loss -> diagnostics.
- Avoid unnecessary catches, indirection, and edge-case machinery.
- Keep only the tests that protect the main behavior.
- Write docs and notebooks that explain how the pieces fit together.

## Phase 1: Simplify the current scaffold

### Stage 1: Data path

- [x] Use `uv` for the project environment.
- [x] Load ColliderML through the Hugging Face `datasets` interface.
- [x] Add a basic dataset download script.
- [x] Make `calo_hits` the default and only training input.
- [x] Remove tracker, particle, and track dependencies from the default runtime path.
- [x] Keep a simple, stable calo event contract built around coordinates, energy, and detector identity.

Progress note: the repo already has a working dataset loader, but the default path is still mixed tracker plus calo. That is the first thing to fix.

### Stage 2: Point views

- [x] Build one event into a simple calo-only point view in `src/collider_fm/views.py`.
- [x] Keep the feature contract easy to read and document clearly.
- [x] Preserve detector identity in a way students can understand.
- [x] Start with simple global-view augmentations that are geometry-safe.
- [~] Add local and masked views only after the base calo path is clear.

Progress note: the current point-view code is centralized already, which is good. It now includes a simple masked-view path that keeps point order fixed so the training logic stays easy to understand.

### Stage 3: Model and training loop

- [x] Keep the PTv3-based Panda scaffold as the starting point.
- [x] Keep the student and EMA teacher structure.
- [x] Simplify the model interfaces so it is obvious what each method returns.
- [x] Move the default SSL objective toward point-level training instead of only pooled event embeddings.
- [ ] Keep the small debug model path for smoke tests.
- [x] Add checkpoint save and reload to the training loop.

Progress note: the current training loop already runs, but the representation path is still more scaffold than final design.

## Phase 2: Practical diagnostics and usability

### Stage 4: Essential diagnostics

- [x] Keep a saved diagnostics script.
- [x] Update diagnostics to the calo-only contract.
- [x] Add simple prototype-usage and embedding-stat summaries.
- [x] Add frozen-embedding export for later probing.
- [ ] Keep the plots understandable rather than overly elaborate.

### Stage 5: Teaching materials

- [ ] Write a tutorial notebook that explains the whole pipeline step by step.
- [ ] Show how raw ColliderML calo data becomes a point view.
- [ ] Show how the model consumes that point view.
- [ ] Show one short training step and how the loss is formed.
- [ ] Show how to inspect saved diagnostics and exported embeddings.

This notebook is a first-class deliverable for this phase, not an afterthought.

## Immediate implementation order

1. Update the codebase to use `calo_hits` only in the default path.
2. Simplify `src/collider_fm/views.py` around a clean calo-only point-view contract.
3. Update `scripts/train.py`, `scripts/smoke_test_model.py`, `scripts/inspect_data.py`, and `scripts/plot_diagnostics.py` to match.
4. Keep only the most useful tests and update them to the new contract.
5. Add checkpoint save or reload and embedding export.
6. Write the tutorial notebook.

## Future work

- [ ] Add richer Panda-style local and masked objectives once the base path is simple and solid.
- [ ] Improve prototype-collapse monitoring if needed.
- [ ] Reintroduce tracker hits later as a second modality, but only after the calo-only path is stable.
- [ ] Add distributed training only after the single-GPU version is easy to understand and debug.
