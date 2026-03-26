# Implementation Guidelines

This document records the intended design target for the current ColliderML pretraining direction and notes where the checked-in code already matches it.

## Scope

- Data source: ColliderML Release 1
- Active modality: calorimeter hits only
- Current feature contract: `[x, y, z, energy]`
- Near-term goal: reusable point-level backbone pretraining, not downstream task heads

Tracker hits, multimodal fusion, and downstream reconstruction heads remain out of scope for the current runtime path.

## Data pipeline target

- Load ColliderML through Hugging Face `datasets`
- Build each event into a point cloud with `[x, y, z, energy]`
- Keep enough bookkeeping to support teacher/student alignment and masking

Current status:

- implemented in `src/collider_fm/data.py`
- implemented in `src/collider_fm/views.py`
- current bookkeeping fields include `offset`, `source_index`, `patch_id`, `mask`, and `view_kind`
- project-level split aliases map `train` to `train[:950000]` and `val` to `train[950000:1000000]`, with bounded sub-slices such as `val[:100]`

## Backbone target

- Panda-inspired hierarchical point encoder based on Point Transformer V3
- point-level outputs rather than event-only pooled outputs
- projection head plus prototype layer kept separate from the backbone

Current status:

- implemented in simplified form in `src/collider_fm/model.py`
- current runtime supports a compact diagnostics model and a larger training model
- the training path uses PTv3 upcasting so losses can be applied at original point resolution

## Pretraining target

- student/teacher EMA training
- teacher sees cleaner global views
- student sees masked or otherwise harder views
- use prototype-distribution matching with temperature asymmetry and center stabilization

Current status:

- implemented as a simplified masked-global point-level recipe
- current view builder returns two teacher global views and two masked student global views
- current loss combines:
  - point-level prototype matching on shared `source_index`
  - masked pooled prototype matching per event

## Augmentation target

Collider-safe transforms should stay consistent with detector geometry.

Current implemented transforms:

- azimuthal rotation around the beam axis
- coordinate jitter
- energy jitter
- contiguous cropping
- point dropout
- coarse patch masking

Still missing relative to the full intended recipe:

- explicit local student crop views as a first-class training path
- a closer match to the full Panda global/local/masked view mix

## Training pipeline target

- stream cached ColliderML calo-hit events
- build multi-view batches
- train student/teacher model
- save checkpoints
- log validation-friendly metrics
- support short real GPU runs and SLURM launches

Current status:

- implemented in `scripts/train.py`
- current outputs include `config.json`, `metrics.jsonl`, and checkpoints under `runs/<run_name>_<timestamp>/checkpoints/`
- current metrics include loss, prototype entropy, embedding norm, masked fraction, teacher schedules, and center norm
- dataset loading can now pin a Hugging Face revision and optionally require local cached files only
- project defaults are now centralized in `config/default.yaml` and loaded through OmegaConf
- runtime scripts now keep the merged config as `DictConfig` and only materialize plain containers when writing JSON or logging artifacts

## Diagnostics target

- tutorial notebook explaining the repo workflow
- scriptable diagnostics for raw data, views, and checkpoint-backed representations
- run-level plotting for completed training jobs

Current status:

- `notebooks/dataset_walkthrough.ipynb` covers the dataset and dataloader path
- `notebooks/model_walkthrough.ipynb` covers views, model internals, training, and SSL validation
- `scripts/plot_diagnostics.py` covers raw/view/model diagnostics and can use checkpoints plus `metrics.jsonl`
- `scripts/plot_training_run.py` writes run-level metric plots into a subfolder inside the run directory

## Remaining high-value work

- run longer pretraining jobs and compare completed runs
- add explicit local-view training support
- add frozen-embedding export utilities
- add lightweight downstream evaluation or probing scripts
