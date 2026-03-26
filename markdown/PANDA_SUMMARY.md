# Panda Summary

This note is a short practical summary of Panda as it relates to the current ColliderFM codebase.

## Core idea

Panda is a sensor-level foundation model that learns point-level detector representations directly from sparse 3D detector activity. A shared backbone is pretrained with self-distillation, then reused for downstream tasks.

For this repository, the useful mental model is:

- a hierarchical point-native encoder
- student/teacher self-distillation
- prototype targets rather than labels
- point-level learning rather than a single global event token

## Relevant parts for this repo

The current ColliderFM implementation borrows these main ideas:

- Point Transformer V3 as the backbone family
- EMA teacher updated from the student
- projection head plus prototype layer
- masked-view learning and teacher/student agreement
- temperature asymmetry and a running center for stabilization

## Important differences from the paper

The current repo does not try to reproduce full Panda parity yet.

Current simplifications:

- calo hits only rather than Panda's original detector setting
- a smaller, easier-to-read point-level training model
- two teacher global views and two masked student global views
- no decoder and no downstream task heads
- no full local-view training path yet

So the present code should be understood as a Panda-inspired ColliderML pretraining scaffold, not a paper-faithful reimplementation.

## How it maps to the current code

- `src/collider_fm/_panda/`
  - vendored PTv3 components used by the runtime path
- `src/collider_fm/views.py`
  - builds the point-view and masked-global training views
- `src/collider_fm/model.py`
  - student/teacher model, projection heads, prototype head, EMA update, and losses
- `scripts/train.py`
  - current training loop and schedules
- `Panda_repo/`
  - reference submodule for reading the original implementation

## What to remember

If you need one concise summary for this repo, use this:

Panda is the design reference for a point-level student-teacher SSL pipeline, and this repository currently implements a simpler calo-only version of that idea on ColliderML.
