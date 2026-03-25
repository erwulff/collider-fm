# Collider Foundation Model

Collider Foundation Model is a small research codebase for learning reusable representations from collider calorimeter data.
The current phase is intentionally narrow and simple: calo-only self-distillation on ColliderML Release 1.

## Current focus

- Use only `calo_hits` from ColliderML.
- Build one event into a sparse 3D point cloud.
- Train a small Panda-style student/teacher model.
- Include one simple masked-view variant without making the code hard to follow.
- Keep the code easy to read and easy to teach.

The main data path is:

`ColliderMLDataset` -> `build_point_view_from_event` -> `build_distillation_views` -> `PandaSelfDistillation` -> diagnostics or exported embeddings.

## Repository layout

```text
/
├── src/collider_fm/            # Core package code
├── tests/                      # Small focused tests
├── scripts/                    # Download, inspect, train, diagnostics, export
├── notebooks/                  # Tutorial and exploration notebooks
├── slurm/                      # Cluster job scripts
├── Panda_repo/                 # Panda reference submodule
├── pyproject.toml              # Project metadata and dependencies
├── PLAN.md                     # Implementation roadmap
├── IMPLEMENTATION_GUIDELINES.md
└── HPC.md                      # Cluster notes
```

## Environment setup

The project uses `uv` and targets Python 3.12.

```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

Some packages such as `spconv` and `torch-scatter` may need to be installed on compute nodes.

## Dataset

The project uses ColliderML Release 1:

- Hugging Face: https://huggingface.co/datasets/ColliderML/ColliderML
- Project site: https://colliderml.github.io/

This phase uses only `calo_hits`.

Each point keeps:

- 3D coordinates: `x`, `y`, `z`
- deposited energy
- a simple calorimeter type flag: ECal or HCal

On this cluster, the documented Hugging Face cache location is `/mnt/ceph/users/ewulff/data/hf`.

## Common workflows

Download the calo-hit subset:

```bash
uv run python scripts/download_data.py --pu-config pu0 --num-proc 12
```

Inspect one event:

```bash
uv run python scripts/inspect_data.py
```

Run the GPU smoke test:

```bash
sbatch slurm/test_model.slurm
```

Run a tiny tracked training job:

```bash
uv run python scripts/train.py --num-epochs 1 --max-train-batches 1 --max-val-batches 1
```

Generate saved diagnostics:

```bash
uv run python scripts/plot_diagnostics.py --detail-split train[0:1] --representation-split train[:10]
```

Export frozen embeddings from a checkpoint:

```bash
uv run python scripts/export_embeddings.py --checkpoint runs/<run-name>/checkpoint.pt
```

## Tutorial notebook

The teaching notebook for this phase lives in `notebooks/calo_pipeline_tutorial.ipynb`.
It walks through the full pipeline step by step:

1. loading ColliderML calo events
2. building point views
3. creating augmented and masked SSL views
4. running the model
5. understanding the loss
6. saving checkpoints and exporting embeddings

## Development notes

- Keep functions short and obvious.
- Prefer direct code over deep abstractions.
- Keep tests focused on the main behavior only.
- Use `PLAN.md` for the roadmap and `IMPLEMENTATION_GUIDELINES.md` for the detailed technical target.

Format Python files with:

```bash
uv run black .
```

## References

- Panda paper: https://arxiv.org/abs/2512.01324
- ColliderML dataset: https://huggingface.co/datasets/ColliderML/ColliderML
- ColliderML website: https://colliderml.github.io/
