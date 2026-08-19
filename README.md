# Collider Foundation Model

ColliderFM is a self-supervised learning project on ColliderML Release 1. The current runtime path is calo-only: build point-cloud views from `calo_hits`, train a Panda-inspired student-teacher model, save checkpoints and metrics, and generate diagnostics and run-level plots.


## Code overview

- `src/collider_fm/data.py` loads ColliderML and applies the project split conventions
- `src/collider_fm/views.py` builds the current point-view and masking pipeline
- `src/collider_fm/model.py` contains the current point-level student-teacher training scaffold
- `scripts/train.py` runs training with checkpointing, JSONL metrics, optional Comet logging, and mixed precision
- `scripts/evaluate.py` runs the label-free pretraining evaluation harness (collapse detection: stable rank, prototype usage, NN view-retrieval, alignment/uniformity) on held-out events
- `scripts/plot_diagnostics.py` and `scripts/plot_training_run.py` cover diagnostics and completed-run plotting
- `notebooks/dataset_walkthrough.ipynb`, `notebooks/sonata_views.ipynb`, and `notebooks/model_walkthrough.ipynb` explain the data and training path

## Repository layout

```text
/
|- src/collider_fm/            # Package code for data, views, model, diagnostics, and vendored PTv3 pieces
|- tests/                      # Unit tests for the current pipeline
|- scripts/                    # Download, inspection, training, evaluation, and plotting scripts
|- notebooks/                  # Newcomer walkthrough notebooks
|- slurm/                      # Cluster job scripts for setup, downloads, and training
|- apptainer/                  # Container helper scripts
|- Panda_repo/                 # Panda reference submodule
|- markdown/                   # Project markdown docs except README.md and AGENTS.md
|- README.md                   # User-facing overview
|- AGENTS.md                   # Short operational guidance for coding agents
|- pyproject.toml              # Project metadata and dependencies
`- uv.lock                     # Locked dependency resolution for uv
```

## Setup

The project targets Python 3.12 and uses `uv` for dependency management.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

For cluster-specific setup and SLURM usage, see `markdown/HPC.md`.

## Documentation

- `markdown/WORKFLOWS.md`: runtime details, config overrides, caching, local commands, outputs, and logging
- `markdown/HPC.md`: cluster setup and checked-in SLURM jobs
- `markdown/PLAN.md`: current roadmap and next priorities
- `AGENTS.md`: short operational instructions for coding agents

## References

- Panda paper: https://arxiv.org/abs/2512.01324
- ColliderML dataset: https://huggingface.co/datasets/ColliderML/ColliderML
- ColliderML website: https://colliderml.github.io/
