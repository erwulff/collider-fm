# Collider Foundation Model

This project aims to implement foundation models for particle collider data in high-energy physics.

## Project Overview

The goal is to learn reusable representations from large-scale simulated HEP data using a self-distillation learning strategy.

## Repository Structure

```text
/
├── AGENTS.md                  # Primary agent-facing repo guide
├── src/
│   └── collider_fm/           # Core library code
│       ├── __init__.py        # Package exports
│       ├── model.py           # Early Panda-style model scaffold
│       └── data.py            # ColliderML dataset and DataLoader code
├── scripts/                   # Standalone Python scripts
│   ├── download_data.py       # Download ColliderML data from Hugging Face
│   └── inspect_data.py        # Data visualization and inspection utilities
├── slurm/                     # SLURM submission scripts for HPC
│   ├── create_uv_venv.slurm   # Create a project virtual environment on a compute node
│   ├── download.slurm         # Job for downloading dataset subsets
│   ├── test_model.slurm       # Job for testing the model on a GPU
│   └── install_deps.slurm     # Job for installing specialized dependencies
├── Panda_repo/                # Git submodule checkout of the Panda reference implementation
├── logs_slurm/                # Tracked placeholder for SLURM stdout/stderr
├── GEMINI.md                  # Legacy project context
├── PLAN.md                    # Detailed development roadmap
├── HPC.md                     # Documentation for HPC resources
└── requirements.txt           # Python dependency list
```

## Current State

This repository is still in an early implementation phase.

- The dataset loader and inspection scripts are present and usable.
- The model code is scaffold-level and uses the `Panda_repo` submodule as a reference dependency.
- Some historical docs describe intended files or workflows that have not been added yet. Prefer [AGENTS.md](AGENTS.md) for current operational guidance.

## Getting Started

### Installation

The project uses `uv` for dependency management. A virtual environment is expected at `.venv`, but it is not committed to the repository.

On this cluster, load the `uv` module before using `uv`:

```bash
module load uv
```

The module sets `UV_CACHE_DIR=$HOME/.cache/uv` and appends its own `uv` binary to `PATH`, so a user-installed `uv` still takes precedence.

To set up the environment manually:
```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

On the cluster, the intended setup flow is:

```bash
sbatch slurm/create_uv_venv.slurm
```

Some dependencies such as `spconv` and `torch-scatter` may need to be installed on compute nodes rather than the login node.
`torch-scatter` is configured in `pyproject.toml` to build from source instead of using a wheel.

### Dataset

The project uses the [ColliderML dataset](https://huggingface.co/datasets/ColliderML/ColliderML). To download a subset or the full dataset, use the provided script:

```bash
uv run python scripts/download_data.py --pu-config pu0 --num-proc 12
```

## Usage

### Training and Evaluation

The core model logic resides in `src/collider_fm/model.py`. The current smoke-test job is:

```bash
sbatch slurm/test_model.slurm
```

Note: this job depends on the `Panda_repo` submodule referenced by the model scaffold.

### Data Inspection

To visualize a 3D event from the dataset:
```bash
uv run python scripts/inspect_data.py
```
This will generate an `event_0_3d.png` file in the root directory.

## References

*   **Panda Paper:** [arXiv:2512.01324](https://arxiv.org/abs/2512.01324)
*   **ColliderML:** [Hugging Face](https://huggingface.co/datasets/ColliderML/ColliderML) | [Website](https://colliderml.github.io/)
