# Panda on ColliderML

This project implements the Panda self-distillation methodology on the ColliderML dataset for high-energy physics.

## Project Overview

The goal is to learn reusable, sensor-level representations from large-scale simulated HEP data using a hierarchical sparse 3D encoder (Point Transformer V3) and a self-distillation learning strategy as described in the Panda paper.

## Repository Structure

```text
/
├── src/
│   └── collider_fm/           # Core library code
│       ├── __init__.py        # Package exports
│       ├── model.py           # Panda model implementation (PTv3 backbone)
│       └── data.py            # ColliderML PyTorch Dataset and DataLoader
├── scripts/                   # Standalone Python scripts
│   ├── download_data.py       # Script to download ColliderML from Hugging Face
│   └── inspect_data.py        # Data visualization and inspection utilities
├── slurm/                     # SLURM submission scripts for HPC
│   ├── download.slurm         # Job for downloading the full dataset
│   ├── test_model.slurm       # Job for testing the model on a GPU
│   └── install_deps.slurm     # Job for installing specialized dependencies
├── Panda_repo/                # Submodule/Clone of the original Panda repository
├── GEMINI.md                  # Project mandates and context
├── PLAN.md                    # Detailed development roadmap
├── HPC.md                     # Documentation for HPC resources
└── WORKING_NOTES.md           # Log of current work and progress
```

## Getting Started

### Installation

The project uses `uv` for dependency management. A virtual environment is located in `.venv`.

To set up the environment (if not already done):
```bash
# Using the uv path from GEMINI.md
~/.local/bin/uv venv
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt
```

Note: Some dependencies like `spconv` and `torch-scatter` might require specialized installation on compute nodes via `slurm/install_deps.slurm`.

### Dataset

The project uses the [ColliderML dataset](https://huggingface.co/datasets/ColliderML/ColliderML). To download a subset or the full dataset, use the provided script:

```bash
python scripts/download_data.py --pu-config pu0 --num-proc 12
```

## Usage

### Training and Evaluation

The core model logic resides in `src/collider_fm/model.py`. You can run a basic test of the model initialization and forward pass using SLURM:

```bash
sbatch slurm/test_model.slurm
```

### Data Inspection

To visualize a 3D event from the dataset:
```bash
python scripts/inspect_data.py
```
This will generate an `event_0_3d.png` file in the root directory.

## HPC Resources

This project is configured to run on HPC clusters using SLURM. Compute-heavy tasks (training, large-scale data processing) should always be submitted via `sbatch`. Logs are stored in `logs_slurm/`.

## References

*   **Panda Paper:** [arXiv:2512.01324](https://arxiv.org/abs/2512.01324)
*   **ColliderML:** [Hugging Face](https://huggingface.co/datasets/ColliderML/ColliderML) | [Website](https://colliderml.github.io/)
