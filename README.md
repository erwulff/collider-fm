# Collider Foundation Model

Collider Foundation Model is an early-stage Python project for learning reusable representations from particle collider data. The current focus is adapting Panda-style self-distillation workflows to the ColliderML dataset.

## Project goal

The project explores self-distillation methods for high-energy physics data, with Panda-style sensor-level representation learning as the starting point. The near-term goal is to get a practical training and evaluation workflow running on ColliderML before scaling to larger experiments.

## Current status

- `src/collider_fm/data.py` provides dataset loading and basic collation.
- `src/collider_fm/model.py` contains the current model scaffold and smoke-test path.
- `scripts/download_data.py` and `scripts/inspect_data.py` cover basic data acquisition and inspection.
- `Panda_repo/` is kept as a reference implementation, while the runtime code vendors the PTv3 pieces currently needed under `src/collider_fm/_panda/`.

## Repository layout

```text
/
├── src/collider_fm/            # Python package code
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts for downloads, inspection, and smoke tests
├── slurm/                      # Batch jobs for cluster workflows
├── apptainer/                  # Container helper scripts
├── Panda_repo/                 # Panda reference submodule
├── logs_slurm/                 # Placeholder directory for SLURM logs
├── pyproject.toml              # Project metadata and tool configuration
├── uv.lock                     # Locked dependency resolution for uv
├── PLAN.md                     # Roadmap and task tracking
└── HPC.md                      # Cluster and SLURM notes
```

## Environment setup

The project uses `uv` for dependency management and targets Python 3.12.

On this cluster, load the `uv` module first:

```bash
module load uv
```

Then create and activate a virtual environment and sync dependencies:

```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

Some dependencies such as `spconv` and `torch-scatter` may need to be installed on compute nodes rather than the login node. `torch-scatter` is configured in `pyproject.toml` to build from source.

If you prefer to create the environment through SLURM, use:

```bash
sbatch slurm/create_uv_venv.slurm
```

## Dataset

The project uses the ColliderML dataset:

- Hugging Face: https://huggingface.co/datasets/ColliderML/ColliderML
- Project site: https://colliderml.github.io/

ColliderML configurations combine a physics process such as `ttbar`, `ggf`, or `dihiggs`, a pileup setting such as `pu0` or `pu200`, and an object type such as `particles`, `tracker_hits`, `calo_hits`, or `tracks`.

On this cluster, the documented Hugging Face cache location is `/mnt/ceph/users/ewulff/data/hf`.

## Common workflows

Download a ColliderML subset:

```bash
uv run python scripts/download_data.py --pu-config pu0 --num-proc 12
```

Inspect a sample event:

```bash
uv run python scripts/inspect_data.py
```

Run the model smoke test on a GPU allocation:

```bash
sbatch slurm/test_model.slurm
```

The smoke-test entrypoint lives in `scripts/smoke_test_model.py`.

## Development notes

Python formatting is handled with Black and configured in `pyproject.toml` with a line length of 160 characters.

```bash
uv run black .
```

For roadmap and cluster-specific guidance:

- `PLAN.md` tracks milestones and open work.
- `HPC.md` covers cluster usage, SLURM notes, and environment details.

## References

- Panda paper: https://arxiv.org/abs/2512.01324
- ColliderML dataset: https://huggingface.co/datasets/ColliderML/ColliderML
- ColliderML website: https://colliderml.github.io/
