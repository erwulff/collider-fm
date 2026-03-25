# Collider Foundation Model

Collider Foundation Model is a small research codebase for self-supervised learning on collider calorimeter data.
The current repository state is a clean, calo-only prototype built around ColliderML Release 1 and a small Panda-style student/teacher model.

## What is implemented now

- `calo_hits` is the default and only input modality in the training path.
- Each event is converted into a sparse 3D point cloud with simple per-point features.
- The model trains with two global teacher views and optional local and masked student views.
- Training saves checkpoints and per-epoch metrics in `runs/<run-name>/`.
- Diagnostics and run-plot scripts produce presentation-ready SSL figures.
- The repository includes a teaching notebook that explains the full pipeline step by step.

The main path through the code is:

`ColliderMLDataset` -> `build_point_view_from_event` -> `build_distillation_views` -> `PandaSelfDistillation` -> checkpoints, diagnostics, and exported embeddings.

## Repository layout

```text
/
├── src/collider_fm/            # Core package code
├── scripts/                    # Download, inspect, train, and plotting utilities
├── tests/                      # Focused regression tests
├── notebooks/                  # Tutorial and diagnostics notebooks
├── slurm/                      # Cluster job scripts
├── Panda_repo/                 # Panda reference submodule
├── PLAN.md                     # Short roadmap
├── IMPLEMENTATION_GUIDELINES.md
├── HPC.md                      # Cluster-specific notes
├── pyproject.toml              # Python dependencies and tool config
└── main.py                     # Tiny entrypoint placeholder
```

## Current data and model contract

This phase uses only ColliderML `calo_hits`.

Each point keeps:

- coordinates: `x`, `y`, `z`
- deposited energy
- a simple ECal or HCal flag

The point-view representation in `src/collider_fm/views.py` also keeps two simple boolean masks:

- `hidden_mask`: points whose input energy is hidden from the student
- `loss_mask`: points that contribute to the student loss

That makes the current SSL setup easy to follow:

- global views: nothing hidden, all points used in the loss
- local views: points outside one neighborhood are hidden and ignored in the loss
- masked views: randomly hidden points are used in the loss

## Environment setup

The project uses `uv` and targets Python 3.12.

```bash
module load uv
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

Some packages such as `spconv` and `torch-scatter` may need to be installed on compute nodes rather than the login node.
See `HPC.md` for the recommended SLURM flow.

## Dataset

The project uses ColliderML Release 1:

- Hugging Face: https://huggingface.co/datasets/ColliderML/ColliderML
- Project site: https://colliderml.github.io/

The documented cache location on this cluster is `/mnt/ceph/users/ewulff/data/hf`.

## Common workflows

Download the calo-hit tables:

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

Run a tiny local training sanity check on a GPU node:

```bash
uv run python scripts/train.py --num-epochs 1 --max-train-batches 1 --max-val-batches 1 --add-local-view --add-masked-view
```

Run the current SSL baseline through SLURM:

```bash
sbatch slurm/train_small.slurm
```

Plot a finished training run:

```bash
uv run python scripts/plot_training_run.py runs/<run-name>
```

Generate richer checkpoint diagnostics:

```bash
uv run python scripts/plot_diagnostics.py --checkpoint runs/<run-name>/checkpoint.pt --output-dir diagnostics/<name>
```

Export frozen embeddings from a checkpoint:

```bash
uv run python scripts/export_embeddings.py --checkpoint runs/<run-name>/checkpoint.pt
```

## Where outputs go

- `runs/<run-name>/`
  - `config.json`
  - `metrics.jsonl`
  - `checkpoint.pt`
  - `plots/` from `scripts/plot_training_run.py`
- `diagnostics/<name>/`
  - saved plots and summaries from `scripts/plot_diagnostics.py`

Generated outputs are ignored by git.

## Teaching notebook

The teaching notebook for the current pipeline is `notebooks/calo_pipeline_tutorial.ipynb`.
It explains:

1. loading raw ColliderML calo events
2. building point views
3. creating global, local, and masked views
4. running the model
5. computing the loss
6. saving checkpoints and exporting embeddings

`notebooks/plot_diagnostics_explorer.ipynb` is a second notebook for interactive diagnostics work.

## Development notes

- Keep functions short and direct.
- Prefer explicit data flow over deep abstractions.
- Keep tests focused on core behavior.
- Prefer SLURM for heavy downloads or GPU runs.
- Keep markdown files aligned with the real repository state.

Generated output folders such as `runs/`, `diagnostics/`, and one-off event images are local artifacts and should stay out of git.

Format Python files with:

```bash
uv run black .
```

Run the test suite with:

```bash
uv run python -m unittest discover -s tests
```

## References

- Panda paper: https://arxiv.org/abs/2512.01324
- ColliderML dataset: https://huggingface.co/datasets/ColliderML/ColliderML
- ColliderML website: https://colliderml.github.io/
