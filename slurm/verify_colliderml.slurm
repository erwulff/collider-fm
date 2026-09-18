#!/bin/bash

# Verify the downloaded ColliderML Parquet loads through the package's own
# loader, and time it — this is what tells us whether the colliderml path is
# usable for training, not just that the bytes arrived.
# Run after slurm/download_colliderml.slurm completes.

# Walltime limit
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Job name
#SBATCH -J verify_colliderml

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

module load slurm uv

source .venv/bin/activate

# Reading only — no network needed.
export HF_HUB_OFFLINE=1
export COLLIDERML_DATA_DIR=/mnt/ceph/users/ewulff/data/colliderml
export HF_HOME=/mnt/ceph/users/ewulff/huggingface

python - <<'PY'
import os
import time

from colliderml.core.loader import load_tables
from colliderml.core.tables import collect_tables

DATA_DIR = os.environ["COLLIDERML_DATA_DIR"]

# lazy=True is mandatory here: lazy=False calls pl.read_parquet on every shard,
# which would try to materialise the full 840 GB calo_hits config.
for objects in (["calo_hits"], ["calo_hits", "particles"], ["tracker_hits", "tracks"]):
    start = time.perf_counter()
    tables = load_tables(
        {
            "channels": "ttbar",
            "pileup": "pu0",
            "objects": objects,
            "split": "train",
            "lazy": True,
            "max_events": 5,
            "data_dir": DATA_DIR,
        }
    )
    frames = collect_tables(tables)
    elapsed = time.perf_counter() - start
    print(f"{'+'.join(objects)}: {elapsed:.1f}s")
    for name, df in frames.items():
        print(f"  {name:14s} {df.shape} event_ids={df['event_id'].to_list()}")
        print(f"  {'':14s} columns={df.columns}")
PY
