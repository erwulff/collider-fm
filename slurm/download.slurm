#!/bin/bash

# Walltime limit
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G

# Job name
#SBATCH -J download_pu0

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Load modules if necessary
module load slurm uv

# Activate the local virtual environment
source .venv/bin/activate

# Run the Python script to download the pu0 datasets
uv run python scripts/download_data.py \
    download.dataset_types=[ttbar] \
    download.object_types=[calo_hits,tracker_hits,particles] \
    download.pu_config=pu0 \
    download.cache_dir=/mnt/ceph/users/ewulff/data/hf \
    download.num_proc=16
