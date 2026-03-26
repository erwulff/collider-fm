#!/bin/bash

# Walltime limit
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=12
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
    --dataset-types ttbar \
    --object-types calo_hits tracker_hits particles \
    --pu-config pu0 \
    --cache-dir /mnt/ceph/users/ewulff/data/hf \
    --dataset-revision e28a24cc9c1641a478ae4e5bc3b376eb624b7283 \
    --num-proc 12
