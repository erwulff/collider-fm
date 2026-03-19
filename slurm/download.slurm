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
module --force purge; module load modules/2.2-20230808
module load slurm gcc cmake cuda/12.1.1 cudnn/8.9.2.26-12.x nccl openmpi apptainer uv

# Activate the local virtual environment
source .venv/bin/activate

# Run the Python script to download the pu0 datasets
uv run python scripts/download_data.py \
    --dataset-types ttbar \
    --object-types particles tracker_hits calo_hits tracks \
    --pu-config pu0 \
    --cache-dir /mnt/ceph/users/ewulff/data/hf \
    --num-proc 12
