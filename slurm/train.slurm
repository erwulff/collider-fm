#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100
#SBATCH -J train
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

set -euo pipefail

source slurm/load_env.sh

nvidia-smi

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    training.num_epochs=10 \
    training.batch_size=16 \
    training.train_split=train \
    training.val_split=train[0:1000] \
    data.dataset_revision=e28a24cc9c1641a478ae4e5bc3b376eb624b7283 \
    data.local_files_only=true \
    training.num_workers=8 \
    training.prefetch_factor=2 \
    training.pin_memory=true \
    training.log_backend=comet \
    training.run_name=bs16-epoch10
