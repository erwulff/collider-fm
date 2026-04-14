#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-80gb
#SBATCH -J sonata_full
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

source slurm/load_env.sh

nvidia-smi

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    model.recipe=sonata \
    data.local_files_only=true \
    training.num_epochs=5 \
    training.batch_size=8 \
    training.train_split=train \
    training.val_split=val \
    training.log_backend=comet \
    training.run_name=sonata_full
