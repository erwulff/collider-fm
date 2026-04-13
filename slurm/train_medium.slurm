#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100
#SBATCH -J train_medium
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err


source slurm/load_env.sh

nvidia-smi

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    training.num_epochs=12 \
    training.batch_size=1 \
    training.max_train_batches=24 \
    training.max_val_batches=6 \
    training.train_split=train[:96] \
    training.val_split=train[96:120] \
    training.log_backend=jsonl \
    training.run_name=slurm_medium_train
