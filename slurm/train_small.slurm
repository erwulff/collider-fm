#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100
#SBATCH -J train_small
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err



source slurm/load_env.sh

nvidia-smi

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    training.num_epochs=4 \
    training.batch_size=1 \
    training.max_train_batches=8 \
    training.max_val_batches=2 \
    training.train_split=train[:32] \
    training.val_split=train[32:40] \
    training.log_backend=jsonl \
    training.run_name=slurm_debug_train_small
