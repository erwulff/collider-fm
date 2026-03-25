#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-40gb
#SBATCH -J train_small
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

set -euo pipefail

source slurm/load_env.sh

nvidia-smi

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    --num-epochs 4 \
    --batch-size 1 \
    --max-train-batches 8 \
    --max-val-batches 2 \
    --train-split "train[:32]" \
    --val-split "train[32:40]" \
    --log-backend jsonl \
    --run-name "slurm_debug_train_small"
