#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-40gb
#SBATCH -J train_medium
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

set -euo pipefail

source slurm/load_env.sh

nvidia-smi

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    --num-epochs 12 \
    --batch-size 1 \
    --max-train-batches 24 \
    --max-val-batches 6 \
    --train-split "train[:96]" \
    --val-split "train[96:120]" \
    --log-backend jsonl \
    --run-name "slurm_medium_train"
