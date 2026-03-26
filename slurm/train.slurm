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
    --num-epochs 10 \
    --batch-size 16 \
    --train-split "train" \
    --val-split "train[0:1000]" \
    --dataset-revision e28a24cc9c1641a478ae4e5bc3b376eb624b7283 \
    --local-files-only \
    --num-workers 8 \
    --prefetch-factor 2 \
    --pin-memory \
    --log-backend comet \
    --run-name "bs16-epoch10"
