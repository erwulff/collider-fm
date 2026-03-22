#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C h100
#SBATCH -J train_small
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer uv

source .venv/bin/activate

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    --num-epochs 1 \
    --batch-size 1 \
    --max-train-batches 1 \
    --max-val-batches 1 \
    --train-split "train[:1]" \
    --val-split "train[1:2]"
