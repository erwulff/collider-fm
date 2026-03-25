#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C h100
#SBATCH -J train_ssl_baseline
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer uv

source .venv/bin/activate

echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/train.py \
    --num-epochs 20 \
    --batch-size 1 \
    --max-train-batches 100 \
    --max-val-batches 20 \
    --max-calo-hits 256 \
    --log-every 10 \
    --add-local-view \
    --local-fraction 0.5 \
    --add-masked-view \
    --mask-fraction 0.3 \
    --train-split "train[:200]" \
    --val-split "train[200:240]" \
    --log-backend jsonl \
    --run-name monday_ssl_baseline
