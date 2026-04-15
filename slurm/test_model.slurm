#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-80gb
#SBATCH -J test_model
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

source slurm/load_env.sh

nvidia-smi
echo "Using uv: $(command -v uv)"
uv run python --version

uv run python scripts/smoke_test_model.py
