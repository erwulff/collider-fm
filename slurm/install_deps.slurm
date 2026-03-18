#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH -C h100
#SBATCH -J install_deps
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Load modules
module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer uv

# Activate the local virtual environment
source .venv/bin/activate

echo "Using uv: $(command -v uv)"
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-unset}"

# Re-install spconv and torch-scatter on the compute node to match GLIBC 2.28
echo "Installing spconv-cu126..."
uv pip install spconv-cu126 --force-reinstall

echo "Installing torch-scatter from source to match local GLIBC..."
# Using --no-binary to force compilation on the compute node
uv pip install torch-scatter --no-build-isolation --no-binary torch-scatter --force-reinstall
