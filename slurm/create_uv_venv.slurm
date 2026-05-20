#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH -C h100
#SBATCH -J create_uv_venv
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err


# Load modules
module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer uv

# Create env with Python 3.12 using uv
uv venv --python 3.12

# Activate the local virtual environment
source .venv/bin/activate
export UV_NO_PYTHON_DOWNLOADS=1

echo "Using uv: $(command -v uv)"
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-unset}"

# Install the project dependencies from pyproject.toml.
# torch and torchvision resolve from the CUDA 12.8 PyTorch index, and
# torch-scatter is built from source per the uv settings in pyproject.toml.
uv sync --locked
 
