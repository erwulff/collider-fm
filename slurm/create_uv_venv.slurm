#!/bin/bash
#SBATCH -t 2:00:00
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
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer

# Local uv installation
UV=~/.local/bin/uv

# Create env with Python 3.12 using uv
$UV venv --python 3.12

# Activate the local virtual environment
source .venv/bin/activate

$UV pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Build torch-scatter from source to ensure compatibility with GLIBC on the compute node
$UV pip install --no-binary torch-scatter --verbose torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html --no-build-isolation

# Install the rest of the dependencies from requirements.txt
$UV pip install -r requirements.txt
 
