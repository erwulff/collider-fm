#!/bin/bash

if [ ! -d ".venv" ]; then
    echo "Missing .venv. Run slurm/create_uv_venv.slurm first."
    exit 1
fi

module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer uv

source .venv/bin/activate
echo "Using Python: $(which python)"
python --version

echo "Using uv: $(command -v uv)"
echo "uv version: $(uv --version)"
echo "uv Python version: $(uv run python --version)"
