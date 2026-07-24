#!/bin/bash

module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer uv

if [ ! -f .venv/bin/activate ]; then
    echo "Expected .venv/bin/activate to exist. Run slurm/create_uv_venv.slurm first." >&2
    exit 1
fi

source .venv/bin/activate
export UV_NO_PYTHON_DOWNLOADS=1

# The dataset is staged in the HF cache on Ceph (data.local_files_only=true);
# keep workers from reaching out to huggingface.co at startup.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Using Python: $(which python)"
python --version

echo "Using uv: $(command -v uv)"
echo "uv version: $(uv --version)"
