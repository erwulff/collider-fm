#!/bin/bash

# Walltime limit
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G

# Job name
#SBATCH -J download_pu0

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Load modules if necessary
module load slurm uv

# Activate the local virtual environment
source .venv/bin/activate

# Load tokens (e.g. HF_TOKEN) from the repo .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env not found in $(pwd)" >&2
    exit 1
fi

# Download needs network access; force online regardless of the
# submitting shell's environment (sbatch --export=ALL can carry over
# HF_HUB_OFFLINE=1 from a prior load_env.sh session).
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# Run the Python script to download the pu0 datasets
python scripts/download_data.py \
    download.dataset_types=[ttbar] \
    download.object_types=[calo_hits,tracker_hits,particles,tracks] \
    download.pu_config=pu0 \
    download.cache_dir=/mnt/ceph/users/ewulff/data/hf \
    download.num_proc=16
