#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH -C h100
#SBATCH -J train_multigpu
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

source slurm/load_env.sh

PREFIX="${1:-}"
SUFFIX="$(date +%Y%m%d_%H%M%S_%6N)"
RUN_NAME="${PREFIX:+${PREFIX}_}train_multigpu_${SUFFIX}"

nvidia-smi

echo "Using uv: $(command -v uv)"
python --version

python scripts/train.py \
    training.num_gpus=8 \
    training.num_workers=4 \
    data.local_files_only=true \
    training.num_epochs=5 \
    training.batch_size=12 \
    training.train_split=train \
    training.val_split=val \
    training.log_backend=comet \
    training.run_name="$RUN_NAME" \
    training.overwrite=true \
    training.log_every_n_steps=50
