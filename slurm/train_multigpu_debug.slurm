#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -C a100-80gb
#SBATCH -J dbg_train_multigpu
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

source slurm/load_env.sh

PREFIX="${1:-}"
SUFFIX="$(date +%Y%m%d_%H%M%S_%6N)"
RUN_NAME="${PREFIX:+${PREFIX}_}train_multigpu_debug_${SUFFIX}"

nvidia-smi

echo "Using uv: $(command -v uv)"
python --version

python scripts/train.py \
    training.num_gpus=2 \
    training.num_workers=4 \
    training.max_train_batches=20 \
    training.max_val_batches=5 \
    data.local_files_only=true \
    training.num_epochs=5 \
    training.batch_size=4 \
    training.log_backend=comet \
    training.run_name="$RUN_NAME"
