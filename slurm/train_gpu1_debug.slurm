#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100
#SBATCH -J train_sonata_debug
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err


source slurm/load_env.sh

PREFIX="${1:-}"
RUN_NAME="${PREFIX:+${PREFIX}_}sonata_debug"

nvidia-smi

echo "Using uv: $(command -v uv)"
python --version

python scripts/train.py \
    data.local_files_only=true \
    training.num_epochs=4 \
    training.batch_size=4 \
    training.max_train_batches=20 \
    training.max_val_batches=4 \
    training.train_split=train[:200] \
    training.val_split=val[:40] \
    training.log_backend=jsonl \
    training.run_name="$RUN_NAME"
