#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-80gb
#SBATCH -J train_smoke
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Smoke / CI-style training: 1 epoch, tiny run, verifies end-to-end training wiring
# (datasets load, model builds, step + checkpoint + eval complete, auto-eval runs).
# Usage when changing training code, not for real experiments.

source slurm/load_env.sh

nvidia-smi
echo "Using uv: $(command -v uv)"
python --version

python scripts/train.py \
    training.num_gpus=1 \
    training.num_workers=2 \
    training.num_epochs=1 \
    training.batch_size=4 \
    training.train_split=train[:32] \
    training.val_split=val[:8] \
    training.log_backend=jsonl \
    training.run_name=train_smoke \
    training.log_every_n_steps=1 \
    data.local_files_only=true \
    training.eval_after_training=false
