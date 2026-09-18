#!/bin/bash
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH -C h100
#SBATCH -J train
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Full-scale Panda-paper-config training (91M backbone params, patch 256), 8xH100.
# batch_size=8 per GPU (global batch 64). Long runs: extends the 72h default QOS
# to 7 days; for >7 days, resume from the latest checkpoint of the same run_name.
#
# Per-run knobs:
#   RUN_NAME_PREFIX  (optional string prefix for RUN_NAME, passed as $1)
#
# Edit the training-shape params below per run. For short smoke runs use
# slurm/train_smoke.slurm.
#
# Auto-eval runs on the best checkpoint when training finishes
# (training.eval_after_training=true by default).

source slurm/load_env.sh

# --- EDIT: run identity ---
RUN_NAME_PREFIX="${1:-}"
SUFFIX="$(date +%Y%m%d_%H%M%S_%6N)"
RUN_NAME="${RUN_NAME_PREFIX:+${RUN_NAME_PREFIX}_}train_${SUFFIX}"

# --- EDIT: training schedule ---
NUM_EPOCHS=30
BATCH_SIZE=8
NUM_WORKERS=4
LOG_EVERY_N_STEPS=50

nvidia-smi
echo "Using uv: $(command -v uv)"
python --version

python scripts/train.py \
    training.num_gpus=8 \
    training.num_workers="$NUM_WORKERS" \
    training.num_epochs="$NUM_EPOCHS" \
    training.batch_size="$BATCH_SIZE" \
    training.train_split=train \
    training.val_split=val \
    training.log_backend=comet \
    training.run_name="$RUN_NAME" \
    training.log_every_n_steps="$LOG_EVERY_N_STEPS" \
    data.local_files_only=true \
    model.training.backbone.enc_channels="[48,96,192,384,512]" \
    model.training.backbone.enc_num_head="[3,6,12,24,32]" \
    model.training.backbone.enc_depths="[3,3,3,9,3]" \
    model.training.backbone.enc_patch_size="[256,256,256,256,256]"
