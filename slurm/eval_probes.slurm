#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C h100
#SBATCH -J eval_probes
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Panda-style linear-probe evaluation (frozen backbone features + random-init and
# raw-input baselines). Writes artifacts to runs/<run>/eval/<checkpoint>/probes/.
#
# Reuses cached backbone features from runs/<run>/eval/backbone_features/<ckpt>/ when
# they exist, so a second eval of the same checkpoint skips the ~40 min collection
# and finishes in ~5 min.
#
# Per-run knobs:
#   CHECKPOINT       Training run dir (runs/<run>) - resolves latest checkpoint; or
#                    path to a specific .pt file. Empty string = random-init eval.
#   VAL_SPLIT        Eval-data slice (e.g. 'val[:1000]'); smaller = quicker smoke.
#   MAX_EVENTS       Number of events to encode for the main metrics in this eval.
#
# For a quick direction-check on a fresh run: use VAL_SPLIT='val[:400]' and small
# probe splits (HANDLED inside scripts/evaluate.py via probe_train_events/val_events).
# To run the full-size probe (2000 train / 500 val events), leave the defaults.
#
# Auto-eval runs automatically after training; you'll rarely need to submit this by
# hand unless adding a new baseline or re-running with new probe settings.

source slurm/load_env.sh

# --- EDIT: what to evaluate ---
CHECKPOINT="runs/aug19_scale_step4_20260819_045904_369066"
# CHECKPOINT="runs/train_default_10ep_20260814_123609_922892"
# CHECKPOINT=""   # random-init baseline

nvidia-smi
echo "Using Python: $(which python)"

python scripts/evaluate.py \
    evaluation.checkpoint="$CHECKPOINT" \
    evaluation.enable_probes=true \
    evaluation.probe_random_init_baseline=true \
    evaluation.probe_raw_input_baseline=true \
    data.local_files_only=true
