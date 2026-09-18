#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-80gb
#SBATCH -J eval_tsne
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# t-SNE + PCA plots of frozen features, colored by particle type and event id
# (Panda-style). Full-up-cast by default; enable up_cast(2) (pretraining space) for
# the second row of plots when you want the bigger picture.
#
# Per-run knobs:
#   CHECKPOINT       Training run dir (runs/<run>) - auto-resolves latest checkpoint.
#   TSNE_MAX_EVENTS  Events to include in the t-SNE; ~100 is a nice visual density.
#   TSNE_MAX_POINTS  Per-event point budget; ~20000 is a reasonable cap.

source slurm/load_env.sh

# --- EDIT: what to plot ---
CHECKPOINT="runs/train_default_10ep_20260814_123609_922892"
TSNE_MAX_EVENTS=100
TSNE_MAX_POINTS=20000

nvidia-smi
echo "Using Python: $(which python)"

python scripts/evaluate.py \
    evaluation.checkpoint="$CHECKPOINT" \
    evaluation.enable_tsne=true \
    evaluation.tsne_upcast2=true \
    evaluation.tsne_max_events="$TSNE_MAX_EVENTS" \
    evaluation.tsne_max_points="$TSNE_MAX_POINTS" \
    data.local_files_only=true
