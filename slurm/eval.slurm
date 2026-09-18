#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C a100-80gb
#SBATCH -J eval
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Core label-free evaluation: participation ratio, prototype usage, alignment,
# uniformity, NN view-retrieval. No t-SNE, no probes. Usage after training or when
# refreshing metrics against a different checkpoint.
#
# Per-run knobs:
#   CHECKPOINT       Training run dir (runs/<run>) - auto-resolves latest checkpoint;
#                    or path to a specific .pt. Empty = random-init baseline.
#   MAX_EVENTS       Events encoded for the metrics in this eval.
#   TSNE / UPCAST2   Toggle t-SNE / PCA visualizations (full-up-cast and up_cast(2)).

source slurm/load_env.sh

# --- EDIT: what to evaluate ---
CHECKPOINT="runs/train_default_10ep_20260814_123609_922892"
# CHECKPOINT="runs/aug19_scale_step4_20260819_045904_369066"
# CHECKPOINT=""   # random-init baseline
MAX_EVENTS=""    # empty = use eval_config.max_events from default.yaml (4000)
TSNE=false       # true adds t-SNE/PCA plots under eval/<ckpt>/viz/
UPCAST2=false    # true also computes up_cast(2) (pretraining loss space) projections

nvidia-smi
echo "Using Python: $(which python)"

EXTRA_ARGS=()
if [ -n "$MAX_EVENTS" ]; then EXTRA_ARGS+=(evaluation.max_events="$MAX_EVENTS"); fi
if [ "$TSNE" = true ]; then EXTRA_ARGS+=(evaluation.enable_tsne=true evaluation.tsne_max_events=100 evaluation.tsne_max_points=20000); fi
if [ "$UPCAST2" = true ]; then EXTRA_ARGS+=(evaluation.tsne_upcast2=true); fi

python scripts/evaluate.py \
    evaluation.checkpoint="$CHECKPOINT" \
    data.local_files_only=true \
    "${EXTRA_ARGS[@]}"
