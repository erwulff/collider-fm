#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One task per config; %4 keeps at most 4 concurrent downloads
#SBATCH -a 0-3%4

# Job name
#SBATCH -J download_colliderml

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%A_%a.out
#SBATCH -e logs_slurm/log_%x_%A_%a.err

module load slurm uv

# Activate the local virtual environment
source .venv/bin/activate

# Load tokens (e.g. HF_TOKEN) from the repo .env — without a token the Hub
# throttles to unauthenticated rate limits, which matters at this volume.
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env not found in $(pwd)" >&2
    exit 1
fi

# Download needs network access; force online regardless of the submitting
# shell's environment (sbatch --export=ALL can carry over HF_HUB_OFFLINE=1
# from a prior load_env.sh session).
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# Keep both the payload and the hf_xet chunk cache off the home filesystem.
export COLLIDERML_DATA_DIR=/mnt/ceph/users/ewulff/data/colliderml
export HF_HOME=/mnt/ceph/users/ewulff/huggingface

CONFIGS=(ttbar_pu0_calo_hits ttbar_pu0_particles ttbar_pu0_tracker_hits ttbar_pu0_tracks)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
REVISION=64c3d2f112df3d5d20979d22da7cfdff13e10c4b

echo "Downloading $CONFIG @ $REVISION -> $COLLIDERML_DATA_DIR"

# No --max-events: full config via snapshot_download, which is parallel and
# resumable. Re-submitting the same task skips files already complete.
colliderml download \
    --config "$CONFIG" \
    --revision "$REVISION" \
    --out "$COLLIDERML_DATA_DIR"

status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: colliderml download failed for $CONFIG (exit $status)" >&2
    exit $status
fi

# Sanity check: every Release-1 config is 1000 shards of 1000 events.
SHARD_DIR="$COLLIDERML_DATA_DIR/CERN__ColliderML-Release-1/$CONFIG/data/$CONFIG"
N_SHARDS=$(ls "$SHARD_DIR"/train-*.parquet 2>/dev/null | wc -l)
echo "$CONFIG: $N_SHARDS/1000 shards, $(du -sh "$SHARD_DIR" | cut -f1) on disk"
if [ "$N_SHARDS" -ne 1000 ]; then
    echo "ERROR: expected 1000 shards for $CONFIG, found $N_SHARDS" >&2
    exit 1
fi

python -c "
import pyarrow.parquet as pq
f = pq.ParquetFile('$SHARD_DIR/train-00000-of-01000.parquet')
print('shard 0:', f.metadata.num_rows, 'events,', len(f.schema_arrow), 'columns')
print('columns:', f.schema_arrow.names)
"
