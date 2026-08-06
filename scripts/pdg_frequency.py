"""Tally pdg_id frequency across the ColliderML particles config.

Loads the sibling ``<type>_<pu>_particles`` config through
:class:`collider_fm.data.ColliderMLDataset` (correct revision / cache /
``local_files_only`` from the project config, same split resolution as the eval
harness) and reports how often each ``pdg_id`` appears among *all* particle records --
not just calorimeter contributors -- via a single pyarrow ``value_counts`` pass (no
per-event Python decode, so the full 50k-event split tallies in seconds rather than
minutes).

Also collapses the per-pdg counts into the coarse 7-bucket calorimetry roles used by
the t-SNE colormap (``collider_fm.evaluation_labels.pdg_bucket``), so the report shows
how large each color bucket is.

Examples:
  uv run python scripts/pdg_frequency.py                       # held-out val split
  uv run python scripts/pdg_frequency.py --split train[:8]     # tiny smoke check
  HF_HUB_OFFLINE=1 uv run python scripts/pdg_frequency.py data.local_files_only=true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pyarrow.compute as pc

from collider_fm.data import ColliderMLDataset
from collider_fm.evaluation_labels import PDG_BUCKET_NAMES, pdg_bucket
from collider_fm.project_config import build_config_arg_parser, load_project_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_config_arg_parser(
        description="Tally pdg_id frequency across the ColliderML particles config.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/pdg_frequency.py\n"
            "  uv run python scripts/pdg_frequency.py --split train[:8]\n"
            "  HF_HUB_OFFLINE=1 uv run python scripts/pdg_frequency.py data.local_files_only=true"
        ),
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Project split to scan (default: val). Bounded slices like train[:8] are supported.",
    )
    return parser


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    config = load_project_config(cli_args.config, cli_args.overrides)
    dc = config.data
    print(f"revision={dc.dataset_revision}  local_files_only={dc.local_files_only}")
    print(
        f"Loading {dc.dataset_type}_{dc.pu_config}_particles split={cli_args.split} ...",
        flush=True,
    )

    collider_ds = ColliderMLDataset(
        dataset_name=dc.dataset_name,
        dataset_type=dc.dataset_type,
        pu_config=dc.pu_config,
        object_types=["particles"],
        split=cli_args.split,
        cache_dir=dc.cache_dir,
        dataset_revision=dc.dataset_revision,
        local_files_only=dc.local_files_only,
    )
    # The particles config gets no torch formatting, so this is the raw HF Dataset.
    hf = collider_ds.datasets["particles"]
    print(f"num events: {len(hf)}", flush=True)

    # pdg_id is a ragged list<int> column (one inner list per event). Flatten every
    # particle's pdg across all events and value_counts in one vectorized pass.
    col = hf.data.column("pdg_id")  # ChunkedArray (list<int64>)
    flat = col.combine_chunks().values  # flat Int64Array, one entry per particle
    vc = pc.value_counts(flat)  # StructArray: {values, counts}
    pdgs = vc.field("values").to_pylist()
    cnts = vc.field("counts").to_pylist()
    total = int(sum(cnts))

    print(f"total particles: {total}\n", flush=True)
    print(f"{'pdg_id':>12}  {'count':>12}  {'pct':>7}")
    for pdg, c in sorted(zip(pdgs, cnts), key=lambda kv: -kv[1]):
        print(f"{pdg:>12}  {c:>12}  {100.0 * c / total:>6.2f}%")

    # Collapse to the 7 coarse calorimetry-role buckets (the t-SNE colormap). Bucket
    # the distinct pdgs and weight by their counts (not per-distinct-pdg).
    bucket_idx = pdg_bucket(np.asarray(pdgs))
    cnt_arr = np.asarray(cnts, dtype=np.int64)
    bucket_counts = np.zeros(len(PDG_BUCKET_NAMES), dtype=np.int64)
    known = bucket_idx >= 0
    np.add.at(bucket_counts, bucket_idx[known], cnt_arr[known])
    unknown = int(cnt_arr[~known].sum())

    print("\n--- by calorimetry role (t-SNE colormap buckets) ---")
    print(f"{'bucket':>16}  {'count':>12}  {'pct':>7}")
    for idx, name in enumerate(PDG_BUCKET_NAMES):
        c = int(bucket_counts[idx])
        print(f"{name:>16}  {c:>12}  {100.0 * c / total:>6.2f}%")
    if unknown:
        print(f"{'(unknown)':>16}  {unknown:>12}  {100.0 * unknown / total:>6.2f}%")


if __name__ == "__main__":
    main()
