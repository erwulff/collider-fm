from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
)


def plot_event_3d(event: Mapping[str, Any], event_idx: int = 0) -> None:
    """Save a simple 3D scatter plot for one raw calorimeter event.

    Args:
        event (Mapping[str, Any]): Raw ColliderML event dict with a `calo_hits`
            sub-mapping.
        event_idx (int, optional): Event index for the output filename and
            title. Defaults to 0.
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    calo_hits = cast(Mapping[str, Any], event["calo_hits"])
    energy = calo_hits["total_energy"]
    mask = energy > 0
    ax.scatter(
        calo_hits["z"][mask],
        calo_hits["x"][mask],
        calo_hits["y"][mask],
        s=energy[mask] * 10,
        alpha=0.35,
        label="Calo Hits",
        c=energy[mask],
        cmap="inferno",
    )

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title(f"ColliderML calorimeter event {event_idx} - ttbar pu0")
    ax.legend()

    output_path = PROJECT_ROOT / f"event_{event_idx}_3d.png"
    plt.savefig(output_path)
    print(f"Saved event visualization to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = build_config_arg_parser(
        description="Inspect one cached ColliderML calorimeter event.",
        epilog="Example:\n  uv run python scripts/inspect_data.py data.local_files_only=true",
    )
    cli_args = parser.parse_args()
    project_config = load_project_config(cli_args.config, cli_args.overrides)
    data_config = project_config.data
    dataset = ColliderMLDataset(
        dataset_name=data_config.dataset_name,
        dataset_type=data_config.dataset_type,
        pu_config=data_config.pu_config,
        cache_dir=data_config.cache_dir,
        dataset_revision=data_config.dataset_revision,
        split="train[:5]",
        object_types=["calo_hits"],
        local_files_only=data_config.local_files_only,
    )
    print("Loading calorimeter event 0 for visualization...")
    sample = dataset[0]
    plot_event_3d(sample, 0)

    print("\nBasic Stats for Event 0:")
    print(f"Number of calo hits: {len(sample['calo_hits']['x'])}")

    calo_hits = sample["calo_hits"]
    print(f"Calo x range: [{calo_hits['x'].min():.2f}, {calo_hits['x'].max():.2f}]")
    print(f"Calo y range: [{calo_hits['y'].min():.2f}, {calo_hits['y'].max():.2f}]")
    print(f"Calo z range: [{calo_hits['z'].min():.2f}, {calo_hits['z'].max():.2f}]")
    print(f"Calo energy range: [{calo_hits['total_energy'].min():.2f}, {calo_hits['total_energy'].max():.2f}]")
