from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset


def plot_event_3d(event: dict[str, object], event_idx: int = 0) -> None:
    """Save a simple 3D scatter plot for one raw calorimeter event."""

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    calo_hits = event["calo_hits"]
    energy = calo_hits["energy"]
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
    dataset = ColliderMLDataset(split="train[:5]", object_types=["calo_hits"])
    print("Loading calorimeter event 0 for visualization...")
    sample = dataset[0]
    plot_event_3d(sample, 0)

    print("\nBasic Stats for Event 0:")
    print(f"Number of calo hits: {len(sample['calo_hits']['x'])}")

    calo_hits = sample["calo_hits"]
    print(f"Calo x range: [{calo_hits['x'].min():.2f}, {calo_hits['x'].max():.2f}]")
    print(f"Calo y range: [{calo_hits['y'].min():.2f}, {calo_hits['y'].max():.2f}]")
    print(f"Calo z range: [{calo_hits['z'].min():.2f}, {calo_hits['z'].max():.2f}]")
    print(
        f"Calo energy range: [{calo_hits['energy'].min():.2f}, {calo_hits['energy'].max():.2f}]"
    )
