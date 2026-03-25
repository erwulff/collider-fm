from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.data import ColliderMLDataset
from collider_fm.views import CALO_TYPE_NAMES, build_point_view_from_event


def plot_event_3d(event: dict[str, dict[str, object]], event_index: int = 0) -> None:
    calo_hits = event["calo_hits"]
    x = torch.as_tensor(calo_hits["x"])
    y = torch.as_tensor(calo_hits["y"])
    z = torch.as_tensor(calo_hits["z"])
    energy = torch.as_tensor(calo_hits["energy"])
    calo_type = build_point_view_from_event(event, device=torch.device("cpu"))[
        "calo_type"
    ]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = getattr(ax, "scatter")(
        np.asarray(z.tolist(), dtype=float),
        np.asarray(x.tolist(), dtype=float),
        np.asarray(y.tolist(), dtype=float),
        c=np.asarray(energy.tolist(), dtype=float),
        s=4,
        alpha=0.6,
        cmap="inferno",
    )
    fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1, label="energy")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title(f"ColliderML calorimeter event {event_index}")
    plt.savefig(f"event_{event_index}_calo.png")

    counts = {
        name: int((calo_type == value).sum().item())
        for value, name in CALO_TYPE_NAMES.items()
    }
    print(f"Saved event visualization to event_{event_index}_calo.png")
    print(f"ECal hits: {counts['ecal']}")
    print(f"HCal hits: {counts['hcal']}")


def main() -> None:
    dataset = ColliderMLDataset(split="train[:5]", object_types=["calo_hits"])
    print("Loading event 0 for calorimeter inspection...")
    sample = dataset[0]
    plot_event_3d(sample, 0)

    calo_hits = sample["calo_hits"]
    print("\nBasic stats for event 0:")
    print(f"Number of calo hits: {len(torch.as_tensor(calo_hits['x']))}")
    print(
        f"Energy range: [{torch.as_tensor(calo_hits['energy']).min():.4f}, {torch.as_tensor(calo_hits['energy']).max():.4f}]"
    )
    print(
        f"x range: [{torch.as_tensor(calo_hits['x']).min():.2f}, {torch.as_tensor(calo_hits['x']).max():.2f}]"
    )
    print(
        f"y range: [{torch.as_tensor(calo_hits['y']).min():.2f}, {torch.as_tensor(calo_hits['y']).max():.2f}]"
    )
    print(
        f"z range: [{torch.as_tensor(calo_hits['z']).min():.2f}, {torch.as_tensor(calo_hits['z']).max():.2f}]"
    )


if __name__ == "__main__":
    main()
