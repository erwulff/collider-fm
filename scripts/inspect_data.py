import matplotlib.pyplot as plt
import sys
import os

# Add src to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from collider_fm.data import ColliderMLDataset


def plot_event_3d(event, event_idx=0):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot tracker hits
    th = event["tracker_hits"]
    ax.scatter(th["z"], th["x"], th["y"], s=1, alpha=0.5, label="Tracker Hits", c="blue")

    # Plot calo hits
    ch = event["calo_hits"]
    # Filter out zero energy hits if any
    mask = ch["total_energy"] > 0
    ax.scatter(ch["z"][mask], ch["x"][mask], ch["y"][mask], s=ch["total_energy"][mask] * 10, alpha=0.3, label="Calo Hits", c="red")

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title(f"ColliderML Event {event_idx} - ttbar pu0")
    ax.legend()

    plt.savefig(f"event_{event_idx}_3d.png")
    print(f"Saved event visualization to event_{event_idx}_3d.png")


if __name__ == "__main__":
    dataset = ColliderMLDataset(split="train[:5]")
    print("Loading event 0 for visualization...")
    sample = dataset[0]
    plot_event_3d(sample, 0)

    print("\nBasic Stats for Event 0:")
    print(f"Number of tracker hits: {len(sample['tracker_hits']['x'])}")
    print(f"Number of calo hits:    {len(sample['calo_hits']['x'])}")
    print(f"Number of particles:    {len(sample['particles']['px'])}")

    # Print some coordinate ranges to verify
    th = sample["tracker_hits"]
    print(f"Tracker x range: [{th['x'].min():.2f}, {th['x'].max():.2f}]")
    print(f"Tracker y range: [{th['y'].min():.2f}, {th['y'].max():.2f}]")
    print(f"Tracker z range: [{th['z'].min():.2f}, {th['z'].max():.2f}]")
