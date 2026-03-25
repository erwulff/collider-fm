from collections.abc import Sequence
from typing import Any, TypedDict

import torch
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader, Dataset


DEFAULT_OBJECT_TYPES = ("tracker_hits", "calo_hits")


class RawTrackerHits(TypedDict, total=False):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    time: torch.Tensor
    detector: torch.Tensor
    volume_id: torch.Tensor
    layer_id: torch.Tensor
    surface_id: torch.Tensor
    true_x: torch.Tensor
    true_y: torch.Tensor
    true_z: torch.Tensor
    particle_id: torch.Tensor


class RawCaloHits(TypedDict, total=False):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    total_energy: torch.Tensor
    detector: torch.Tensor
    contrib_particle_ids: list[list[int]]
    contrib_energies: list[list[float]]
    contrib_times: list[list[float]]


class RawColliderEvent(TypedDict):
    tracker_hits: RawTrackerHits
    calo_hits: RawCaloHits


def _convert_list_value(value: list[Any]) -> Any:
    if len(value) > 0 and isinstance(value[0], list):
        return value

    array = np.asarray(value)
    if array.dtype == object:
        return value
    return torch.tensor(array)


class ColliderMLDataset(Dataset):
    """
    A PyTorch Dataset for loading ColliderML data from Hugging Face datasets.
    It combines multiple configurations (e.g., particles, tracker_hits, calo_hits)
    for the same events.
    """

    def __init__(
        self,
        dataset_name="CERN/ColliderML-Release-1",
        dataset_type="ttbar",
        pu_config="pu0",
        object_types: Sequence[str] = DEFAULT_OBJECT_TYPES,
        split="train",
        cache_dir="/mnt/ceph/users/ewulff/data/hf",
    ):

        self.dataset_name = dataset_name
        self.object_types = tuple(object_types)
        if not self.object_types:
            raise ValueError("'object_types' must contain at least one dataset configuration.")
        self.datasets = {}

        for obj_type in self.object_types:
            config_name = f"{dataset_type}_{pu_config}_{obj_type}"
            print(f"Loading {config_name}...")
            self.datasets[obj_type] = load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)

        # All datasets should have the same number of rows (events)
        self.num_events = len(self.datasets[self.object_types[0]])
        for obj_type in self.object_types:
            dataset_length = len(self.datasets[obj_type])
            if dataset_length != self.num_events:
                raise ValueError(
                    f"Dataset {obj_type} has {dataset_length} rows, expected {self.num_events}."
                )

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> RawColliderEvent:
        event: dict[str, dict[str, Any]] = {}
        for obj_type, ds in self.datasets.items():
            # Get the row (event) from the dataset
            row = ds[idx]

            # Convert list columns to tensors where appropriate
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, list):
                    processed_row[key] = _convert_list_value(value)
                else:
                    processed_row[key] = value

            event[obj_type] = processed_row

        return event  # type: ignore[return-value]


def collate_fn(batch):
    """
    Custom collate function to handle variable-length events.
    In HEP, each event has a different number of particles/hits.
    For point-cloud based models like Panda, we often want to
    keep events separate until the view-building stage.
    """
    return batch


if __name__ == "__main__":
    # Test the dataset
    print("Testing ColliderMLDataset...")
    dataset = ColliderMLDataset(split="train[:10]")
    print(f"Dataset size: {len(dataset)}")

    # Get the first event
    sample = dataset[0]
    print("\nKeys in first sample:")
    for obj_type in sample.keys():
        print(f"  {obj_type}: {sample[obj_type].keys()}")

    # Check tracker hits
    tracker_hits = sample["tracker_hits"]
    num_hits = len(tracker_hits["x"])
    print(f"\nNumber of tracker hits in event 0: {num_hits}")
    print(f"First 5 tracker hit x coordinates: {tracker_hits['x'][:5]}")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i} size: {len(batch)}")
        if i >= 0:
            break
