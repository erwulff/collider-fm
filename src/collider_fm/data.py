from __future__ import annotations

"""Dataset helpers for the calo-only ColliderFM pipeline."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader, Dataset

DEFAULT_OBJECT_TYPES = ("calo_hits",)
CALO_ENERGY_KEYS = ("energy", "totalenergy", "total_energy")
TRAIN_SPLIT_START = 0
TRAIN_SPLIT_STOP = 950000
VAL_SPLIT_START = 950000
VAL_SPLIT_STOP = 1000000
TRAIN_SPLIT_ALIAS = f"train[:{TRAIN_SPLIT_STOP}]"
VAL_SPLIT_ALIAS = f"train[{VAL_SPLIT_START}:{VAL_SPLIT_STOP}]"


def _parse_project_split_slice(split: str) -> tuple[str, int | None, int | None]:
    if split in {"train", "val"}:
        return split, None, None
    if not split.endswith("]") or "[" not in split:
        return split, None, None
    prefix, slice_expr = split[:-1].split("[", maxsplit=1)
    if prefix not in {"train", "val"} or ":" not in slice_expr:
        return split, None, None
    start_text, stop_text = slice_expr.split(":", maxsplit=1)
    start = None if start_text == "" else int(start_text)
    stop = None if stop_text == "" else int(stop_text)
    return prefix, start, stop


def _resolve_split_window(split_name: str) -> tuple[int, int]:
    if split_name == "train":
        return TRAIN_SPLIT_START, TRAIN_SPLIT_STOP
    if split_name == "val":
        return VAL_SPLIT_START, VAL_SPLIT_STOP
    raise ValueError(f"Unsupported project split alias: {split_name}")


def resolve_colliderml_split(split: str) -> str:
    """Map project-level train/val aliases onto explicit HF `train[...]` slices.

    The upstream dataset only exposes `train`, so we reserve the last 50k events for
    validation and allow bounded slicing within the project-level `train` and `val`
    windows.
    """

    split_name, start, stop = _parse_project_split_slice(split)
    if split_name not in {"train", "val"}:
        return split

    window_start, window_stop = _resolve_split_window(split_name)
    relative_start = 0 if start is None else start
    relative_stop = (window_stop - window_start) if stop is None else stop

    if relative_start < 0 or relative_stop < 0:
        raise ValueError(f"Negative indices are not supported for split '{split}'.")
    if relative_start > relative_stop:
        raise ValueError(
            f"Split '{split}' has start {relative_start} larger than stop {relative_stop}."
        )

    window_size = window_stop - window_start
    if relative_stop > window_size:
        raise ValueError(
            f"Split '{split}' exceeds the project {split_name} window of {window_size} events."
        )

    absolute_start = window_start + relative_start
    absolute_stop = window_start + relative_stop
    if absolute_start == 0:
        return f"train[:{absolute_stop}]"
    return f"train[{absolute_start}:{absolute_stop}]"


def _convert_list_value(value: list[Any]) -> Any:
    """Convert flat numeric lists to tensors while keeping ragged lists intact."""

    if len(value) > 0 and isinstance(value[0], list):
        return value

    array = np.asarray(value)
    if array.dtype == object:
        return value
    return torch.tensor(array)


def _apply_object_aliases(obj_type: str, row: dict[str, Any]) -> dict[str, Any]:
    """Add stable field aliases used by the rest of the project."""

    if obj_type != "calo_hits":
        return row

    energy_key = next((key for key in CALO_ENERGY_KEYS if key in row), None)
    if energy_key is None:
        return row

    energy_value = row[energy_key]
    row.setdefault("energy", energy_value)
    row.setdefault("totalenergy", energy_value)
    row.setdefault("total_energy", energy_value)
    return row


class ColliderMLDataset(Dataset):
    """
    A PyTorch Dataset for loading ColliderML data from Hugging Face datasets.
    It combines one or more ColliderML object configurations for the same events.

    The current project default is calorimeter hits only. Calorimeter rows receive a
    canonical `energy` alias so downstream code can use one stable field name while
    remaining compatible with both `totalenergy` and `total_energy` dataset variants.
    """

    def __init__(
        self,
        dataset_name: str = "CERN/ColliderML-Release-1",
        dataset_type: str = "ttbar",
        pu_config: str = "pu0",
        object_types: Sequence[str] = DEFAULT_OBJECT_TYPES,
        split: str = "train",
        cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
        dataset_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:

        self.dataset_name = dataset_name
        self.object_types = tuple(object_types)
        self.dataset_revision = dataset_revision
        self.local_files_only = local_files_only
        if not self.object_types:
            raise ValueError(
                "'object_types' must contain at least one dataset configuration."
            )
        self.datasets = {}
        resolved_split = resolve_colliderml_split(split)

        for obj_type in self.object_types:
            config_name = f"{dataset_type}_{pu_config}_{obj_type}"
            print(f"Loading {config_name}...")
            self.datasets[obj_type] = load_dataset(
                dataset_name,
                config_name,
                split=resolved_split,
                cache_dir=cache_dir,
                revision=dataset_revision,
                download_config=DownloadConfig(local_files_only=local_files_only),
            )

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

    def __getitem__(self, idx: int) -> dict[str, dict[str, Any]]:
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

            event[obj_type] = _apply_object_aliases(obj_type, processed_row)

        return event


def collate_fn(
    batch: list[dict[str, dict[str, Any]]],
) -> list[dict[str, dict[str, Any]]]:
    """
    Custom collate function to handle variable-length events.
    ColliderML events have variable numbers of calorimeter cells, so we keep events
    separate until the view-building stage.
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

    calo_hits = sample["calo_hits"]
    num_hits = len(calo_hits["x"])
    print(f"\nNumber of calorimeter hits in event 0: {num_hits}")
    print(f"First 5 calorimeter energies: {calo_hits['energy'][:5]}")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i} size: {len(batch)}")
        if i >= 0:
            break
