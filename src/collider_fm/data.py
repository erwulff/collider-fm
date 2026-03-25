from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


DEFAULT_OBJECT_TYPES = ("calo_hits",)
CALO_ENERGY_KEYS = ("energy", "totalenergy", "total_energy")


def _convert_list_value(value: list[object]) -> object:
    if value and isinstance(value[0], list):
        return value

    array = np.asarray(value)
    if array.dtype == object:
        return value
    return torch.tensor(array)


def _add_calo_energy_aliases(row: dict[str, object]) -> dict[str, object]:
    energy_key = next((key for key in CALO_ENERGY_KEYS if key in row), None)
    if energy_key is None:
        return row

    energy = row[energy_key]
    row.setdefault("energy", energy)
    row.setdefault("totalenergy", energy)
    row.setdefault("total_energy", energy)
    return row


class ColliderMLDataset(Dataset):
    """Small wrapper around the ColliderML Hugging Face dataset.

    For this phase we keep the default path intentionally simple: one event is one
    dictionary containing only the requested object tables, with `calo_hits` as the
    default and recommended choice.
    """

    def __init__(
        self,
        dataset_name: str = "CERN/ColliderML-Release-1",
        dataset_type: str = "ttbar",
        pu_config: str = "pu0",
        object_types: Sequence[str] = DEFAULT_OBJECT_TYPES,
        split: str = "train",
        cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
    ) -> None:
        self.dataset_name = dataset_name
        self.object_types = tuple(object_types)
        if not self.object_types:
            raise ValueError(
                "'object_types' must contain at least one dataset configuration."
            )

        self.datasets: dict[str, Any] = {}
        for object_type in self.object_types:
            config_name = f"{dataset_type}_{pu_config}_{object_type}"
            self.datasets[object_type] = load_dataset(
                dataset_name, config_name, split=split, cache_dir=cache_dir
            )

        first_object_type = self.object_types[0]
        self.num_events = len(self.datasets[first_object_type])
        for object_type in self.object_types[1:]:
            if len(self.datasets[object_type]) != self.num_events:
                raise ValueError(
                    "All requested ColliderML tables must have the same number of events."
                )

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, index: int) -> dict[str, dict[str, object]]:
        event: dict[str, dict[str, object]] = {}
        for object_type, dataset in self.datasets.items():
            row: dict[str, object] = dataset[index]
            processed_row: dict[str, object] = {}
            for key, value in row.items():
                if isinstance(value, list):
                    processed_row[key] = _convert_list_value(value)
                else:
                    processed_row[key] = value

            if object_type == "calo_hits":
                processed_row = _add_calo_energy_aliases(processed_row)
            event[object_type] = processed_row
        return event


def collate_fn(
    batch: list[dict[str, dict[str, object]]],
) -> list[dict[str, dict[str, object]]]:
    """Keep events as a plain list until the view-building step."""
    return batch
