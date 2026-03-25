import unittest
from unittest.mock import patch
from typing import cast

import torch

from collider_fm.data import DEFAULT_OBJECT_TYPES, ColliderMLDataset, collate_fn


class DatasetTests(unittest.TestCase):
    def fake_load_dataset(self, dataset_name, config_name, split, cache_dir):
        self.assertEqual(dataset_name, "CERN/ColliderML-Release-1")
        self.assertEqual(split, "train[:2]")
        self.assertEqual(cache_dir, "/tmp/hf-cache")

        rows = {
            "ttbar_pu0_calo_hits": [
                {
                    "x": [1.0, 2.0],
                    "y": [3.0, 4.0],
                    "z": [5.0, 6.0],
                    "detector": [10, 13],
                    "total_energy": [7.0, 8.0],
                    "contrib_particle_ids": [[1], [2]],
                },
                {
                    "x": [9.0],
                    "y": [10.0],
                    "z": [11.0],
                    "detector": [9],
                    "energy": [12.0],
                    "contrib_particle_ids": [[3]],
                },
            ]
        }
        return rows[config_name]

    @patch("collider_fm.data.load_dataset")
    def test_dataset_defaults_to_calo_hits(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset
        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")

        self.assertEqual(dataset.object_types, DEFAULT_OBJECT_TYPES)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(mock_load_dataset.call_count, 1)

    @patch("collider_fm.data.load_dataset")
    def test_dataset_adds_energy_aliases(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset
        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")
        event = dataset[0]
        energy = cast(torch.Tensor, event["calo_hits"]["energy"])
        total_energy = cast(torch.Tensor, event["calo_hits"]["total_energy"])

        self.assertTrue(torch.equal(energy, torch.tensor([7.0, 8.0])))
        self.assertTrue(torch.equal(total_energy, torch.tensor([7.0, 8.0])))
        self.assertEqual(event["calo_hits"]["contrib_particle_ids"], [[1], [2]])

    def test_collate_fn_preserves_event_list(self):
        batch = cast(
            list[dict[str, dict[str, object]]],
            [
                {"calo_hits": {"x": torch.tensor([1.0])}},
                {"calo_hits": {"x": torch.tensor([2.0])}},
            ],
        )
        self.assertIs(collate_fn(batch), batch)


if __name__ == "__main__":
    unittest.main()
