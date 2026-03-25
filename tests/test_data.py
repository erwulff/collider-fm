import unittest
from unittest.mock import patch

import torch

from collider_fm.data import DEFAULT_OBJECT_TYPES, ColliderMLDataset, collate_fn


class DatasetTests(unittest.TestCase):
    def fake_load_dataset(self, dataset_name, config_name, split, cache_dir):
        self.assertEqual(dataset_name, "CERN/ColliderML-Release-1")
        self.assertEqual(split, "train[:2]")
        self.assertEqual(cache_dir, "/tmp/hf-cache")

        rows = {
            "ttbar_pu0_tracker_hits": [
                {"x": [1.0, 2.0], "labels": [1, 2], "contrib_particle_ids": [[1], [2]]},
                {"x": [3.0], "labels": [3], "contrib_particle_ids": [[3]]},
            ],
            "ttbar_pu0_calo_hits": [
                {"x": [4.0], "labels": [4], "contrib_particle_ids": [[4]]},
                {"x": [5.0, 6.0], "labels": [5, 6], "contrib_particle_ids": [[5], [6]]},
            ],
        }
        return rows[config_name]

    @patch("collider_fm.data.load_dataset")
    def test_dataset_defaults_to_tracker_and_calo_hits(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset

        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")

        self.assertEqual(dataset.object_types, DEFAULT_OBJECT_TYPES)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(mock_load_dataset.call_count, 2)

    @patch("collider_fm.data.load_dataset")
    def test_dataset_converts_numeric_lists_to_tensors(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset

        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")
        event = dataset[0]

        self.assertIsInstance(event["tracker_hits"]["x"], torch.Tensor)
        self.assertTrue(torch.equal(event["tracker_hits"]["labels"], torch.tensor([1, 2])))
        self.assertEqual(event["tracker_hits"]["contrib_particle_ids"], [[1], [2]])

    def test_collate_fn_preserves_event_list(self):
        batch = [{"tracker_hits": {"x": torch.tensor([1.0])}}, {"tracker_hits": {"x": torch.tensor([2.0])}}]

        collated = collate_fn(batch)

        self.assertIs(collated, batch)
