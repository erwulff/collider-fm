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
                {
                    "x": [1.0, 2.0],
                    "y": [0.5, 1.5],
                    "z": [3.0, 4.0],
                    "time": [5.0, 6.0],
                    "detector": [7, 8],
                    "volume_id": [9, 10],
                    "layer_id": [11, 12],
                    "labels": [1, 2],
                    "contrib_particle_ids": [[1], [2]],
                },
                {
                    "x": [3.0],
                    "y": [2.5],
                    "z": [5.0],
                    "time": [7.0],
                    "detector": [13],
                    "volume_id": [14],
                    "layer_id": [15],
                    "labels": [3],
                    "contrib_particle_ids": [[3]],
                },
            ],
            "ttbar_pu0_calo_hits": [
                {
                    "x": [4.0],
                    "y": [8.0],
                    "z": [12.0],
                    "total_energy": [16.0],
                    "detector": [17],
                    "labels": [4],
                    "contrib_particle_ids": [[4]],
                },
                {
                    "x": [5.0, 6.0],
                    "y": [9.0, 10.0],
                    "z": [13.0, 14.0],
                    "total_energy": [18.0, 19.0],
                    "detector": [20, 21],
                    "labels": [5, 6],
                    "contrib_particle_ids": [[5], [6]],
                },
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

    @patch("collider_fm.data.load_dataset")
    def test_dataset_preserves_detector_metadata_as_tensors(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset

        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")
        event = dataset[0]

        self.assertTrue(torch.equal(event["tracker_hits"]["detector"], torch.tensor([7, 8])))
        self.assertTrue(torch.equal(event["tracker_hits"]["volume_id"], torch.tensor([9, 10])))
        self.assertTrue(torch.equal(event["tracker_hits"]["layer_id"], torch.tensor([11, 12])))
        self.assertTrue(torch.equal(event["calo_hits"]["detector"], torch.tensor([17])))

    def test_collate_fn_preserves_event_list(self):
        batch = [{"tracker_hits": {"x": torch.tensor([1.0])}}, {"tracker_hits": {"x": torch.tensor([2.0])}}]

        collated = collate_fn(batch)

        self.assertIs(collated, batch)
