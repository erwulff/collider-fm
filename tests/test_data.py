import unittest
from unittest.mock import patch

import torch

from collider_fm.data import (
    CALO_DATASET_ENERGY_KEY,
    DEFAULT_OBJECT_TYPES,
    TRAIN_SPLIT_ALIAS,
    VAL_SPLIT_ALIAS,
    ColliderMLDataset,
    collate_fn,
    resolve_colliderml_split,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.expected_revision = None
        self.expected_local_files_only = False
        self.expected_split = "train[:2]"

    def fake_load_dataset(
        self,
        dataset_name,
        config_name,
        split,
        cache_dir,
        revision,
        download_config,
    ):
        self.assertEqual(dataset_name, "CERN/ColliderML-Release-1")
        self.assertEqual(split, self.expected_split)
        self.assertEqual(cache_dir, "/tmp/hf-cache")
        self.assertEqual(revision, self.expected_revision)
        self.assertEqual(
            download_config.local_files_only, self.expected_local_files_only
        )

        rows = {
            "ttbar_pu0_calo_hits": [
                {
                    "x": [1.0, 2.0],
                    "y": [3.0, 4.0],
                    "z": [5.0, 6.0],
                    CALO_DATASET_ENERGY_KEY: [7.0, 8.0],
                },
                {
                    "x": [9.0],
                    "y": [10.0],
                    "z": [11.0],
                    CALO_DATASET_ENERGY_KEY: [12.0],
                },
            ],
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
    def test_dataset_preserves_mock_row_values(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset

        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")
        event = dataset[0]

        self.assertEqual(event["calo_hits"]["x"], [1.0, 2.0])
        self.assertEqual(event["calo_hits"][CALO_DATASET_ENERGY_KEY], [7.0, 8.0])

    @patch("collider_fm.data.load_dataset")
    def test_dataset_preserves_total_energy_field(self, mock_load_dataset):
        mock_load_dataset.side_effect = self.fake_load_dataset

        dataset = ColliderMLDataset(split="train[:2]", cache_dir="/tmp/hf-cache")
        event = dataset[1]

        self.assertEqual(event["calo_hits"][CALO_DATASET_ENERGY_KEY], [12.0])

    @patch("collider_fm.data.load_dataset")
    def test_dataset_passes_revision_and_local_only_to_hf(self, mock_load_dataset):
        self.expected_revision = "abc123"
        self.expected_local_files_only = True
        mock_load_dataset.side_effect = self.fake_load_dataset

        dataset = ColliderMLDataset(
            split="train[:2]",
            cache_dir="/tmp/hf-cache",
            dataset_revision="abc123",
            local_files_only=True,
        )

        self.assertEqual(len(dataset), 2)
        _, kwargs = mock_load_dataset.call_args
        self.assertEqual(kwargs["revision"], "abc123")
        self.assertTrue(kwargs["download_config"].local_files_only)

    def test_collate_fn_preserves_event_list(self):
        batch = [
            {"calo_hits": {"x": torch.tensor([1.0])}},
            {"calo_hits": {"x": torch.tensor([2.0])}},
        ]

        collated = collate_fn(batch)

        self.assertIs(collated, batch)

    def test_resolve_colliderml_split_aliases(self):
        self.assertEqual(resolve_colliderml_split("train"), TRAIN_SPLIT_ALIAS)
        self.assertEqual(resolve_colliderml_split("val"), VAL_SPLIT_ALIAS)
        self.assertEqual(resolve_colliderml_split("train[:32]"), "train[:32]")
        self.assertEqual(resolve_colliderml_split("val[:100]"), "train[950000:950100]")
        self.assertEqual(
            resolve_colliderml_split("val[:50000]"), "train[950000:1000000]"
        )
        self.assertRaises(ValueError, resolve_colliderml_split, "train[:960000]")
        self.assertRaises(ValueError, resolve_colliderml_split, "val[:960000]")

    def test_resolve_colliderml_split_rejects_train_requests_past_holdout_boundary(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "exceeds the project train window"):
            resolve_colliderml_split("train[:960000]")

    def test_resolve_colliderml_split_rejects_val_requests_past_holdout_boundary(self):
        with self.assertRaisesRegex(ValueError, "exceeds the project val window"):
            resolve_colliderml_split("val[:60000]")

    @patch("collider_fm.data.load_dataset")
    def test_dataset_uses_project_val_alias(self, mock_load_dataset):
        self.expected_split = VAL_SPLIT_ALIAS
        mock_load_dataset.side_effect = self.fake_load_dataset

        ColliderMLDataset(split="val", cache_dir="/tmp/hf-cache")

        _, kwargs = mock_load_dataset.call_args
        self.assertEqual(kwargs["split"], VAL_SPLIT_ALIAS)

    @patch("collider_fm.data.load_dataset")
    def test_dataset_uses_project_val_slice_alias(self, mock_load_dataset):
        self.expected_split = "train[950000:950100]"
        mock_load_dataset.side_effect = self.fake_load_dataset

        ColliderMLDataset(split="val[:100]", cache_dir="/tmp/hf-cache")

        _, kwargs = mock_load_dataset.call_args
        self.assertEqual(kwargs["split"], "train[950000:950100]")


if __name__ == "__main__":
    unittest.main()
