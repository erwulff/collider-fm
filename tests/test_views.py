import unittest

import torch

from collider_fm.views import (
    DEFAULT_POINT_GRID_SIZE,
    POINT_FEATURE_DIM,
    augment_point_view,
    batch_point_views,
    build_distillation_views,
    build_point_view_from_event,
    sample_hit_indices,
)


class ViewTests(unittest.TestCase):
    def make_event(self):
        return {
            "calo_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([4.0, 5.0, 6.0]),
                "z": torch.tensor([7.0, 8.0, 9.0]),
                "totalenergy": torch.tensor([10.0, 11.0, 12.0]),
            }
        }

    def test_build_point_view_from_event_uses_calo_hits_only(self):
        event = self.make_event()

        view = build_point_view_from_event(event, device=torch.device("cpu"))
        augmented = augment_point_view(view, coord_noise_scale=0.0, feat_noise_scale=0.0)

        self.assertEqual(tuple(view["coord"].shape), (3, 3))
        self.assertEqual(tuple(view["feat"].shape), (3, POINT_FEATURE_DIM))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([3])))
        self.assertAlmostEqual(view["grid_size"].item(), DEFAULT_POINT_GRID_SIZE)
        self.assertTrue(torch.equal(augmented["coord"], view["coord"]))
        self.assertTrue(torch.equal(augmented["feat"], view["feat"]))
        self.assertTrue(torch.equal(view["energy"], torch.tensor([10.0, 11.0, 12.0])))

    def test_augment_point_view_preserves_indices_and_updates_energy(self):
        view = build_point_view_from_event(self.make_event(), device=torch.device("cpu"))

        augmented = augment_point_view(view, coord_noise_scale=0.5, feat_noise_scale=0.1)

        self.assertTrue(torch.allclose(augmented["feat"][:, :3], augmented["coord"]))
        self.assertTrue(torch.equal(augmented["feat"][:, 3], augmented["energy"]))
        self.assertTrue(torch.equal(augmented["source_index"], view["source_index"]))
        self.assertEqual(augmented["mask"].dtype, torch.bool)

    def test_batch_point_views_builds_cumulative_offsets_and_unique_indices(self):
        first = build_point_view_from_event(self.make_event(), device=torch.device("cpu"))
        second = build_point_view_from_event(self.make_event(), device=torch.device("cpu"))

        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (6, 3))
        self.assertEqual(tuple(batched["feat"].shape), (6, POINT_FEATURE_DIM))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([3, 6])))
        self.assertTrue(torch.equal(batched["source_index"], torch.tensor([0, 1, 2, 3, 4, 5])))
        self.assertTrue(torch.equal(batched["patch_id"], torch.tensor([0, 0, 0, 1, 1, 1])))

    def test_build_distillation_views_batches_multiple_events(self):
        views = build_distillation_views(
            [self.make_event(), self.make_event()],
            device=torch.device("cpu"),
            coord_noise_scale=0.0,
            feat_noise_scale=0.0,
        )

        self.assertEqual(len(views), 2)
        self.assertTrue(torch.equal(views[0]["offset"], torch.tensor([3, 6])))
        self.assertTrue(torch.equal(views[0]["feat"], views[1]["feat"]))

    def test_sample_hit_indices_keeps_all_hits_when_below_limit(self):
        indices = sample_hit_indices(num_hits=5, max_hits=8, device=torch.device("cpu"))

        self.assertTrue(torch.equal(indices, torch.arange(5)))

    def test_build_point_view_rejects_empty_event(self):
        event = {
            "calo_hits": {
                "x": torch.tensor([]),
                "y": torch.tensor([]),
                "z": torch.tensor([]),
                "energy": torch.tensor([]),
            }
        }

        with self.assertRaisesRegex(ValueError, "zero calorimeter hits"):
            build_point_view_from_event(event, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
