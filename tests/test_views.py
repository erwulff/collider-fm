import unittest

import torch

from collider_fm.views import (
    DEFAULT_POINT_GRID_SIZE,
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
                "detector": torch.tensor([10, 13, 9]),
                "total_energy": torch.tensor([10.0, 11.0, 12.0]),
            }
        }

    def test_build_point_view_from_event_is_calo_only(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )

        self.assertEqual(tuple(view["coord"].shape), (3, 3))
        self.assertEqual(tuple(view["feat"].shape), (3, 2))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([3])))
        self.assertAlmostEqual(view["grid_size"].item(), DEFAULT_POINT_GRID_SIZE)
        self.assertTrue(torch.equal(view["energy"], torch.tensor([10.0, 11.0, 12.0])))
        self.assertTrue(torch.equal(view["calo_type"], torch.tensor([0.0, 1.0, 0.0])))

    def test_augment_point_view_preserves_shape_and_detector_type(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        augmented = augment_point_view(
            view, coord_noise_scale=0.0, energy_jitter_scale=0.0
        )

        self.assertEqual(tuple(augmented["coord"].shape), (3, 3))
        self.assertEqual(tuple(augmented["feat"].shape), (3, 2))
        self.assertTrue(torch.equal(augmented["calo_type"], view["calo_type"]))
        self.assertTrue(torch.equal(augmented["energy"], view["energy"]))

    def test_batch_point_views_builds_cumulative_offsets(self):
        first = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        second = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (6, 3))
        self.assertEqual(tuple(batched["feat"].shape), (6, 2))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([3, 6])))

    def test_build_distillation_views_batches_multiple_events(self):
        views = build_distillation_views(
            [self.make_event(), self.make_event()], device=torch.device("cpu")
        )

        self.assertEqual(len(views), 2)
        self.assertTrue(torch.equal(views[0]["offset"], torch.tensor([3, 6])))
        self.assertEqual(tuple(views[0]["feat"].shape), (6, 2))

    def test_sample_hit_indices_keeps_all_hits_when_below_limit(self):
        indices = sample_hit_indices(num_hits=5, max_hits=8, device=torch.device("cpu"))
        self.assertTrue(torch.equal(indices, torch.arange(5)))


if __name__ == "__main__":
    unittest.main()
