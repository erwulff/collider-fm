import unittest

import torch

from collider_fm.views import (
    DEFAULT_POINT_GRID_SIZE,
    augment_point_view,
    batch_point_views,
    build_distillation_views,
    build_point_view_from_event,
    normalize_feature,
    sample_hit_indices,
)


class ViewTests(unittest.TestCase):
    def make_event(self):
        return {
            "tracker_hits": {
                "x": torch.tensor([1.0, 2.0]),
                "y": torch.tensor([3.0, 4.0]),
                "z": torch.tensor([5.0, 6.0]),
                "time": torch.tensor([7.0, 8.0]),
            },
            "calo_hits": {
                "x": torch.tensor([10.0]),
                "y": torch.tensor([11.0]),
                "z": torch.tensor([12.0]),
                "total_energy": torch.tensor([13.0]),
            },
        }

    def test_build_point_view_from_event_combines_hits(self):
        event = self.make_event()

        view = build_point_view_from_event(event, device=torch.device("cpu"))
        augmented = augment_point_view(view, coord_noise_scale=0.0, feat_noise_scale=0.0)

        self.assertEqual(tuple(view["coord"].shape), (3, 3))
        self.assertEqual(tuple(view["feat"].shape), (3, 6))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([3])))
        self.assertAlmostEqual(view["grid_size"].item(), DEFAULT_POINT_GRID_SIZE)
        self.assertTrue(torch.equal(augmented["coord"], view["coord"]))
        self.assertTrue(torch.equal(augmented["feat"], view["feat"]))

    def test_augment_point_view_recomputes_derived_features(self):
        view = build_point_view_from_event(self.make_event(), device=torch.device("cpu"))

        augmented = augment_point_view(view, coord_noise_scale=0.5, feat_noise_scale=0.0)

        self.assertTrue(torch.allclose(augmented["feat"][:, :3], augmented["coord"]))
        expected_radius = normalize_feature(torch.linalg.norm(augmented["coord"], dim=1))
        self.assertTrue(torch.allclose(augmented["feat"][:, 3], expected_radius))
        self.assertTrue(torch.equal(augmented["feat"][:, 5], view["feat"][:, 5]))

    def test_batch_point_views_builds_cumulative_offsets(self):
        first = {
            "coord": torch.randn(2, 3),
            "feat": torch.randn(2, 6),
            "offset": torch.tensor([2]),
            "grid_size": torch.tensor(DEFAULT_POINT_GRID_SIZE),
        }
        second = {
            "coord": torch.randn(3, 3),
            "feat": torch.randn(3, 6),
            "offset": torch.tensor([3]),
            "grid_size": torch.tensor(DEFAULT_POINT_GRID_SIZE),
        }

        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (5, 3))
        self.assertEqual(tuple(batched["feat"].shape), (5, 6))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([2, 5])))

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


if __name__ == "__main__":
    unittest.main()
