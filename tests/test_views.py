import unittest

import torch

from collider_fm.views import augment_point_view, batch_point_views, build_point_view_from_event


class ViewTests(unittest.TestCase):
    def test_build_point_view_from_event_combines_hits(self):
        event = {
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

        view = build_point_view_from_event(event, device=torch.device("cpu"))
        augmented = augment_point_view(view, coord_noise_scale=0.0, feat_noise_scale=0.0)

        self.assertEqual(tuple(view["coord"].shape), (3, 3))
        self.assertEqual(tuple(view["feat"].shape), (3, 6))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([3])))
        self.assertTrue(torch.equal(augmented["coord"], view["coord"]))
        self.assertTrue(torch.equal(augmented["feat"], view["feat"]))

    def test_batch_point_views_builds_cumulative_offsets(self):
        first = {
            "coord": torch.randn(2, 3),
            "feat": torch.randn(2, 6),
            "offset": torch.tensor([2]),
            "grid_size": torch.tensor(10.0),
        }
        second = {
            "coord": torch.randn(3, 3),
            "feat": torch.randn(3, 6),
            "offset": torch.tensor([3]),
            "grid_size": torch.tensor(10.0),
        }

        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (5, 3))
        self.assertEqual(tuple(batched["feat"].shape), (5, 6))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([2, 5])))


if __name__ == "__main__":
    unittest.main()
