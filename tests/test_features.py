import unittest

import torch

from collider_fm.features import build_model_inputs, build_multimodal_points


class FeatureTests(unittest.TestCase):
    def make_event(self):
        return {
            "tracker_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "y": torch.tensor([10.0, 11.0, 12.0, 13.0]),
                "z": torch.tensor([20.0, 21.0, 22.0, 23.0]),
                "time": torch.tensor([0.5, 1.5, 2.5, 3.5]),
                "detector": torch.tensor([1, 1, 2, 2]),
                "volume_id": torch.tensor([3, 3, 4, 4]),
                "layer_id": torch.tensor([5, 6, 7, 8]),
                "surface_id": torch.tensor([9, 10, 11, 12]),
                "particle_id": torch.tensor([100, 101, 102, 103]),
                "true_x": torch.tensor([30.0, 31.0, 32.0, 33.0]),
            },
            "calo_hits": {
                "x": torch.tensor([6.0, 7.0, 8.0]),
                "y": torch.tensor([14.0, 15.0, 16.0]),
                "z": torch.tensor([24.0, 25.0, 26.0]),
                "total_energy": torch.tensor([1.0, 3.0, 7.0]),
                "detector": torch.tensor([13, 14, 15]),
                "contrib_particle_ids": [[1], [2], [3]],
            },
        }

    def test_build_multimodal_points_combines_modalities_without_truth_fields(self):
        points = build_multimodal_points(self.make_event(), device=torch.device("cpu"))
        model_inputs = build_model_inputs(points)

        self.assertEqual(tuple(points.coord.shape), (7, 3))
        self.assertEqual(tuple(points.tracker_continuous.shape), (4, 4))
        self.assertEqual(tuple(points.calo_continuous.shape), (3, 4))
        self.assertTrue(torch.equal(points.modality_id, torch.tensor([0, 0, 0, 0, 1, 1, 1])))
        self.assertTrue(torch.equal(points.offset, torch.tensor([7])))
        self.assertNotIn("particle_id", model_inputs)
        self.assertNotIn("true_x", model_inputs)
        self.assertNotIn("contrib_particle_ids", model_inputs)

    def test_build_multimodal_points_uses_stable_point_ids_after_capping(self):
        points = build_multimodal_points(
            self.make_event(),
            device=torch.device("cpu"),
            max_tracker_hits=2,
            max_calo_hits=2,
        )

        self.assertTrue(torch.equal(points.point_id, torch.tensor([0, 3, 4, 6])))
        self.assertTrue(torch.equal(points.tracker_index, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(points.calo_index, torch.tensor([2, 3])))

    def test_build_multimodal_points_log_scales_calo_energy(self):
        points = build_multimodal_points(self.make_event(), device=torch.device("cpu"))

        expected = torch.log1p(torch.tensor([1.0, 3.0, 7.0]))
        self.assertTrue(torch.allclose(points.calo_continuous[:, 3], expected))


if __name__ == "__main__":
    unittest.main()
