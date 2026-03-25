import unittest

import numpy as np
import torch

from collider_fm.diagnostics import compute_pca, sample_indices, tensor_summary
from collider_fm.view_diagnostics import summarize_ssl_view, summarize_ssl_view_set
from collider_fm.views import SSLViewConfig, build_ssl_views


class DiagnosticsTests(unittest.TestCase):
    def make_event(self):
        return {
            "tracker_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "y": torch.tensor([5.0, 6.0, 7.0, 8.0]),
                "z": torch.tensor([9.0, 10.0, 11.0, 12.0]),
                "time": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "detector": torch.tensor([1, 1, 2, 2]),
                "volume_id": torch.tensor([3, 3, 4, 4]),
                "layer_id": torch.tensor([5, 6, 7, 8]),
                "surface_id": torch.tensor([9, 10, 11, 12]),
            },
            "calo_hits": {
                "x": torch.tensor([20.0, 21.0, 22.0]),
                "y": torch.tensor([23.0, 24.0, 25.0]),
                "z": torch.tensor([26.0, 27.0, 28.0]),
                "total_energy": torch.tensor([1.0, 2.0, 3.0]),
                "detector": torch.tensor([13, 14, 15]),
            },
        }

    def test_compute_pca_handles_single_row(self):
        features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        projected = compute_pca(features, n_components=2)

        self.assertEqual(projected.shape, (1, 2))
        self.assertTrue(np.allclose(projected, np.zeros((1, 2), dtype=np.float32)))

    def test_sample_indices_is_reproducible(self):
        first = sample_indices(num_items=100, max_items=10, seed=7)
        second = sample_indices(num_items=100, max_items=10, seed=7)

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 10)

    def test_tensor_summary_reports_shape_and_stats(self):
        summary = tensor_summary(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        self.assertEqual(summary["shape"], [2, 2])
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["mean"], 2.5)

    def test_summarize_ssl_view_reports_counts_and_fractions(self):
        torch.manual_seed(19)
        config = SSLViewConfig(teacher_global_views=2, phi_rotation_max=0.0, coord_jitter_scale=0.0)
        view_set = build_ssl_views([self.make_event()], device=torch.device("cpu"), config=config)

        summary = summarize_ssl_view(view_set["teacher_global"][0])

        self.assertEqual(summary["view_type"], "teacher_global")
        self.assertEqual(summary["num_events"], 1)
        self.assertGreater(summary["selected_point_count"], 0)
        self.assertGreaterEqual(summary["visible_fraction"], 0.0)
        self.assertLessEqual(summary["visible_fraction"], 1.0)
        self.assertAlmostEqual(summary["tracker_fraction"] + summary["calo_fraction"], 1.0)

    def test_summarize_ssl_view_set_aggregates_by_view_family(self):
        torch.manual_seed(23)
        config = SSLViewConfig(
            teacher_global_views=2,
            student_global_views=1,
            student_local_views=1,
            student_masked_views=1,
            phi_rotation_max=0.0,
            coord_jitter_scale=0.0,
        )
        view_set = build_ssl_views(
            [self.make_event(), self.make_event()],
            device=torch.device("cpu"),
            config=config,
        )

        summary = summarize_ssl_view_set(view_set)

        self.assertEqual(summary["teacher_global"]["num_views"], 2)
        self.assertEqual(summary["student_local"]["num_views"], 1)
        self.assertEqual(len(summary["student_masked"]["views"]), 1)
        self.assertGreater(summary["teacher_global"]["mean_selected_point_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
