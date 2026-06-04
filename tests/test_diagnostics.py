import unittest

import numpy as np
import torch

from collider_fm.diagnostics import compute_pca, tensor_summary


class DiagnosticsTests(unittest.TestCase):
    def test_compute_pca_handles_single_row(self):
        features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        projected = compute_pca(features, n_components=2)

        self.assertEqual(projected.shape, (1, 2))
        self.assertTrue(np.allclose(projected, np.zeros((1, 2), dtype=np.float32)))

    def test_tensor_summary_reports_shape_and_stats(self):
        summary = tensor_summary(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        self.assertEqual(summary["shape"], [2, 2])
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
