import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

# t-SNE / matplotlib are only needed for tsne_plot; keep Agg + no display.
os.environ.setdefault("MPLBACKEND", "Agg")

from collider_fm.visualization import (
    match_to_input_coords,
    tsne_plot,
)


class MatchToInputCoordsTests(unittest.TestCase):
    def test_identical_coords_is_identity(self):
        # Output == input -> each output's nearest input is itself (row i -> i).
        coord = torch.randn(50, 3)
        idx = match_to_input_coords(coord, coord)
        self.assertTrue(torch.equal(idx, torch.arange(50)))

    def test_permuted_output_recovers(self):
        # Output is a permutation of input -> recover the permutation.
        coord = torch.randn(40, 3)
        perm = torch.randperm(40)
        idx = match_to_input_coords(coord[perm], coord)
        self.assertTrue(torch.equal(idx, perm))

    def test_output_subset_maps_to_correct_rows(self):
        # Output is a subset of input rows -> each maps back to its source row.
        coord = torch.randn(30, 3)
        keep = torch.tensor([3, 7, 11, 20])
        idx = match_to_input_coords(coord[keep], coord)
        self.assertTrue(torch.equal(idx, keep))

    def test_distinct_points_match_nearest(self):
        # Two well-separated clusters -> each output maps to its own cluster's input.
        a = torch.randn(10, 3) + 10.0
        b = torch.randn(10, 3) - 10.0
        in_coord = torch.cat([a, b], dim=0)
        out_coord = torch.cat([a[:3], b[:3]], dim=0)  # rows 0,1,2 and 10,11,12
        idx = match_to_input_coords(out_coord, in_coord)
        self.assertTrue(torch.equal(idx, torch.tensor([0, 1, 2, 10, 11, 12])))


class TsnePlotTests(unittest.TestCase):
    def test_writes_png_file(self):
        # Smoke: synthetic 2-cluster features -> PNG file exists after plotting.
        torch.manual_seed(0)
        a = torch.randn(60, 8) + 5.0
        b = torch.randn(60, 8) - 5.0
        features = torch.cat([a, b], dim=0)
        color = torch.cat([torch.zeros(60), torch.ones(60)]).long()

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tsne.png"
            tsne_plot(features, color, path, title="smoke", max_points=200, seed=0)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_drops_unknown_labels(self):
        # All-unknown (-1) color -> nothing to plot -> no file written (no crash).
        features = torch.randn(20, 4)
        color = torch.full((20,), -1, dtype=torch.long)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tsne.png"
            tsne_plot(features, color, path, max_points=200, seed=0)
            self.assertFalse(path.exists())

    def test_categorical_plot_writes_png(self):
        # Discrete categories path: integer bucket ids -> PNG with a legend, not a
        # colorbar. Exercises the pdg-bucket coloring code path.
        torch.manual_seed(0)
        a = torch.randn(60, 8) + 5.0
        b = torch.randn(60, 8) - 5.0
        features = torch.cat([a, b], dim=0)
        color = torch.cat([torch.zeros(60), torch.ones(60)]).long()
        categories = [("a", "tab:red"), ("b", "tab:blue")]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tsne_cat.png"
            tsne_plot(
                features, color, path, title="cat", max_points=200, seed=0,
                categories=categories,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
