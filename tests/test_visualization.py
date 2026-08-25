import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

# t-SNE / matplotlib are only needed for tsne_plot; keep Agg + no display.
os.environ.setdefault("MPLBACKEND", "Agg")

from collider_fm._panda.structure import Point
from collider_fm.visualization import (
    _cluster_dominant_pdg,
    make_2d_embedding_plots,
    match_to_input_coords,
    pca_plot,
    tsne_plot,
    upcast2_input_map,
)
from collider_fm.visualization import TsnePointCollection


def _pooling_point(feat, parent, inverse):
    """Build a Point that looks like a GridPooling output: feat + parent + inverse.

    ``inverse`` is the [N_finer] -> coarse cluster map stored by GridPooling; ``parent``
    is the finer-level Point. Mirrors model_base.py:553-554.
    """
    return Point({"feat": feat, "pooling_parent": parent, "pooling_inverse": inverse})


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


class Upcast2InputMapTests(unittest.TestCase):
    """The up_cast(2) point -> input-cluster inversion (verified exact on real data)."""

    def _build_chain(self):
        """A 4-level chain: stage0(8) <- stage1(4) <- stage2(2) <- stage3(1).

        Each level halves the count (pairs of finer points pool into one coarser).
        Returns the deepest point (stage3) with the full breadcrumb chain attached,
        exactly as the backbone leaves it. up_cast(2) from stage3 lands at stage1 (4
        points) with a 1-level remainder (stage1 -> stage0) to invert.
        """
        # stage0: 8 points; consecutive pairs pool into stage1's 4 points.
        s0 = Point({"feat": torch.arange(1, 9).reshape(8, 1).float(),
                    "coord": torch.randn(8, 3)})
        inv01 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # stage0 -> stage1 cluster
        s1 = _pooling_point(torch.arange(1, 5).reshape(4, 1).float(), s0, inv01)

        inv12 = torch.tensor([0, 0, 1, 1])  # stage1 -> stage2 cluster
        s2 = _pooling_point(torch.arange(1, 3).reshape(2, 1).float(), s1, inv12)

        inv23 = torch.tensor([0, 0])  # stage2 (2 pts) -> stage3 (1 cluster)
        s3 = _pooling_point(torch.arange(1, 2).reshape(1, 1).float(), s2, inv23)
        return s3

    def test_upcast2_lands_two_levels_finer(self):
        # From stage3 (1 pt), up_cast(2) -> stage1 (4 pts); remainder chain depth = 1.
        s3 = self._build_chain()
        upcast_point, input_to_cluster = upcast2_input_map(s3, up_cast_level=2)
        self.assertEqual(upcast_point.feat.shape[0], 4)  # stage1 has 4 points
        self.assertEqual(input_to_cluster.shape[0], 8)   # 8 stage-0 input points

    def test_input_to_cluster_recovers_exact_membership(self):
        # Each up_cast(2) point (stage1) corresponds to a pair of stage-0 points.
        s3 = self._build_chain()
        _, input_to_cluster = upcast2_input_map(s3, up_cast_level=2)
        # stage1 point 0 <- stage0 {0,1}, point 1 <- {2,3}, point 2 <- {4,5}, 3 <- {6,7}.
        expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.assertTrue(torch.equal(input_to_cluster, expected))

    def test_remainder_zero_when_chain_shallow(self):
        # up_cast_level == chain depth (3 pooling levels) -> lands at stage0, no
        # remainder, identity map.
        s3 = self._build_chain()
        upcast_point, input_to_cluster = upcast2_input_map(s3, up_cast_level=3)
        self.assertEqual(upcast_point.feat.shape[0], 8)  # stage0
        self.assertTrue(torch.equal(input_to_cluster, torch.arange(8)))


class ClusterDominantPdgTests(unittest.TestCase):
    def test_energy_weighted_dominant(self):
        # 2 clusters over 4 input points. Cluster 0: hits with pdg 211 (energy 1) and
        # 11 (energy 5) -> 11 wins (more energy). Cluster 1: pdg 211 (energy 3) only.
        input_to_cluster = np.array([0, 0, 1, 1])
        raw_hit = np.array([0, 1, 2, 3])
        pdg_per_hit = np.array([211, 11, 211, 211])
        hit_energy = np.array([1.0, 5.0, 3.0, 0.0])

        out = _cluster_dominant_pdg(input_to_cluster, raw_hit, pdg_per_hit, hit_energy, 2)
        np.testing.assert_array_equal(out, [11, 211])

    def test_all_unknown_is_minus_one(self):
        # Every hit's pdg is -1 -> clusters get -1.
        out = _cluster_dominant_pdg(
            np.array([0, 1]), np.array([0, 1]),
            np.array([-1, -1]), np.array([1.0, 1.0]), 2,
        )
        np.testing.assert_array_equal(out, [-1, -1])

    def test_antiparticles_not_treated_as_unknown(self):
        # Antiparticles are valid negative pdgs (-11 = e+); they must NOT be filtered
        # out by the `!= -1` known-mask. A positron-dominated cluster returns -11, not -1.
        out = _cluster_dominant_pdg(
            np.array([0, 0, 1]), np.array([0, 1, 2]),
            np.array([-11, 211, -1]),  # cluster0: e+ vs pi+ ; cluster1: unknown
            np.array([5.0, 1.0, 3.0]),
            2,
        )
        np.testing.assert_array_equal(out, [-11, -1])  # e+ wins cluster0 by energy


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


class PcaPlotTests(unittest.TestCase):
    def test_writes_png_file(self):
        # Smoke: synthetic 2-cluster features -> PNG file exists after plotting.
        torch.manual_seed(0)
        a = torch.randn(60, 8) + 5.0
        b = torch.randn(60, 8) - 5.0
        features = torch.cat([a, b], dim=0)
        color = torch.cat([torch.zeros(60), torch.ones(60)]).long()

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pca.png"
            pca_plot(features, color, path, title="smoke", max_points=200, seed=0)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_drops_unknown_labels(self):
        # All-unknown (-1) color -> nothing to plot -> no file written (no crash).
        features = torch.randn(20, 4)
        color = torch.full((20,), -1, dtype=torch.long)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pca.png"
            pca_plot(features, color, path, max_points=200, seed=0)
            self.assertFalse(path.exists())


class Make2dEmbeddingPlotsTests(unittest.TestCase):
    def _collection(self, n=40):
        torch.manual_seed(0)
        return TsnePointCollection(
            features=torch.randn(n, 8),
            pdg_id=torch.cat([torch.full((n // 2,), 11), torch.full((n // 2,), 211)]).long(),
            event_id=torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long(),
            num_events=2,
        )

    def test_writes_pdg_and_event_id_plots(self):
        with TemporaryDirectory() as tmp:
            paths = make_2d_embedding_plots(self._collection(), tmp, seed=0)
            names = {Path(p).name for p in paths}
            self.assertEqual(names, {"tsne_pdg_id.png", "tsne_event_id.png"})

    def test_subdir_writes_to_subdirectory(self):
        with TemporaryDirectory() as tmp:
            paths = make_2d_embedding_plots(self._collection(), tmp, seed=0, subdir="upcast2")
            self.assertTrue(all("upcast2" in p for p in paths))
            self.assertTrue((Path(tmp) / "upcast2" / "tsne_event_id.png").exists())

    def test_pca_method_writes_pca_plots(self):
        with TemporaryDirectory() as tmp:
            paths = make_2d_embedding_plots(self._collection(), tmp, method="pca", seed=0)
            names = {Path(p).name for p in paths}
            self.assertEqual(names, {"pca_pdg_id.png", "pca_event_id.png"})


if __name__ == "__main__":
    unittest.main()
