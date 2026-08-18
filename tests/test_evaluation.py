import math
import unittest

import torch

from collider_fm.evaluation import (
    EmbeddingCollection,
    alignment,
    nn_retrieval,
    participation_ratio,
    summarize,
    uniformity,
)


class ParticipationRatioTests(unittest.TestCase):
    def test_rank_one_matrix_is_one(self):
        # Every row identical -> rank 1 -> participation ratio ~ 1.
        row = torch.randn(1, 4)
        features = row.repeat(1000, 1)

        rank, spectrum = participation_ratio(features)

        self.assertAlmostEqual(rank, 1.0, places=5)
        self.assertEqual(len(spectrum), 4)
        # Only the leading singular value is non-trivial.
        self.assertGreater(spectrum[0], 1.0)
        for value in spectrum[1:]:
            self.assertLess(value, spectrum[0] * 1e-3)

    def test_isotropic_matrix_approaches_dim(self):
        # Wide-spread isotropic features -> participation ratio close to D.
        torch.manual_seed(0)
        features = torch.randn(4000, 8)

        rank, _ = participation_ratio(features)

        self.assertGreater(rank, 6.5)
        self.assertLessEqual(rank, 8.0 + 1e-6)

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            participation_ratio(torch.randn(4, 4, 4))


class AlignmentTests(unittest.TestCase):
    def test_identical_pairs_align_to_zero(self):
        crops = torch.randn(64, 16)
        self.assertAlmostEqual(alignment(crops, crops), 0.0, places=6)

    def test_distinct_pairs_positive(self):
        torch.manual_seed(1)
        a = torch.randn(64, 16)
        b = torch.randn(64, 16)
        self.assertGreater(alignment(a, b), 0.0)


class UniformityTests(unittest.TestCase):
    def test_collapsed_is_zero(self):
        # All points identical -> all pairwise distances 0 -> uniformity ~ 0.
        single = torch.randn(1, 8)
        collapsed = single.repeat(200, 1)
        self.assertAlmostEqual(uniformity(collapsed), 0.0, places=6)

    def test_spread_is_negative(self):
        torch.manual_seed(2)
        spread = torch.randn(512, 16)
        self.assertLess(uniformity(spread), 0.0)

    def test_single_point_is_zero(self):
        self.assertEqual(uniformity(torch.randn(1, 8)), 0.0)


class NNRetrievalTests(unittest.TestCase):
    def test_identical_pool_is_perfect(self):
        # Each query's twin is the identical pool vector -> R@1 = R@5 = 1.
        torch.manual_seed(3)
        crops = torch.randn(128, 32)
        result = nn_retrieval(crops, crops)
        self.assertAlmostEqual(result["r_at_1"], 1.0, places=6)
        self.assertAlmostEqual(result["r_at_5"], 1.0, places=6)

    def test_shuffled_pool_degrades(self):
        # Shuffling the pool breaks most pairings -> R@1 well below 1.
        torch.manual_seed(4)
        crops = torch.randn(256, 32)
        perm = torch.randperm(256)
        result = nn_retrieval(crops, crops[perm])
        self.assertLess(result["r_at_1"], 0.5)
        # ... but R@5 recovers some.
        self.assertGreaterEqual(result["r_at_5"], result["r_at_1"])

    def test_empty_returns_zeros(self):
        result = nn_retrieval(torch.empty(0, 8), torch.empty(0, 8))
        self.assertEqual(result["r_at_1"], 0.0)
        self.assertEqual(result["r_at_5"], 0.0)

    def test_pool_smaller_than_k(self):
        # Pool size (3) < k=5: R@5 is trivially 1.0; R@1 still meaningful.
        torch.manual_seed(5)
        crops = torch.randn(3, 16)
        result = nn_retrieval(crops, crops)
        self.assertAlmostEqual(result["r_at_1"], 1.0, places=6)
        self.assertAlmostEqual(result["r_at_5"], 1.0, places=6)


class SummarizeTests(unittest.TestCase):
    @staticmethod
    def _collection(bincount, num_points=2000, dim=16, num_events=32):
        torch.manual_seed(0)
        return EmbeddingCollection(
            crop0=torch.randn(num_events, dim),
            crop1=torch.randn(num_events, dim),
            point_subsample=torch.randn(num_points, dim),
            prototype_bincount=bincount,
        )

    def test_uniform_prototype_usage_is_healthy(self):
        k = 4096
        # Uniform usage: each prototype gets the same count -> near-max entropy,
        # effective count ~ K, no dead prototypes.
        bincount = torch.full((k,), 12, dtype=torch.long)
        metrics = summarize(self._collection(bincount))

        self.assertAlmostEqual(metrics["prototype_entropy"], math.log(k), places=4)
        self.assertAlmostEqual(metrics["prototype_effective_count"], k, delta=1.0)
        self.assertEqual(metrics["num_dead_prototypes"], 0)
        self.assertEqual(metrics["num_active_prototypes"], k)
        self.assertEqual(metrics["num_empty_prototypes"], 0)

    def test_collapsed_prototype_usage(self):
        k = 4096
        # All mass on one prototype -> entropy 0, effective count 1, rest dead.
        bincount = torch.zeros(k, dtype=torch.long)
        bincount[0] = 10000
        metrics = summarize(self._collection(bincount))

        # prototype_entropy clamps p to 1e-8, so a collapsed distribution retains
        # a tiny floor (~7e-4 for K=4096), not exactly 0.
        self.assertAlmostEqual(metrics["prototype_entropy"], 0.0, delta=0.01)
        self.assertAlmostEqual(metrics["prototype_effective_count"], 1.0, delta=0.01)
        self.assertEqual(metrics["num_dead_prototypes"], k - 1)
        self.assertEqual(metrics["num_active_prototypes"], 1)
        self.assertEqual(metrics["num_empty_prototypes"], k - 1)

    def test_relative_dead_threshold_not_tripped_by_uniform(self):
        # Regression: an absolute 0.1%-of-pool threshold flagged a uniform 4096
        # prototype space as entirely dead. The relative threshold must not.
        k = 4096
        bincount = torch.full((k,), 12, dtype=torch.long)  # 0.024% of pool each
        metrics = summarize(self._collection(bincount))
        self.assertLess(metrics["num_dead_prototypes"], k)


if __name__ == "__main__":
    unittest.main()
