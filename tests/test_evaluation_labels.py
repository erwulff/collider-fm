import unittest

import numpy as np

from collider_fm.evaluation_labels import (
    PDG_BUCKET_NAMES,
    compute_dominance_report,
    dominant_particle,
    dominance_report,
    event_dominant_particles,
    format_dominance_report,
    pdg_bucket,
)


class DominantParticleTests(unittest.TestCase):
    def test_single_contributor_fraction_one(self):
        # One contributor -> it dominates with fraction 1.0, count 1.
        ids = [[7], [12], [3]]
        ens = [[0.5], [2.0], [0.1]]
        dom_pid, dom_frac, counts = dominant_particle(ids, ens)

        np.testing.assert_array_equal(dom_pid, [7, 12, 3])
        np.testing.assert_allclose(dom_frac, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(counts, [1, 1, 1])

    def test_multi_contributor_argmax(self):
        # The largest-energy contributor wins; fraction is its share of the total.
        ids = [[1, 2, 3]]
        ens = [[1.0, 3.0, 1.0]]  # total 5.0, winner index 1 (pid 2) with 3/5
        dom_pid, dom_frac, counts = dominant_particle(ids, ens)

        self.assertEqual(int(dom_pid[0]), 2)
        self.assertAlmostEqual(float(dom_frac[0]), 0.6, places=5)
        self.assertEqual(int(counts[0]), 3)

    def test_zero_contributors(self):
        # No contributors -> particle_id -1, fraction 0, count 0.
        dom_pid, dom_frac, counts = dominant_particle([[]], [[]])

        self.assertEqual(int(dom_pid[0]), -1)
        self.assertAlmostEqual(float(dom_frac[0]), 0.0)
        self.assertEqual(int(counts[0]), 0)

    def test_zero_total_energy_falls_back_to_first(self):
        # All-zero energies -> total 0 -> fall back to the first contributor (frac 0).
        ids = [[5, 6]]
        ens = [[0.0, 0.0]]
        dom_pid, dom_frac, counts = dominant_particle(ids, ens)

        self.assertEqual(int(dom_pid[0]), 5)
        self.assertAlmostEqual(float(dom_frac[0]), 0.0)
        self.assertEqual(int(counts[0]), 2)

    def test_positional_pairing(self):
        # contrib_energies[i][j] pairs with contrib_particle_ids[i][j] per cell.
        ids = [[10, 20], [30]]
        ens = [[0.1, 0.9], [1.0]]
        dom_pid, dom_frac, counts = dominant_particle(ids, ens)

        np.testing.assert_array_equal(dom_pid, [20, 30])
        np.testing.assert_allclose(dom_frac, [0.9, 1.0])
        np.testing.assert_array_equal(counts, [2, 1])


class DominanceReportTests(unittest.TestCase):
    def test_mixed_distribution(self):
        # 4 single-contributor hits (frac 1.0) + 1 shared hit (frac 0.6).
        counts = np.array([1, 1, 1, 1, 2])
        frac = np.array([1.0, 1.0, 1.0, 1.0, 0.6])

        report = dominance_report(counts, frac)

        self.assertEqual(report["num_hits"], 5)
        self.assertAlmostEqual(report["pct_single_contributor"], 80.0)
        self.assertAlmostEqual(report["pct_shared"], 20.0)
        self.assertAlmostEqual(report["pct_dominant_ge_0.9"], 80.0)  # the four 1.0s
        self.assertAlmostEqual(report["pct_dominant_ge_0.5"], 100.0)  # 0.6 >= 0.5
        # shared-only stats cover the single shared hit (frac 0.6).
        self.assertEqual(report["shared_num_hits"], 1)
        self.assertAlmostEqual(report["shared_dominant_frac_median"], 0.6)

    def test_all_single_contributor_no_shared_block(self):
        # No shared hits -> the shared_* keys are omitted.
        counts = np.array([1, 1, 1])
        frac = np.array([1.0, 1.0, 1.0])

        report = dominance_report(counts, frac)

        self.assertAlmostEqual(report["pct_single_contributor"], 100.0)
        self.assertAlmostEqual(report["pct_shared"], 0.0)
        self.assertNotIn("shared_num_hits", report)

    def test_contributor_count_summary(self):
        counts = np.array([1, 1, 2, 3, 10])
        frac = np.array([1.0, 1.0, 0.7, 0.5, 0.4])

        report = dominance_report(counts, frac)

        self.assertEqual(report["contributor_count_max"], 10)
        self.assertAlmostEqual(report["contributor_count_mean"], 3.4)
        self.assertEqual(report["contributor_count_median"], 2.0)


class FormatDominanceReportTests(unittest.TestCase):
    def test_renders_all_sections(self):
        # A report with the shared-only block -> text has header + all sections.
        report = {
            "num_events_scanned": 5,
            "num_hits": 100,
            "contributor_count_mean": 1.5,
            "contributor_count_median": 1.0,
            "contributor_count_p99": 4.0,
            "contributor_count_max": 5,
            "pct_single_contributor": 80.0,
            "pct_shared": 20.0,
            "dominant_frac_mean": 0.9,
            "dominant_frac_median": 1.0,
            "pct_dominant_ge_0.9": 80.0,
            "pct_dominant_ge_0.5": 100.0,
            "shared_num_hits": 20,
            "shared_dominant_frac_median": 0.6,
            "shared_dominant_frac_mean": 0.55,
            "shared_pct_dominant_ge_0.9": 50.0,
            "shared_pct_dominant_ge_0.5": 100.0,
        }
        text = format_dominance_report(report)
        self.assertIn("Dominance report", text)
        self.assertIn("events scanned : 5", text)
        self.assertIn("single contributor : 80.0%", text)
        self.assertIn(">=0.9  : 80.0%", text)
        self.assertIn("shared-only (20 hits)", text)
        self.assertTrue(text.endswith("\n"))

    def test_omits_shared_block_when_absent(self):
        # No shared_* keys -> no shared-only section.
        report = {
            "num_events_scanned": 1, "num_hits": 3,
            "pct_single_contributor": 100.0, "pct_shared": 0.0,
        }
        text = format_dominance_report(report)
        self.assertNotIn("shared-only", text)


class ComputeDominanceReportTests(unittest.TestCase):
    def test_aggregates_across_events_and_caps(self):
        # Two synthetic events; max_events caps the scan to one.
        events = [
            {
                "contrib_particle_ids": [[1], [2, 3]],
                "contrib_energies": [[1.0], [0.3, 0.7]],
            },
            {
                "contrib_particle_ids": [[4], [5]],
                "contrib_energies": [[1.0], [1.0]],
            },
        ]

        report, n = compute_dominance_report(events, max_events=1)

        self.assertEqual(n, 1)
        self.assertEqual(report["num_hits"], 2)  # only the first event's 2 cells
        self.assertAlmostEqual(report["pct_single_contributor"], 50.0)

    def test_no_cap_scans_all(self):
        events = [
            {"contrib_particle_ids": [[1], [2, 3]], "contrib_energies": [[1.0], [0.3, 0.7]]},
            {"contrib_particle_ids": [[4], [5]], "contrib_energies": [[1.0], [1.0]]},
        ]

        report, n = compute_dominance_report(events)

        self.assertEqual(n, 2)
        self.assertEqual(report["num_hits"], 4)


class EventDominantParticlesTests(unittest.TestCase):
    def test_dominant_and_pdg_join(self):
        # Two cells: cell 0 dominated by pid 2 (pdg 211), cell 1 single pid 5 (pdg 13).
        event = {
            "contrib_particle_ids": [[1, 2], [5]],
            "contrib_energies": [[0.3, 0.7], [1.0]],
        }
        pid_to_pdg = {2: 211, 5: 13, 1: 11}

        dom_pid, pdg, dom_frac, counts = event_dominant_particles(event, pid_to_pdg)

        np.testing.assert_array_equal(dom_pid, [2, 5])
        np.testing.assert_array_equal(pdg, [211, 13])
        np.testing.assert_allclose(dom_frac, [0.7, 1.0])
        np.testing.assert_array_equal(counts, [2, 1])

    def test_unknown_pdg_is_minus_one(self):
        # Dominant particle not in the map -> pdg_id -1.
        event = {"contrib_particle_ids": [[9]], "contrib_energies": [[1.0]]}

        _, pdg, _, _ = event_dominant_particles(event, {1: 11})

        self.assertEqual(int(pdg[0]), -1)

    def test_no_pid_map_all_minus_one(self):
        event = {"contrib_particle_ids": [[1], [2, 3]], "contrib_energies": [[1.0], [0.4, 0.6]]}

        _, pdg, _, _ = event_dominant_particles(event, None)

        np.testing.assert_array_equal(pdg, [-1, -1])


class PdgBucketTests(unittest.TestCase):
    def test_known_types_map_to_their_bucket(self):
        # One representative per bucket (incl. antiparticles); rare codes -> other.
        pdgs = [
            11, -11, 22, 13, -13,                       # e±, γ, μ±
            211, -211, 321, -321, 2212, -2212,          # charged hadron
            111, 2112, -2112, 130, 310, 311, -311, 3122, -3122,  # neutral hadron
            1000010020, 1000140280,                     # nuclei
            999, -541, 0,                               # other (rare / unlisted)
        ]
        out = pdg_bucket(pdgs)
        expected = (
            [0, 0, 1, 2, 2]
            + [3, 3, 3, 3, 3, 3]
            + [4, 4, 4, 4, 4, 4, 4, 4, 4]
            + [5, 5]
            + [6, 6, 6]
        )
        np.testing.assert_array_equal(out, expected)

    def test_unknown_sentinel_is_minus_one(self):
        # -1 (missing pid) -> -1 so downstream -1-dropping treats it as "no label".
        np.testing.assert_array_equal(pdg_bucket([-1, 11, -1]), [-1, 0, -1])

    def test_preserves_shape(self):
        out = pdg_bucket(np.array([[11, 22], [211, -1]]))
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_array_equal(out, [[0, 1], [3, -1]])

    def test_bucket_names_match_expected_roles(self):
        self.assertEqual(
            PDG_BUCKET_NAMES,
            ["e±", "γ", "μ±", "charged hadron", "neutral hadron", "nucleus", "other"],
        )


if __name__ == "__main__":
    unittest.main()
