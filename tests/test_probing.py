import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from collider_fm.evaluation_labels import PDG_BUCKET_NAMES
from collider_fm.probing import (
    ENERGY_PROBE_CLASSES,
    ProbeDataCollection,
    _classification_metrics,
    collect_raw_input_probe_data,
    event_class_energies,
    format_probes_report,
    load_probe_features,
    plot_confusion_matrix,
    plot_energy_scatter,
    plot_loss_curves,
    probe_features_signature,
    save_probe_features,
    train_energy_probe,
    train_segmentation_probe,
)


class EventClassEnergiesTests(unittest.TestCase):
    def test_sums_energies_by_bucket(self):
        # Hit 0: photon (22, 2.0) + neutral hadron (2112, 3.0). Hit 1: charged hadrons
        # (211, 1.0; -211, 4.0). Hit 2: electron (11, 5.0, ignored) + unknown pid (0.5,
        # ignored). Expected order follows ENERGY_PROBE_CLASSES.
        event = {
            "contrib_particle_ids": [[1, 2], [3, 4], [5, 6]],
            "contrib_energies": [[2.0, 3.0], [1.0, 4.0], [5.0, 0.5]],
        }
        pid_to_pdg = {1: 22, 2: 2112, 3: 211, 4: -211, 5: 11}

        out = event_class_energies(event, pid_to_pdg)

        expected = {"neutral_hadron": 3.0, "photon": 2.0, "charged_hadron": 5.0}
        for k, (name, _) in enumerate(ENERGY_PROBE_CLASSES):
            self.assertAlmostEqual(out[k], expected[name])

    def test_empty_event_is_zeros(self):
        event = {"contrib_particle_ids": [], "contrib_energies": []}
        out = event_class_energies(event, {1: 22})
        np.testing.assert_array_equal(out, np.zeros(len(ENERGY_PROBE_CLASSES)))

    def test_none_pid_map_is_zeros(self):
        event = {"contrib_particle_ids": [[1]], "contrib_energies": [[2.0]]}
        out = event_class_energies(event, None)
        np.testing.assert_array_equal(out, np.zeros(len(ENERGY_PROBE_CLASSES)))


class ClassificationMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        target = np.array([0, 0, 1, 1, 2])
        metrics = _classification_metrics(target, target, ["a", "b", "c", "d"])

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["mean_iou"], 1.0)
        # Absent class "d" contributes zero support and is excluded from the macros.
        self.assertEqual(metrics["per_class"]["d"]["support"], 0)

    def test_absent_class_excluded_from_macros(self):
        # Class 0 perfectly predicted, class 1 never predicted (f1=0), class 2 absent.
        target = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 0, 0])
        metrics = _classification_metrics(pred, target, ["a", "b", "c"])

        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        # macro over present classes {a, b}: (f1_a + 0) / 2.
        f1_a = metrics["per_class"]["a"]["f1"]
        self.assertAlmostEqual(metrics["macro_f1"], f1_a / 2.0)


def _separable_collection(num_points, num_events, dim=8, num_classes=3, seed=0, unknown_fraction=0.0):
    """Synthetic ProbeDataCollection: per-point classes are well-separated feature
    clusters; per-event targets are an exact linear map of the event features (in
    log1p space), so both probes should score near-perfectly."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_points,), generator=generator)
    centers = torch.eye(num_classes, dim) * 10.0
    point_features = centers[labels] + 0.1 * torch.randn(num_points, dim, generator=generator)
    if unknown_fraction > 0:
        n_unknown = int(num_points * unknown_fraction)
        labels[:n_unknown] = -1

    event_features = torch.randn(num_events, dim, generator=generator)
    # Fixed-seed weight so train/val collections share the same feature->target map.
    weight_generator = torch.Generator().manual_seed(42)
    weight = torch.randn(dim, len(ENERGY_PROBE_CLASSES), generator=weight_generator) * 0.3
    target_log = (event_features @ weight + 2.0).clamp_min(0.0)
    event_targets = torch.expm1(target_log)
    return ProbeDataCollection(
        point_features=point_features,
        point_labels=labels,
        event_features=event_features,
        event_targets=event_targets,
        num_events=num_events,
    )


class TrainSegmentationProbeTests(unittest.TestCase):
    def test_separable_features_score_high(self):
        train = _separable_collection(600, 4, seed=0)
        val = _separable_collection(300, 4, seed=1)

        metrics = train_segmentation_probe(train, val, epochs=30, batch_size=128, lr=1e-2, device=torch.device("cpu"), seed=0)

        self.assertGreater(metrics["val"]["accuracy"], 0.95)
        self.assertGreater(metrics["val"]["macro_f1"], 0.95)
        self.assertGreater(metrics["val"]["mean_iou"], 0.9)
        self.assertEqual(set(metrics["val"]["per_class"].keys()), set(PDG_BUCKET_NAMES))

    def test_unknown_labels_dropped(self):
        train = _separable_collection(600, 4, seed=0, unknown_fraction=0.25)
        val = _separable_collection(300, 4, seed=1, unknown_fraction=0.25)

        metrics = train_segmentation_probe(train, val, epochs=5, batch_size=128, device=torch.device("cpu"), seed=0)

        self.assertEqual(metrics["num_train_points"], int((train.point_labels >= 0).sum()))
        self.assertEqual(metrics["num_val_points"], int((val.point_labels >= 0).sum()))

    def test_raises_without_labeled_points(self):
        train = _separable_collection(50, 2, seed=0, unknown_fraction=1.0)
        val = _separable_collection(50, 2, seed=1)
        with self.assertRaises(ValueError):
            train_segmentation_probe(train, val, epochs=1, device=torch.device("cpu"))

    def test_separate_calls_do_not_share_rng(self):
        # Each fit must init from its own seed (not the ambient global RNG).
        train = _separable_collection(600, 4, seed=0)
        val = _separable_collection(300, 4, seed=1)
        a = train_segmentation_probe(train, val, epochs=1, batch_size=128, lr=1e-2, device=torch.device("cpu"), seed=0)
        b = train_segmentation_probe(train, val, epochs=1, batch_size=128, lr=1e-2, device=torch.device("cpu"), seed=0)
        self.assertAlmostEqual(a["final_train_loss"], b["final_train_loss"], places=6)


class ProbeArtifactsTests(unittest.TestCase):
    """The extra data the artifacts need: epoch losses, confusion counts, scatter tensors."""

    @classmethod
    def setUpClass(cls):
        cls.train = _separable_collection(300, 30, seed=0)
        cls.val = _separable_collection(150, 20, seed=1)
        cls.seg = train_segmentation_probe(cls.train, cls.val, epochs=3, batch_size=128, lr=1e-2, device=torch.device("cpu"), seed=0)
        cls.energy = train_energy_probe(cls.train, cls.val, epochs=5, batch_size=64, lr=1e-2, device=torch.device("cpu"), seed=0)

    def test_epoch_losses_match_epochs(self):
        self.assertEqual(len(self.seg["epoch_losses"]), 3)
        self.assertEqual(len(self.energy["epoch_losses"]), 5)
        self.assertAlmostEqual(self.seg["final_train_loss"], self.seg["epoch_losses"][-1])
        # Val-loss curves are recorded per epoch alongside the train curves.
        self.assertEqual(len(self.seg["epoch_val_losses"]), 3)
        self.assertEqual(len(self.energy["epoch_val_losses"]), 5)
        self.assertAlmostEqual(self.seg["final_val_loss"], self.seg["epoch_val_losses"][-1])
        self.assertAlmostEqual(self.energy["final_val_loss"], self.energy["epoch_val_losses"][-1])

    def test_confusion_rows_sum_to_support(self):
        confusion = self.seg["val"]["confusion"]
        self.assertEqual(len(confusion), len(PDG_BUCKET_NAMES))
        for i, name in enumerate(PDG_BUCKET_NAMES):
            self.assertEqual(sum(confusion[i]), self.seg["val"]["per_class"][name]["support"])
        self.assertEqual(sum(sum(row) for row in confusion), self.seg["num_val_points"])

    def test_energy_scatter_shapes_and_targets(self):
        scatter = self.energy["scatter"]
        self.assertEqual(scatter["val_pred"].shape, (self.energy["num_val_events"], len(ENERGY_PROBE_CLASSES)))
        self.assertEqual(scatter["train_target"].shape, (self.energy["num_train_events"], len(ENERGY_PROBE_CLASSES)))
        # Targets echo the collection's event targets verbatim.
        self.assertTrue(torch.equal(scatter["val_target"], self.val.event_targets.float()))

    def test_plots_write_pngs(self):
        with TemporaryDirectory() as tmp:
            plot_confusion_matrix(self.seg["train"]["confusion"], self.seg["val"]["confusion"], PDG_BUCKET_NAMES, Path(tmp) / "confusion.png")
            plot_loss_curves(
                self.seg["epoch_losses"],
                self.energy["epoch_losses"],
                Path(tmp) / "losses.png",
                seg_val_losses=self.seg["epoch_val_losses"],
                energy_val_losses=self.energy["epoch_val_losses"],
            )
            plot_energy_scatter(self.energy["scatter"], self.energy, Path(tmp) / "scatter.png")
            for name in ("confusion.png", "losses.png", "scatter.png"):
                path = Path(tmp) / name
                self.assertTrue(path.exists(), name)
                self.assertGreater(path.stat().st_size, 0)

    def test_report_with_and_without_baseline(self):
        report = format_probes_report(self.seg_and_energy(), None, main_label="trained")
        self.assertIn("Semantic segmentation", report)
        self.assertIn("none", report)  # no baseline
        baseline = self.seg_and_energy()
        report = format_probes_report(self.seg_and_energy(), baseline, main_label="trained")
        self.assertIn("random init", report)
        self.assertIn("Probe convergence", report)
        for name in PDG_BUCKET_NAMES:
            self.assertIn(name, report)
        report = format_probes_report(self.seg_and_energy(), baseline, raw_metrics=baseline, main_label="trained")
        self.assertIn("raw input", report)

    @staticmethod
    def seg_and_energy():
        # _write (and format_probes_report) expect the two-probe dict with scatter popped.
        seg = {k: v for k, v in ProbeArtifactsTests.seg.items() if k != "scatter"}
        energy = {k: v for k, v in ProbeArtifactsTests.energy.items() if k != "scatter"}
        return {"semantic_segmentation": seg, "energy": energy}


class TrainEnergyProbeTests(unittest.TestCase):
    def test_linear_targets_recovered(self):
        train = _separable_collection(10, 400, seed=0)
        val = _separable_collection(10, 200, seed=1)

        metrics = train_energy_probe(train, val, epochs=400, batch_size=400, lr=1e-2, device=torch.device("cpu"), seed=0)

        self.assertGreater(metrics["val"]["mean_r2"], 0.9)
        for name, _ in ENERGY_PROBE_CLASSES:
            self.assertGreater(metrics["val"]["per_class"][name]["r2"], 0.85)
            self.assertGreaterEqual(metrics["val"]["per_class"][name]["mae"], 0.0)

    def test_constant_target_r2_is_zero(self):
        train = _separable_collection(10, 100, seed=0)
        val = _separable_collection(10, 50, seed=1)
        train.event_targets = torch.full_like(train.event_targets, 5.0)
        val.event_targets = torch.full_like(val.event_targets, 5.0)

        metrics = train_energy_probe(train, val, epochs=10, batch_size=100, device=torch.device("cpu"), seed=0)

        for name, _ in ENERGY_PROBE_CLASSES:
            self.assertEqual(metrics["val"]["per_class"][name]["r2"], 0.0)

    def test_raises_with_too_few_events(self):
        train = _separable_collection(10, 1, seed=0)
        val = _separable_collection(10, 50, seed=1)
        with self.assertRaises(ValueError):
            train_energy_probe(train, val, epochs=1, device=torch.device("cpu"))


class ProbeFeatureCacheTests(unittest.TestCase):
    def test_save_load_round_trip(self):
        # Save -> load should reproduce the collection exactly.
        collection = _separable_collection(num_points=40, num_events=4, seed=0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.pt"
            save_probe_features(collection, path)
            loaded = load_probe_features(path)
        self.assertEqual(loaded.num_events, collection.num_events)
        self.assertTrue(torch.equal(loaded.point_features, collection.point_features))
        self.assertTrue(torch.equal(loaded.point_labels, collection.point_labels))
        self.assertTrue(torch.equal(loaded.event_features, collection.event_features))
        self.assertTrue(torch.equal(loaded.event_targets, collection.event_targets))

    def test_signature_changes_with_any_input(self):
        # Two collections that differ in any cached attribute must hash differently.
        kwargs = {"grid_size": 0.002, "max_calo_hits": None}
        base = probe_features_signature(kwargs, max_events=10, max_points=1000, seed=0, num_prototypes=4096)
        self.assertNotEqual(base, probe_features_signature(kwargs, max_events=11, max_points=1000, seed=0, num_prototypes=4096))
        self.assertNotEqual(base, probe_features_signature(kwargs, max_events=10, max_points=1001, seed=0, num_prototypes=4096))
        self.assertNotEqual(base, probe_features_signature(kwargs, max_events=10, max_points=1000, seed=1, num_prototypes=4096))
        self.assertNotEqual(base, probe_features_signature({**kwargs, "grid_size": 0.004}, max_events=10, max_points=1000, seed=0, num_prototypes=4096))
        self.assertNotEqual(base, probe_features_signature(kwargs, max_events=10, max_points=1000, seed=0, num_prototypes=8192))
        # Same inputs -> same signature
        self.assertEqual(base, probe_features_signature(kwargs, max_events=10, max_points=1000, seed=0, num_prototypes=4096))


class CollectRawInputProbeDataTests(unittest.TestCase):
    """The backbone-free collector builds views on CPU with synthetic events."""

    def _fake_events(self, num_events: int = 3, num_hits: int = 200) -> list[dict]:
        rng = np.random.default_rng(0)
        events = []
        for _ in range(num_events):
            pids = rng.choice([22, 2112, 211], size=num_hits)
            events.append(
                {
                    # Normalized coords like the real dataset (raw mm / 5000).
                    "x": rng.uniform(-0.5, 0.5, num_hits),
                    "y": rng.uniform(-0.5, 0.5, num_hits),
                    "z": rng.uniform(-0.5, 0.5, num_hits),
                    "total_energy": rng.uniform(0.1, 5.0, num_hits),
                    "contrib_particle_ids": [[int(p)] for p in rng.integers(1, 10, num_hits)],
                    "contrib_energies": [[float(e)] for e in rng.uniform(0.1, 1.0, num_hits)],
                    "pids": pids,  # for the pid map below
                }
            )
        return events

    def test_collects_views_labels_and_targets(self):
        from collider_fm.project_config import point_view_kwargs, load_project_config

        events = self._fake_events()
        pid_to_pdg = {i: pdg for i, pdg in enumerate([22, 2112, 211, -11, 13, 2212, 211, 22, 2112], start=1)}

        config = load_project_config()
        view_kwargs = point_view_kwargs(config, "training", max_calo_hits=None)
        collection = collect_raw_input_probe_data(
            events,
            pid_to_pdg,
            torch.device("cpu"),
            view_kwargs=view_kwargs,
            max_events=3,
            max_points=1000,
            seed=0,
        )

        self.assertEqual(collection.num_events, 3)
        self.assertEqual(collection.event_features.shape[0], 3)
        self.assertEqual(collection.event_targets.shape, (3, len(ENERGY_PROBE_CLASSES)))
        self.assertGreater(collection.point_features.shape[0], 0)
        self.assertEqual(collection.point_features.shape[0], collection.point_labels.shape[0])
        # Features are the standardized base-view inputs ({x, y, z, E} = 4 channels).
        self.assertEqual(collection.point_features.shape[1], collection.event_features.shape[1])
        # All subsampled labels must be pid buckets (0..6) from the fake map above.
        self.assertTrue(bool(((collection.point_labels >= 0) & (collection.point_labels < len(PDG_BUCKET_NAMES))).all()))


if __name__ == "__main__":
    unittest.main()
