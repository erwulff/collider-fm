import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_training_run.py"
SPEC = importlib.util.spec_from_file_location("plot_training_run", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
plot_training_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_training_run)


class PlotTrainingRunTests(unittest.TestCase):
    def test_metric_series_uses_requested_x_key(self):
        records = [
            {"epoch": 1, "step": 5, "learning_rate": 1.0e-4},
            {"epoch": 1, "step": 10, "learning_rate": 8.0e-5},
        ]

        xs, ys = plot_training_run.metric_series(
            records,
            "learning_rate",
            x_key="step",
        )

        self.assertEqual(xs, [5.0, 10.0])
        self.assertEqual(ys, [1.0e-4, 8.0e-5])

    def test_metric_gap_uses_requested_x_key(self):
        records = [
            {"epoch": 1, "step": 5, "train_loss": 1.5, "val_loss": 2.0},
            {"epoch": 2, "step": 10, "train_loss": 1.0, "val_loss": 1.2},
        ]

        xs, ys = plot_training_run.metric_gap(
            records,
            "train_loss",
            "val_loss",
            x_key="step",
        )

        self.assertEqual(xs, [5.0, 10.0])
        self.assertAlmostEqual(ys[0], 0.5)
        self.assertAlmostEqual(ys[1], 0.2)


if __name__ == "__main__":
    unittest.main()
