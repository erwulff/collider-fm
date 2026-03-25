import json
import tempfile
import unittest
import sys
import importlib.util
from pathlib import Path


PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_training_run_script",
    Path(__file__).resolve().parents[1] / "scripts" / "plot_training_run.py",
)
if PLOT_SPEC is None or PLOT_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/plot_training_run.py for testing.")
plot_training_run = importlib.util.module_from_spec(PLOT_SPEC)
sys.modules[PLOT_SPEC.name] = plot_training_run
PLOT_SPEC.loader.exec_module(plot_training_run)

read_json = plot_training_run.read_json
read_metrics = plot_training_run.read_metrics
keep_last_run = plot_training_run.keep_last_run


class PlotTrainingRunTests(unittest.TestCase):
    def test_read_metrics_loads_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            path.write_text(
                '{"epoch": 1, "train_loss": 1.2}\n{"epoch": 2, "train_loss": 0.8}\n',
                encoding="utf-8",
            )
            records = read_metrics(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["epoch"], 1)
        self.assertEqual(records[1]["train_loss"], 0.8)

    def test_read_json_loads_dictionary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = {"run_name": "demo", "num_epochs": 3}
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_json(path)

        self.assertEqual(loaded, payload)

    def test_keep_last_run_discards_older_restart_records(self):
        records = [
            {"epoch": 1, "global_step": 100},
            {"epoch": 2, "global_step": 200},
            {"epoch": 1, "global_step": 100},
            {"epoch": 2, "global_step": 200},
            {"epoch": 3, "global_step": 300},
        ]

        kept = keep_last_run(records)
        self.assertEqual([record["epoch"] for record in kept], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
