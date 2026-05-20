import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collider_fm.experiment_logging import (
    CompositeLogger,
    comet_config_path,
    comet_is_configured,
    create_experiment_logger,
    write_run_config,
)


class FakeExperiment:
    def __init__(self):
        self.name = None
        self.params = None
        self.metrics = []
        self.finished = False

    def set_name(self, name):
        self.name = name

    def log_parameters(self, params):
        self.params = params

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def end(self):
        self.finished = True


class ExperimentLoggingTests(unittest.TestCase):
    def test_write_run_config_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            path = write_run_config(run_dir, {"alpha": 1, "beta": "two"})

            payload = json.loads(path.read_text())
            self.assertEqual(payload["alpha"], 1)
            self.assertEqual(payload["beta"], "two")

    def test_jsonl_logger_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_experiment_logger(
                "jsonl", Path(tmpdir)
            )
            logger.log_metrics({"epoch": 1, "train_loss": 1.23}, step=4)
            logger.finish()

            records = [
                json.loads(line)
                for line in (Path(tmpdir) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records, [{"epoch": 1, "step": 4, "train_loss": 1.23}])

    def test_auto_logger_combines_jsonl_and_comet_when_configured(self):
        fake_experiment = FakeExperiment()

        def fake_factory(**kwargs):
            self.assertEqual(kwargs["api_key"], "secret")
            self.assertEqual(kwargs["project_name"], "proj")
            self.assertEqual(kwargs["workspace"], "team")
            self.assertEqual(kwargs["run_name"], "demo-run")
            fake_experiment.set_name(kwargs["run_name"])
            return fake_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "demo-run"
            run_dir.mkdir()
            logger = create_experiment_logger(
                "auto",
                run_dir,
                env={
                    "COMET_API_KEY": "secret",
                    "COMET_PROJECT_NAME": "proj",
                    "COMET_WORKSPACE": "team",
                },
                experiment_factory=fake_factory,
            )
            self.assertIsInstance(logger, CompositeLogger)

            logger.log_params({"batch_size": 1})
            logger.log_metrics({"epoch": 1, "val_loss": 0.5}, step=2)
            logger.finish()

            records = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records, [{"epoch": 1, "step": 2, "val_loss": 0.5}])
            self.assertEqual(fake_experiment.params, {"batch_size": 1})
            self.assertEqual(
                fake_experiment.metrics, [({"epoch": 1, "val_loss": 0.5}, 2)]
            )
            self.assertTrue(fake_experiment.finished)
            self.assertEqual(fake_experiment.name, "demo-run")

    def test_comet_is_configured_by_saved_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            config_path = comet_config_path(home)
            config_path.write_text("[comet]\napi_key=dummy\n")

            self.assertTrue(comet_is_configured(env={}, home_dir=home))

    def test_comet_backend_requires_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "~/.comet.config"):
                create_experiment_logger(
                    "comet",
                    Path(tmpdir),
                    env={},
                    home_dir=Path(tmpdir) / "missing-home",
                )

    def test_comet_backend_uses_saved_config_without_env_api_key(self):
        fake_experiment = FakeExperiment()

        def fake_factory(**kwargs):
            self.assertIsNone(kwargs["api_key"])
            self.assertEqual(kwargs["project_name"], "collider-fm")
            self.assertIsNone(kwargs["workspace"])
            self.assertEqual(kwargs["run_name"], "demo-run")
            return fake_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir()
            run_dir = Path(tmpdir) / "demo-run"
            run_dir.mkdir()
            comet_config_path(home).write_text("[comet]\napi_key=dummy\n")
            logger = create_experiment_logger(
                "comet",
                run_dir,
                env={},
                experiment_factory=fake_factory,
                home_dir=home,
            )

            logger.log_metrics({"epoch": 1}, step=1)
            logger.finish()

            self.assertEqual(fake_experiment.metrics, [({"epoch": 1}, 1)])
            self.assertTrue(fake_experiment.finished)


if __name__ == "__main__":
    unittest.main()
