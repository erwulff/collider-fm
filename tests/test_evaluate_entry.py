"""Tests for the reusable eval entry point and the post-training eval hook.

Covers the decision logic and the metrics/training decoupling -- not a real eval run,
which needs a GPU and the dataset.
"""

import inspect
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from collider_fm.metrics import embedding_norm, feature_std, prototype_entropy, prototype_usage


class EvaluationImportDecouplingTests(unittest.TestCase):
    def test_evaluation_does_not_import_ray_or_matplotlib(self):
        # collider_fm.evaluation used to reach prototype_entropy via training_loop,
        # dragging in ray + matplotlib (~21s import) for a pure-torch helper. The
        # helpers now live in metrics.py; lock that in so it cannot regress.
        code = "import sys; import collider_fm.evaluation; " "print('ray' in sys.modules, 'matplotlib' in sys.modules)"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("False False", proc.stdout)


class MovedMetricHelperTests(unittest.TestCase):
    """The four helpers moved from training_loop.py must behave identically."""

    def test_prototype_usage_normalizes(self):
        # One-hot logits over 4 prototypes: each row's argmax is its own prototype.
        logits = torch.eye(4)
        usage = prototype_usage(logits, num_prototypes=4)
        self.assertEqual(usage.shape, (4,))
        self.assertAlmostEqual(float(usage.sum()), 1.0, places=6)

    def test_prototype_usage_counts_unused_prototypes(self):
        # All rows pick prototype 0 -> usage is one-hot at 0, zeros elsewhere.
        logits = torch.tensor([[5.0, 0.0, 0.0], [9.0, 1.0, 2.0]])
        usage = prototype_usage(logits, num_prototypes=3)
        self.assertAlmostEqual(float(usage[0]), 1.0, places=6)
        self.assertAlmostEqual(float(usage[1:].sum()), 0.0, places=6)

    def test_prototype_entropy_uniform_is_log_k(self):
        k = 8
        self.assertAlmostEqual(prototype_entropy(torch.full((k,), 1.0 / k)), math.log(k), places=5)

    def test_prototype_entropy_collapsed_is_zero(self):
        collapsed = torch.zeros(8)
        collapsed[0] = 1.0
        self.assertAlmostEqual(prototype_entropy(collapsed), 0.0, delta=0.01)

    def test_embedding_norm_and_feature_std_handle_empty(self):
        # The guard clause that makes these safe to call inside run_epoch.
        for fn in (embedding_norm, feature_std):
            self.assertEqual(fn(None), 0.0)
            self.assertEqual(fn(torch.empty(0, 4)), 0.0)

    def test_embedding_norm_unit_vectors(self):
        # Rows of norm 1 -> mean norm 1.
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(embedding_norm(embeddings), 1.0, places=6)


class RunEvaluationEntryPointTests(unittest.TestCase):
    def test_run_evaluation_takes_a_config(self):
        # Guards the main()->run_evaluation extraction against drift.
        import evaluate

        params = list(inspect.signature(evaluate.run_evaluation).parameters)
        self.assertEqual(params, ["config"])


class ShouldEvalAfterTrainingTests(unittest.TestCase):
    def _config(self, **training_overrides):
        base = {
            "eval_after_training": True,
            "max_train_batches": None,
            "max_val_batches": None,
        }
        base.update(training_overrides)
        return OmegaConf.create({"training": base})

    def test_runs_for_a_normal_completed_run(self):
        import train

        should, reason = train.should_eval_after_training(self._config(), has_checkpoint=True)
        self.assertTrue(should)
        self.assertEqual(reason, "")

    def test_skips_when_flag_disabled(self):
        import train

        should, reason = train.should_eval_after_training(self._config(eval_after_training=False), has_checkpoint=True)
        self.assertFalse(should)
        self.assertIn("eval_after_training", reason)

    def test_skips_without_checkpoint(self):
        import train

        should, reason = train.should_eval_after_training(self._config(), has_checkpoint=False)
        self.assertFalse(should)
        self.assertIn("no checkpoint", reason)

    def test_skips_for_batch_limited_debug_runs(self):
        import train

        for key in ("max_train_batches", "max_val_batches"):
            should, reason = train.should_eval_after_training(self._config(**{key: 20}), has_checkpoint=True)
            self.assertFalse(should, msg=f"{key} should force a skip")
            self.assertIn(key, reason)


class BuildEvalConfigTests(unittest.TestCase):
    def _config(self):
        return OmegaConf.create(
            {
                "training": {"eval_after_training": True},
                "evaluation": {
                    "checkpoint": None,
                    "run_name": "stale",
                    "enable_tsne": False,
                    "tsne_upcast2": False,
                },
            }
        )

    def test_points_checkpoint_at_run_dir_and_enables_plots(self):
        import train

        run_dir = Path("/tmp/runs/myrun")
        built = train.build_eval_config(self._config(), run_dir, cli_overrides=[])
        self.assertEqual(built.evaluation.checkpoint, str(run_dir))
        self.assertTrue(built.evaluation.enable_tsne)
        self.assertTrue(built.evaluation.tsne_upcast2)
        # run_name cleared so output nests at runs/<run>/eval/<checkpoint>/
        self.assertIsNone(built.evaluation.run_name)

    def test_respects_explicit_cli_optout(self):
        import train

        built = train.build_eval_config(
            self._config(),
            Path("/tmp/runs/myrun"),
            cli_overrides=["evaluation.enable_tsne=false"],
        )
        self.assertFalse(built.evaluation.enable_tsne)

    def test_does_not_mutate_the_input_config(self):
        import train

        config = self._config()
        train.build_eval_config(config, Path("/tmp/runs/myrun"), cli_overrides=[])
        self.assertIsNone(config.evaluation.checkpoint)
        self.assertFalse(config.evaluation.enable_tsne)


if __name__ == "__main__":
    unittest.main()
