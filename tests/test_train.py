import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from collider_fm.project_config import resolve_run_identity, resolve_run_lifecycle


class TrainLifecycleTests(unittest.TestCase):
    def test_resolve_run_identity_generates_default_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("collider_fm.project_config.timestamp_suffix", return_value="20260326_123456"):
                run_dir, run_name = resolve_run_identity(root)

            self.assertEqual(run_name, "run_20260326_123456")
            self.assertEqual(run_dir, root / "runs" / "run_20260326_123456")

    def test_resolve_run_identity_uses_run_dir_name_when_run_name_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            explicit = root / "custom-location"

            run_dir, run_name = resolve_run_identity(
                root, run_dir=str(explicit)
            )

            self.assertEqual(run_dir, explicit)
            self.assertEqual(run_name, "custom-location")

    def test_resolve_run_identity_rejects_run_dir_run_name_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            explicit = root / "custom-location"

            with self.assertRaisesRegex(ValueError, "must refer to the same run identity"):
                resolve_run_identity(
                    root,
                    run_dir=str(explicit),
                    run_name="different-name",
                )

    def test_resolve_run_lifecycle_fails_for_existing_fresh_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "runs" / "demo").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "Run demo already exists"):
                resolve_run_lifecycle(
                    root,
                    ray_storage_path=root / "ray",
                    run_name="demo",
                    resume=False,
                )

    def test_resolve_run_lifecycle_rejects_resume_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                resolve_run_lifecycle(
                    root,
                    ray_storage_path=root / "ray",
                    run_name="demo",
                    resume=True,
                    overwrite=True,
                )

    def test_resolve_run_lifecycle_requires_run_name_for_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "training.resume=true requires an explicit training.run_name"):
                resolve_run_lifecycle(
                    root,
                    ray_storage_path=root / "ray",
                    resume=True,
                )

    def test_resolve_run_lifecycle_requires_ray_storage_on_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "runs" / "demo").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "Ray storage directory"):
                resolve_run_lifecycle(
                    root,
                    ray_storage_path=root / "ray",
                    run_name="demo",
                    resume=True,
                )

    def test_resolve_run_lifecycle_returns_paths_for_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "demo"
            run_dir.mkdir(parents=True)
            ray_run_dir = root / "ray" / "demo"
            ray_run_dir.mkdir(parents=True)

            resolved_run_dir, resolved_run_name = resolve_run_lifecycle(
                root,
                ray_storage_path=root / "ray",
                run_name="demo",
                resume=True,
            )

            self.assertEqual(resolved_run_dir, run_dir)
            self.assertEqual(resolved_run_name, "demo")

    def test_resolve_run_lifecycle_overwrite_removes_existing_run_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "demo"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.jsonl").write_text("old")
            ray_run_dir = root / "ray" / "demo"
            ray_run_dir.mkdir(parents=True)
            (ray_run_dir / "checkpoint_000001").mkdir()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                resolved_run_dir, resolved_run_name = resolve_run_lifecycle(
                    root,
                    ray_storage_path=root / "ray",
                    run_name="demo",
                    overwrite=True,
                )

            self.assertEqual(resolved_run_dir, run_dir)
            self.assertEqual(resolved_run_name, "demo")
            self.assertFalse(run_dir.exists())
            self.assertFalse(ray_run_dir.exists())
            self.assertEqual(len(caught), 1)
            self.assertIn("Overwriting existing run state for demo", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()