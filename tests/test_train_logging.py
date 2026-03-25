import unittest
from io import StringIO
from unittest.mock import Mock

from scripts.train import create_progress_bar_for_stream, target_num_batches


class TrainLoggingTests(unittest.TestCase):
    def test_target_num_batches_uses_smaller_limit(self):
        loader = Mock()
        loader.__len__ = Mock(return_value=7)
        self.assertEqual(target_num_batches(loader, 10), 7)
        self.assertEqual(target_num_batches(loader, 3), 3)

    def test_create_progress_bar_uses_requested_total(self):
        progress = create_progress_bar_for_stream(
            phase="train",
            num_batches=12,
            log_every=5,
            stream=StringIO(),
        )
        try:
            self.assertEqual(progress.total, 12)
            self.assertEqual(progress.desc, "train")
            self.assertEqual(progress.miniters, 5)
        finally:
            progress.close()


if __name__ == "__main__":
    unittest.main()
