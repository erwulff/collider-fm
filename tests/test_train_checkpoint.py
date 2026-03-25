import tempfile
import unittest
import sys
from pathlib import Path

from types import SimpleNamespace

import torch.nn as nn
from torch.optim import AdamW

from collider_fm.model import PandaSelfDistillation, as_point_cloud
import importlib.util


TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_script", Path(__file__).resolve().parents[1] / "scripts" / "train.py"
)
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/train.py for the checkpoint test.")
train_script = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_script
TRAIN_SPEC.loader.exec_module(train_script)

load_checkpoint = train_script.load_checkpoint
save_checkpoint = train_script.save_checkpoint


class DummyBackbone(nn.Module):
    def __init__(self, in_channels=2, out_channels=10, **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, data_dict):
        point = as_point_cloud(data_dict)
        return SimpleNamespace(feat=self.proj(point.feat), offset=point.offset)


class TrainCheckpointTests(unittest.TestCase):
    def test_save_and_load_checkpoint_round_trip(self):
        model = PandaSelfDistillation(
            in_channels=2,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"out_channels": 10, "enc_channels": (8, 10)},
        )
        optimizer = AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            path = save_checkpoint(run_dir, model, optimizer, epoch=3, global_step=7)

            loaded_model = PandaSelfDistillation(
                in_channels=2,
                embed_channels=8,
                num_prototypes=7,
                projection_dim=6,
                prediction_dim=5,
                backbone_cls=DummyBackbone,
                backbone_kwargs={"out_channels": 10, "enc_channels": (8, 10)},
            )
            loaded_optimizer = AdamW(loaded_model.parameters(), lr=1e-4)
            epoch, step = load_checkpoint(str(path), loaded_model, loaded_optimizer)

            self.assertEqual(epoch, 3)
            self.assertEqual(step, 7)


if __name__ == "__main__":
    unittest.main()
