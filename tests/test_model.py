import unittest

import torch

from collider_fm._panda.model_base import PointTransformerV3
from collider_fm.model import (
    create_small_model,
    create_small_sonata_model,
    create_training_model,
    create_training_sonata_model,
)
from collider_fm.views import POINT_FEATURE_DIM


class ModelTests(unittest.TestCase):
    def test_small_sonata_factory_uses_expected_defaults(self):
        model = create_small_sonata_model()

        self.assertEqual(model.grid_size, 0.002)
        self.assertEqual(model.num_global_view, 2)
        self.assertEqual(model.num_local_view, 4)

    def test_create_small_model_returns_sonata(self):
        model = create_small_model()
        self.assertEqual(getattr(model, "model_recipe", "unknown"), "sonata")

    def test_create_training_model_returns_sonata(self):
        model = create_training_model()
        self.assertEqual(getattr(model, "model_recipe", "unknown"), "sonata")

    def test_point_transformer_accepts_flash_attn_backend(self):
        model = PointTransformerV3(
            in_channels=POINT_FEATURE_DIM,
            enc_channels=(8, 12, 16, 24, 32),
            enc_num_head=(1, 1, 2, 4, 4),
            enc_patch_size=(4, 4, 4, 4, 4),
            enc_depths=(1, 1, 1, 1, 1),
            enable_flash=True,
            flash_backend="flash_attn",
            upcast_attention=False,
            upcast_softmax=False,
            enable_rpe=False,
            enc_mode=True,
        )

        first_stage = model.enc[0]
        first_block = first_stage[0]
        self.assertTrue(first_block.attn.enable_flash)
        self.assertEqual(first_block.attn.flash_backend, "flash_attn")


if __name__ == "__main__":
    unittest.main()
