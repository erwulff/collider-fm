import unittest
from typing import cast

import torch

from collider_fm._panda.model_base import PointTransformerV3
from collider_fm.model import (
    IdentityBackbone,
    PandaSelfDistillation,
    create_small_model,
    as_point_cloud,
    create_small_panda_model,
    create_small_sonata_model,
    create_training_panda_model,
    mean_pool_features,
    panda_loss,
    pointwise_panda_loss,
)
from collider_fm.views import DEFAULT_POINT_GRID_SIZE, POINT_FEATURE_DIM


class ModelTests(unittest.TestCase):
    def test_as_point_cloud_adds_defaults(self):
        view = {
            "coord": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "feat": [[1.0, 2.0], [3.0, 4.0]],
        }

        point = as_point_cloud(view, default_grid_size=0.2)

        self.assertTrue(torch.equal(point.offset, torch.tensor([2])))
        self.assertAlmostEqual(point.grid_size.item(), 0.2)
        self.assertEqual(tuple(point.coord.shape), (2, 3))
        self.assertEqual(tuple(point.feat.shape), (2, 2))

    def test_as_point_cloud_uses_project_default_grid_size(self):
        view = {
            "coord": [[0.0, 0.0, 0.0]],
            "feat": [[1.0, 2.0, 3.0, 4.0]],
        }

        point = as_point_cloud(view)

        self.assertAlmostEqual(point.grid_size.item(), DEFAULT_POINT_GRID_SIZE)

    def test_mean_pool_features_uses_offsets(self):
        feat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        offset = torch.tensor([2, 3])

        pooled = mean_pool_features(feat, offset)

        expected = torch.tensor([[2.0, 3.0], [10.0, 20.0]])
        self.assertTrue(torch.allclose(pooled, expected))

    def test_forward_returns_point_level_outputs(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=IdentityBackbone,
            backbone_kwargs={"output_dim": 10},
        )
        batch = {
            "student_views": [
                {
                    "coord": torch.randn(5, 3),
                    "feat": torch.randn(5, POINT_FEATURE_DIM),
                    "offset": torch.tensor([2, 5]),
                    "source_index": torch.tensor([0, 1, 2, 3, 4]),
                    "mask": torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool),
                }
            ],
            "teacher_views": [
                {
                    "coord": torch.randn(5, 3),
                    "feat": torch.randn(5, POINT_FEATURE_DIM),
                    "offset": torch.tensor([2, 5]),
                    "source_index": torch.tensor([0, 1, 2, 3, 4]),
                    "mask": torch.zeros(5, dtype=torch.bool),
                }
            ],
        }

        student_outputs, teacher_outputs = model(batch)

        self.assertEqual(len(student_outputs), 1)
        self.assertEqual(len(teacher_outputs), 1)
        self.assertEqual(tuple(student_outputs[0]["point_logits"].shape), (5, 7))
        self.assertEqual(tuple(student_outputs[0]["masked_logits"].shape), (2, 7))
        self.assertEqual(tuple(teacher_outputs[0]["point_logits"].shape), (5, 7))

    def test_pointwise_panda_loss_matches_shared_indices(self):
        student_logits = torch.randn(4, 5)
        teacher_logits = torch.randn(3, 5)
        student_source_index = torch.tensor([10, 11, 12, 13])
        teacher_source_index = torch.tensor([11, 13, 15])
        student_mask = torch.tensor([0, 1, 0, 1], dtype=torch.bool)

        loss = pointwise_panda_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_source_index=student_source_index,
            teacher_source_index=teacher_source_index,
            student_mask=student_mask,
            center=torch.zeros(1, 5),
            temp_s=0.1,
            temp_t=0.04,
        )

        self.assertGreaterEqual(loss.item(), 0.0)

    def test_pointwise_panda_loss_uses_masked_matches_only(self):
        student_logits = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32
        )
        teacher_logits = torch.tensor(
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=torch.float32
        )
        student_source_index = torch.tensor([10, 11, 12])
        teacher_source_index = torch.tensor([11, 12, 15])
        student_mask = torch.tensor([0, 1, 1], dtype=torch.bool)
        center = torch.zeros(1, 2)

        loss = pointwise_panda_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_source_index=student_source_index,
            teacher_source_index=teacher_source_index,
            student_mask=student_mask,
            center=center,
            temp_s=0.1,
            temp_t=0.04,
        )

        expected = panda_loss(
            student_logits[1:3], teacher_logits[0:2], center, temp_s=0.1, temp_t=0.04
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_pointwise_panda_loss_falls_back_to_all_points_when_mask_empty(self):
        student_logits = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32
        )
        teacher_logits = torch.tensor(
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=torch.float32
        )
        student_source_index = torch.tensor([10, 11, 12])
        teacher_source_index = torch.tensor([11, 12, 15])
        student_mask = torch.zeros(3, dtype=torch.bool)
        center = torch.zeros(1, 2)

        loss = pointwise_panda_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_source_index=student_source_index,
            teacher_source_index=teacher_source_index,
            student_mask=student_mask,
            center=center,
            temp_s=0.1,
            temp_t=0.04,
        )

        expected = panda_loss(
            student_logits[1:3], teacher_logits[0:2], center, temp_s=0.1, temp_t=0.04
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_pointwise_panda_loss_returns_zero_without_overlap(self):
        loss = pointwise_panda_loss(
            student_logits=torch.randn(3, 4),
            teacher_logits=torch.randn(2, 4),
            student_source_index=torch.tensor([1, 2, 3]),
            teacher_source_index=torch.tensor([10, 11]),
            student_mask=torch.tensor([1, 1, 1], dtype=torch.bool),
            center=torch.zeros(1, 4),
            temp_s=0.1,
            temp_t=0.04,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_teacher_update_and_center_update(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=IdentityBackbone,
            backbone_kwargs={"output_dim": 10},
        )
        with torch.no_grad():
            student_linear = cast(torch.nn.Linear, model.student_projector[0])
            student_linear.weight.add_(1.0)

        teacher_before = (
            cast(torch.nn.Linear, model.teacher_projector[0]).weight.detach().clone()
        )
        student_weight = student_linear.weight.detach().clone()
        model.update_teacher(momentum=0.5)

        expected_teacher = teacher_before * 0.5 + student_weight * 0.5
        self.assertTrue(
            torch.allclose(
                cast(torch.nn.Linear, model.teacher_projector[0]).weight,
                expected_teacher,
            )
        )

        teacher_outputs = [
            {
                "point_logits": torch.randn(3, 7),
            }
        ]
        center_before = cast(torch.Tensor, model.center).detach().clone()
        model.update_center(teacher_outputs)
        self.assertFalse(
            torch.allclose(center_before, cast(torch.Tensor, model.center))
        )

    def test_small_model_factory_uses_shared_defaults(self):
        model = create_small_panda_model(
            backbone_cls=IdentityBackbone, backbone_kwargs={"output_dim": 32}
        )

        self.assertEqual(model.grid_size, DEFAULT_POINT_GRID_SIZE)
        self.assertEqual(model.prototype_head.out_features, 32)

    def test_training_model_factory_uses_larger_defaults(self):
        model = create_training_panda_model(
            backbone_cls=IdentityBackbone, backbone_kwargs={"output_dim": 32}
        )

        self.assertEqual(model.grid_size, DEFAULT_POINT_GRID_SIZE)
        self.assertEqual(model.prototype_head.out_features, 256)

    def test_small_sonata_factory_uses_expected_defaults(self):
        model = create_small_sonata_model()

        self.assertEqual(model.grid_size, DEFAULT_POINT_GRID_SIZE)
        self.assertEqual(model.num_prototypes, 32)
        self.assertEqual(model.num_global_view, 2)
        self.assertEqual(model.num_local_view, 4)

    def test_create_small_model_selects_recipe(self):
        legacy_model = create_small_model(
            recipe="legacy",
            backbone_cls=IdentityBackbone,
            backbone_kwargs={"output_dim": 32},
        )
        sonata_model = create_small_model(recipe="sonata")

        self.assertIsInstance(legacy_model, PandaSelfDistillation)
        self.assertEqual(getattr(legacy_model, "model_recipe", "legacy"), "legacy")
        self.assertEqual(getattr(sonata_model, "model_recipe", "unknown"), "sonata")

    def test_point_transformer_accepts_torch_flash_backend(self):
        model = PointTransformerV3(
            in_channels=POINT_FEATURE_DIM,
            enc_channels=(8, 12, 16, 24, 32),
            enc_num_head=(1, 1, 2, 4, 4),
            enc_patch_size=(4, 4, 4, 4, 4),
            enc_depths=(1, 1, 1, 1, 1),
            enable_flash=True,
            flash_backend="torch",
            upcast_attention=False,
            upcast_softmax=False,
            enable_rpe=False,
            enc_mode=True,
        )

        first_stage = model.enc[0]
        first_block = first_stage[0]
        self.assertTrue(first_block.attn.enable_flash)
        self.assertEqual(first_block.attn.flash_backend, "torch")


if __name__ == "__main__":
    unittest.main()
