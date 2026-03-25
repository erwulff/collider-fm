import unittest
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn

from collider_fm.model import (
    PandaSelfDistillation,
    as_point_cloud,
    create_small_panda_model,
    mean_pool_features,
)
from collider_fm.views import DEFAULT_POINT_GRID_SIZE, POINT_FEATURE_DIM


class DummyBackbone(nn.Module):
    def __init__(self, in_channels=POINT_FEATURE_DIM, out_channels=10, **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, data_dict):
        point = as_point_cloud(data_dict)
        return SimpleNamespace(feat=self.proj(point.feat), offset=point.offset)


class ModelTests(unittest.TestCase):
    def test_as_point_cloud_adds_defaults(self):
        point = as_point_cloud({"coord": [[0.0, 0.0, 0.0]], "feat": [[1.0, 0.0]]})
        self.assertTrue(torch.equal(point.offset, torch.tensor([1])))
        self.assertAlmostEqual(point.grid_size.item(), DEFAULT_POINT_GRID_SIZE)

    def test_mean_pool_features_uses_offsets(self):
        feat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        pooled = mean_pool_features(feat, torch.tensor([2, 3]))
        self.assertTrue(
            torch.allclose(pooled, torch.tensor([[2.0, 3.0], [10.0, 20.0]]))
        )

    def test_forward_returns_point_level_shapes(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"out_channels": 10, "enc_channels": (8, 10)},
        )
        views = [
            {
                "coord": torch.randn(5, 3),
                "feat": torch.randn(5, POINT_FEATURE_DIM),
                "offset": torch.tensor([2, 5]),
            }
            for _ in range(3)
        ]
        student_outputs, teacher_outputs = model(views)

        self.assertEqual(len(student_outputs), 3)
        self.assertEqual(len(teacher_outputs), 2)
        self.assertEqual(tuple(student_outputs[0].shape), (5, 7))
        self.assertEqual(tuple(teacher_outputs[0].shape), (5, 7))

    def test_teacher_update_and_center_update(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"out_channels": 10, "enc_channels": (8, 10)},
        )
        with torch.no_grad():
            cast(nn.Linear, model.student_projector[0]).weight.add_(1.0)

        teacher_before = (
            cast(nn.Linear, model.teacher_projector[0]).weight.detach().clone()
        )
        student_weight = (
            cast(nn.Linear, model.student_projector[0]).weight.detach().clone()
        )
        model.update_teacher(momentum=0.5)
        self.assertTrue(
            torch.allclose(
                cast(nn.Linear, model.teacher_projector[0]).weight,
                teacher_before * 0.5 + student_weight * 0.5,
            )
        )

        student_outputs = [torch.randn(5, 7), torch.randn(5, 7), torch.randn(5, 7)]
        teacher_outputs = [torch.randn(5, 7), torch.randn(5, 7)]
        loss = model.distillation_loss(student_outputs, teacher_outputs)
        self.assertEqual(loss.ndim, 0)
        center_before = cast(torch.Tensor, model.center).detach().clone()
        model.update_center(teacher_outputs)
        self.assertFalse(
            torch.allclose(center_before, cast(torch.Tensor, model.center))
        )

    def test_distillation_loss_can_use_loss_masks(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"out_channels": 10, "enc_channels": (8, 10)},
        )
        student_outputs = [torch.randn(5, 7), torch.randn(5, 7), torch.randn(5, 7)]
        teacher_outputs = [torch.randn(5, 7), torch.randn(5, 7)]
        loss_masks = [
            torch.tensor([True, True, True, True, True]),
            torch.tensor([True, False, True, False, True]),
            torch.tensor([False, True, False, True, False]),
        ]

        loss = model.distillation_loss(
            student_outputs, teacher_outputs, loss_masks=loss_masks
        )
        self.assertEqual(loss.ndim, 0)

    def test_small_model_factory_uses_shared_defaults(self):
        model = create_small_panda_model(
            backbone_cls=DummyBackbone,
            backbone_kwargs={"out_channels": 10, "enc_channels": (8, 10)},
        )
        self.assertEqual(model.grid_size, DEFAULT_POINT_GRID_SIZE)
        self.assertEqual(model.prototype_head.out_features, 32)


if __name__ == "__main__":
    unittest.main()
