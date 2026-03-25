import unittest
from types import SimpleNamespace

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
    def __init__(self, in_channels=POINT_FEATURE_DIM, enc_channels=(8, 12), **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_channels, enc_channels[-1], bias=False)

    def forward(self, data_dict):
        point = as_point_cloud(data_dict)
        return SimpleNamespace(feat=self.proj(point.feat), offset=point.offset)


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
            "feat": [[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]],
        }

        point = as_point_cloud(view)

        self.assertAlmostEqual(point.grid_size.item(), DEFAULT_POINT_GRID_SIZE)

    def test_mean_pool_features_uses_offsets(self):
        feat = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [10.0, 20.0],
            ]
        )
        offset = torch.tensor([2, 3])

        pooled = mean_pool_features(feat, offset)

        expected = torch.tensor([[2.0, 3.0], [10.0, 20.0]])
        self.assertTrue(torch.allclose(pooled, expected))

    def test_forward_returns_expected_shapes(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"enc_channels": (8, 10)},
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
        self.assertEqual(tuple(student_outputs[0].shape), (2, 7))
        self.assertEqual(tuple(teacher_outputs[0].shape), (2, 7))

    def test_teacher_update_and_center_update(self):
        model = PandaSelfDistillation(
            in_channels=POINT_FEATURE_DIM,
            embed_channels=8,
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"enc_channels": (8, 10)},
        )
        with torch.no_grad():
            model.student_projector[0].weight.add_(1.0)

        teacher_before = model.teacher_projector[0].weight.detach().clone()
        student_weight = model.student_projector[0].weight.detach().clone()
        model.update_teacher(momentum=0.5)

        expected_teacher = teacher_before * 0.5 + student_weight * 0.5
        self.assertTrue(torch.allclose(model.teacher_projector[0].weight, expected_teacher))

        student_outputs = [torch.randn(2, 7), torch.randn(2, 7), torch.randn(2, 7)]
        teacher_outputs = [torch.randn(2, 7), torch.randn(2, 7)]
        loss = model.distillation_loss(student_outputs, teacher_outputs)

        self.assertEqual(loss.ndim, 0)
        center_before = model.center.detach().clone()
        model.update_center(teacher_outputs)
        self.assertFalse(torch.allclose(center_before, model.center))

    def test_small_model_factory_uses_shared_defaults(self):
        model = create_small_panda_model(backbone_cls=DummyBackbone, backbone_kwargs={"enc_channels": (8, 10)})

        self.assertEqual(model.grid_size, DEFAULT_POINT_GRID_SIZE)
        self.assertEqual(model.prototype_head.out_features, 32)


if __name__ == "__main__":
    unittest.main()
