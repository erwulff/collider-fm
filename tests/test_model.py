import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from collider_fm.features import build_model_inputs, build_multimodal_points
from collider_fm.model import (
    MultimodalPandaSSL,
    PandaSelfDistillation,
    as_point_cloud,
    create_small_multimodal_model,
    create_small_panda_model,
    mean_pool_features,
)
from collider_fm.stems import CaloStem, TrackerStem
from collider_fm.views import DEFAULT_POINT_GRID_SIZE


class DummyBackbone(nn.Module):
    def __init__(self, in_channels=4, enc_channels=(8, 12), **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_channels, enc_channels[-1], bias=False)

    def forward(self, data_dict):
        point = as_point_cloud(data_dict)
        return SimpleNamespace(feat=self.proj(point.feat), offset=point.offset)


class ModelTests(unittest.TestCase):
    def make_multimodal_view(self):
        event = {
            "tracker_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([4.0, 5.0, 6.0]),
                "z": torch.tensor([7.0, 8.0, 9.0]),
                "time": torch.tensor([0.5, 1.5, 2.5]),
                "detector": torch.tensor([1, 1, 2]),
                "volume_id": torch.tensor([3, 4, 5]),
                "layer_id": torch.tensor([6, 7, 8]),
                "surface_id": torch.tensor([9, 10, 11]),
            },
            "calo_hits": {
                "x": torch.tensor([10.0, 11.0]),
                "y": torch.tensor([12.0, 13.0]),
                "z": torch.tensor([14.0, 15.0]),
                "total_energy": torch.tensor([1.0, 2.0]),
                "detector": torch.tensor([3, 4]),
            },
        }
        return build_model_inputs(build_multimodal_points(event, device=torch.device("cpu")))

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
            in_channels=4,
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
                "feat": torch.randn(5, 4),
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
            in_channels=4,
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

    def test_multimodal_model_forward_returns_expected_shapes(self):
        model = MultimodalPandaSSL(
            tracker_stem=TrackerStem(
                continuous_dim=4,
                embed_dim=4,
                detector_vocab=16,
                volume_vocab=16,
                layer_vocab=16,
                surface_vocab=16,
                output_dim=8,
            ),
            calo_stem=CaloStem(continuous_dim=4, embed_dim=4, subsystem_vocab=16, output_dim=8),
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"enc_channels": (8, 10)},
        )
        outputs = model([self.make_multimodal_view(), self.make_multimodal_view(), self.make_multimodal_view()])

        self.assertEqual(len(outputs.student_logits), 3)
        self.assertEqual(len(outputs.teacher_logits), 2)
        self.assertEqual(tuple(outputs.student_logits[0].shape), (1, 7))
        self.assertEqual(tuple(outputs.student_point_features[0].shape), (5, 10))
        self.assertTrue(torch.equal(outputs.student_point_ids[0], torch.tensor([0, 1, 2, 3, 4])))

    def test_multimodal_model_loss_and_center_update(self):
        model = MultimodalPandaSSL(
            tracker_stem=TrackerStem(
                continuous_dim=4,
                embed_dim=4,
                detector_vocab=16,
                volume_vocab=16,
                layer_vocab=16,
                surface_vocab=16,
                output_dim=8,
            ),
            calo_stem=CaloStem(continuous_dim=4, embed_dim=4, subsystem_vocab=16, output_dim=8),
            num_prototypes=7,
            projection_dim=6,
            prediction_dim=5,
            backbone_cls=DummyBackbone,
            backbone_kwargs={"enc_channels": (8, 10)},
        )
        outputs = model([self.make_multimodal_view(), self.make_multimodal_view()])

        loss = model.distillation_loss(outputs)

        self.assertEqual(loss.ndim, 0)
        center_before = model.center.detach().clone()
        model.update_center(outputs)
        self.assertFalse(torch.allclose(center_before, model.center))

    def test_small_multimodal_factory_uses_shared_defaults(self):
        model = create_small_multimodal_model(backbone_cls=DummyBackbone, backbone_kwargs={"enc_channels": (8, 10)})

        self.assertEqual(model.grid_size, DEFAULT_POINT_GRID_SIZE)
        self.assertEqual(model.prototype_head.out_features, 32)


if __name__ == "__main__":
    unittest.main()
