import unittest

import torch

from collider_fm.features import build_multimodal_points
from collider_fm.views import (
    DEFAULT_POINT_GRID_SIZE,
    SSLViewConfig,
    augment_point_view,
    batch_ssl_views,
    flatten_ssl_view_set,
    build_global_view,
    build_local_view,
    build_masked_view,
    build_ssl_views,
    batch_point_views,
    build_distillation_views,
    build_point_view_from_event,
    normalize_feature,
    sample_hit_indices,
)


class ViewTests(unittest.TestCase):
    def make_event(self):
        return {
            "tracker_hits": {
                "x": torch.tensor([1.0, 2.0]),
                "y": torch.tensor([3.0, 4.0]),
                "z": torch.tensor([5.0, 6.0]),
                "time": torch.tensor([7.0, 8.0]),
            },
            "calo_hits": {
                "x": torch.tensor([10.0]),
                "y": torch.tensor([11.0]),
                "z": torch.tensor([12.0]),
                "total_energy": torch.tensor([13.0]),
            },
        }

    def test_build_point_view_from_event_combines_hits(self):
        event = self.make_event()

        view = build_point_view_from_event(event, device=torch.device("cpu"))
        augmented = augment_point_view(view, coord_noise_scale=0.0, feat_noise_scale=0.0)

        self.assertEqual(tuple(view["coord"].shape), (3, 3))
        self.assertEqual(tuple(view["feat"].shape), (3, 6))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([3])))
        self.assertAlmostEqual(view["grid_size"].item(), DEFAULT_POINT_GRID_SIZE)
        self.assertTrue(torch.equal(augmented["coord"], view["coord"]))
        self.assertTrue(torch.equal(augmented["feat"], view["feat"]))

    def test_augment_point_view_recomputes_derived_features(self):
        view = build_point_view_from_event(self.make_event(), device=torch.device("cpu"))

        augmented = augment_point_view(view, coord_noise_scale=0.5, feat_noise_scale=0.0)

        self.assertTrue(torch.allclose(augmented["feat"][:, :3], augmented["coord"]))
        expected_radius = normalize_feature(torch.linalg.norm(augmented["coord"], dim=1))
        self.assertTrue(torch.allclose(augmented["feat"][:, 3], expected_radius))
        self.assertTrue(torch.equal(augmented["feat"][:, 5], view["feat"][:, 5]))

    def test_batch_point_views_builds_cumulative_offsets(self):
        first = {
            "coord": torch.randn(2, 3),
            "feat": torch.randn(2, 6),
            "offset": torch.tensor([2]),
            "grid_size": torch.tensor(DEFAULT_POINT_GRID_SIZE),
        }
        second = {
            "coord": torch.randn(3, 3),
            "feat": torch.randn(3, 6),
            "offset": torch.tensor([3]),
            "grid_size": torch.tensor(DEFAULT_POINT_GRID_SIZE),
        }

        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (5, 3))
        self.assertEqual(tuple(batched["feat"].shape), (5, 6))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([2, 5])))

    def test_build_distillation_views_batches_multiple_events(self):
        views = build_distillation_views(
            [self.make_event(), self.make_event()],
            device=torch.device("cpu"),
            coord_noise_scale=0.0,
            feat_noise_scale=0.0,
        )

        self.assertEqual(len(views), 2)
        self.assertTrue(torch.equal(views[0]["offset"], torch.tensor([3, 6])))
        self.assertTrue(torch.equal(views[0]["feat"], views[1]["feat"]))

    def test_sample_hit_indices_keeps_all_hits_when_below_limit(self):
        indices = sample_hit_indices(num_hits=5, max_hits=8, device=torch.device("cpu"))

        self.assertTrue(torch.equal(indices, torch.arange(5)))

    def make_multimodal_event(self):
        return {
            "tracker_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "y": torch.tensor([5.0, 6.0, 7.0, 8.0]),
                "z": torch.tensor([9.0, 10.0, 11.0, 12.0]),
                "time": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "detector": torch.tensor([1, 1, 2, 2]),
                "volume_id": torch.tensor([3, 3, 4, 4]),
                "layer_id": torch.tensor([5, 6, 7, 8]),
                "surface_id": torch.tensor([9, 10, 11, 12]),
            },
            "calo_hits": {
                "x": torch.tensor([20.0, 21.0, 22.0]),
                "y": torch.tensor([23.0, 24.0, 25.0]),
                "z": torch.tensor([26.0, 27.0, 28.0]),
                "total_energy": torch.tensor([1.0, 2.0, 3.0]),
                "detector": torch.tensor([13, 14, 15]),
            },
        }

    def test_build_global_view_preserves_point_ids_and_modalities(self):
        torch.manual_seed(7)
        points = build_multimodal_points(self.make_multimodal_event(), device=torch.device("cpu"))
        config = SSLViewConfig(global_fraction_min=0.75, global_fraction_max=0.75, phi_rotation_max=0.0, coord_jitter_scale=0.0)

        view = build_global_view(points, config, view_type="teacher_global")

        self.assertEqual(view["view_type"], "teacher_global")
        self.assertEqual(view["point_id"].numel(), 5)
        self.assertEqual(view["visible_point_mask"].sum().item(), 5)
        self.assertTrue(torch.all(torch.isin(view["point_id"], points.point_id)))
        self.assertGreaterEqual(torch.unique(view["modality_id"]).numel(), 2)

    def test_build_local_view_returns_subset_with_preserved_ids(self):
        torch.manual_seed(3)
        points = build_multimodal_points(self.make_multimodal_event(), device=torch.device("cpu"))
        config = SSLViewConfig(local_fraction_min=0.4, local_fraction_max=0.4, phi_rotation_max=0.0, coord_jitter_scale=0.0)

        view = build_local_view(points, config)

        self.assertEqual(view["point_id"].numel(), 3)
        self.assertTrue(torch.all(torch.isin(view["point_id"], points.point_id)))
        self.assertEqual(view["visible_point_mask"].sum().item(), 3)

    def test_build_masked_view_drops_points_but_keeps_modalities_when_possible(self):
        torch.manual_seed(11)
        points = build_multimodal_points(self.make_multimodal_event(), device=torch.device("cpu"))
        config = SSLViewConfig(global_fraction_min=1.0, global_fraction_max=1.0, mask_fraction=0.34, phi_rotation_max=0.0, coord_jitter_scale=0.0)

        view = build_masked_view(points, config)

        self.assertLess(view["point_id"].numel(), points.point_id.numel())
        self.assertTrue(torch.all(torch.isin(view["point_id"], points.point_id)))
        self.assertGreaterEqual(torch.unique(view["modality_id"]).numel(), 2)

    def test_batch_ssl_views_builds_offsets_and_event_ids(self):
        torch.manual_seed(5)
        points_a = build_multimodal_points(self.make_multimodal_event(), device=torch.device("cpu"))
        points_b = build_multimodal_points(self.make_multimodal_event(), device=torch.device("cpu"))
        config = SSLViewConfig(global_fraction_min=0.5, global_fraction_max=0.5, phi_rotation_max=0.0, coord_jitter_scale=0.0)

        batched = batch_ssl_views(
            [
                build_global_view(points_a, config, view_type="teacher_global"),
                build_global_view(points_b, config, view_type="teacher_global"),
            ]
        )

        self.assertTrue(torch.equal(batched["offset"], torch.tensor([4, 8])))
        self.assertTrue(torch.equal(torch.unique(batched["event_id"]), torch.tensor([0, 1])))

    def test_build_ssl_views_returns_structured_batched_views(self):
        torch.manual_seed(13)
        config = SSLViewConfig(
            teacher_global_views=1,
            student_global_views=1,
            student_local_views=1,
            student_masked_views=1,
            global_fraction_min=0.75,
            global_fraction_max=0.75,
            local_fraction_min=0.5,
            local_fraction_max=0.5,
            phi_rotation_max=0.0,
            coord_jitter_scale=0.0,
        )

        view_set = build_ssl_views(
            [self.make_multimodal_event(), self.make_multimodal_event()],
            device=torch.device("cpu"),
            config=config,
        )

        self.assertEqual(sorted(view_set.keys()), ["student_global", "student_local", "student_masked", "teacher_global"])
        self.assertEqual(len(view_set["teacher_global"]), 1)
        self.assertEqual(len(view_set["student_local"]), 1)
        teacher_view = view_set["teacher_global"][0]
        self.assertEqual(teacher_view["view_type"], "teacher_global")
        self.assertTrue(torch.equal(torch.unique(teacher_view["event_id"]), torch.tensor([0, 1])))
        self.assertEqual(teacher_view["offset"].numel(), 2)

    def test_flatten_ssl_view_set_keeps_teacher_views_first(self):
        torch.manual_seed(17)
        config = SSLViewConfig(
            teacher_global_views=2,
            student_global_views=1,
            student_local_views=1,
            student_masked_views=1,
            phi_rotation_max=0.0,
            coord_jitter_scale=0.0,
        )

        view_set = build_ssl_views([self.make_multimodal_event()], device=torch.device("cpu"), config=config)
        flattened = flatten_ssl_view_set(view_set)

        self.assertEqual(
            [view["view_type"] for view in flattened],
            ["teacher_global", "teacher_global", "student_global", "student_local", "student_masked"],
        )


if __name__ == "__main__":
    unittest.main()
