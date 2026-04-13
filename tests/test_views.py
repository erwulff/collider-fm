import unittest

import torch

from collider_fm.views import (
    DEFAULT_POINT_GRID_SIZE,
    POINT_FEATURE_DIM,
    apply_patch_mask,
    augment_point_view,
    batch_point_views,
    build_distillation_views,
    build_point_view_from_event,
    build_sonata_batch,
    crop_point_view,
    rotate_around_beam_axis,
    sample_hit_indices,
)


class ViewTests(unittest.TestCase):
    def make_event(self):
        return {
            "calo_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "y": torch.tensor([4.0, 5.0, 6.0, 7.0]),
                "z": torch.tensor([7.0, 8.0, 9.0, 10.0]),
                "total_energy": torch.tensor([10.0, 11.0, 12.0, 13.0]),
            }
        }

    def test_build_point_view_from_event_uses_calo_hits_only(self):
        event = self.make_event()

        view = build_point_view_from_event(event, device=torch.device("cpu"))

        self.assertEqual(tuple(view["coord"].shape), (4, 3))
        self.assertEqual(tuple(view["feat"].shape), (4, POINT_FEATURE_DIM))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([4])))
        self.assertAlmostEqual(view["grid_size"].item(), DEFAULT_POINT_GRID_SIZE)
        self.assertTrue(
            torch.equal(view["total_energy"], torch.tensor([10.0, 11.0, 12.0, 13.0]))
        )

    def test_rotate_around_beam_axis_preserves_z(self):
        coord = torch.tensor([[1.0, 0.0, 3.0]])

        rotated = rotate_around_beam_axis(coord, angle=3.14159265 / 2.0)

        self.assertAlmostEqual(rotated[0, 2].item(), 3.0, places=5)
        self.assertAlmostEqual(rotated[0, 0].item(), 0.0, places=4)

    def test_crop_point_view_reduces_point_count(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )

        cropped = crop_point_view(view, keep_ratio=0.5)

        self.assertEqual(cropped["coord"].shape[0], 2)
        self.assertTrue(torch.equal(cropped["offset"], torch.tensor([2])))

    def test_apply_patch_mask_marks_at_least_one_patch(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu"), grid_size=1.0
        )

        masked = apply_patch_mask(view, mask_fraction=0.5)

        self.assertTrue(masked["mask"].any())
        self.assertEqual(masked["mask"].dtype, torch.bool)

    def test_augment_point_view_preserves_contract(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )

        augmented = augment_point_view(
            view,
            coord_noise_scale=0.1,
            feat_noise_scale=0.1,
            crop_keep_ratio=1.0,
            mask_fraction=0.5,
            point_dropout=0.0,
            view_kind="student_masked",
        )

        self.assertEqual(augmented["feat"].shape[1], POINT_FEATURE_DIM)
        self.assertTrue(torch.equal(augmented["feat"][:, 3], augmented["total_energy"]))
        self.assertEqual(augmented["view_kind"], "student_masked")

    def test_batch_point_views_builds_cumulative_offsets_and_unique_indices(self):
        first = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu"), grid_size=100.0
        )
        second = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu"), grid_size=100.0
        )

        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (8, 3))
        self.assertEqual(tuple(batched["feat"].shape), (8, POINT_FEATURE_DIM))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([4, 8])))
        self.assertTrue(
            torch.equal(batched["source_index"], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        )

    def test_build_distillation_views_returns_student_and_teacher_batches(self):
        batch = build_distillation_views(
            [self.make_event(), self.make_event()],
            device=torch.device("cpu"),
            coord_noise_scale=0.0,
            feat_noise_scale=0.0,
            global_crop_ratio=1.0,
            student_mask_fraction=0.5,
            point_dropout=0.0,
        )

        self.assertEqual(len(batch["student_views"]), 2)
        self.assertEqual(len(batch["teacher_views"]), 2)
        self.assertTrue(
            torch.equal(batch["student_views"][0]["offset"], torch.tensor([4, 8]))
        )
        self.assertTrue(batch["student_views"][0]["mask"].any())
        self.assertFalse(batch["teacher_views"][0]["mask"].any())

    def test_build_sonata_batch_returns_packed_global_and_local_views(self):
        batch = build_sonata_batch(
            [self.make_event(), self.make_event()],
            device=torch.device("cpu"),
            coord_noise_scale=0.0,
            feat_noise_scale=0.0,
            point_dropout=0.0,
            num_global_views=2,
            num_local_views=3,
            global_crop_min_ratio=1.0,
            global_crop_max_ratio=1.0,
            local_crop_min_ratio=1.0,
            local_crop_max_ratio=1.0,
        )

        self.assertEqual(tuple(batch["global_coord"].shape), (16, 3))
        self.assertEqual(tuple(batch["global_feat"].shape), (16, POINT_FEATURE_DIM))
        self.assertTrue(
            torch.equal(batch["global_offset"], torch.tensor([4, 8, 12, 16]))
        )
        self.assertEqual(tuple(batch["local_coord"].shape), (24, 3))
        self.assertTrue(
            torch.equal(batch["local_offset"], torch.tensor([4, 8, 12, 16, 20, 24]))
        )
        self.assertTrue(
            torch.equal(batch["global_coord"], batch["global_origin_coord"])
        )

    def test_sample_hit_indices_keeps_all_hits_when_below_limit(self):
        indices = sample_hit_indices(num_hits=5, max_hits=8, device=torch.device("cpu"))

        self.assertTrue(torch.equal(indices, torch.arange(5)))

    def test_build_point_view_rejects_empty_event(self):
        event = {
            "calo_hits": {
                "x": torch.tensor([]),
                "y": torch.tensor([]),
                "z": torch.tensor([]),
                "total_energy": torch.tensor([]),
            }
        }

        with self.assertRaisesRegex(ValueError, "zero calorimeter hits"):
            build_point_view_from_event(event, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
