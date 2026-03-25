import unittest

import torch

from collider_fm.views import (
    DEFAULT_POINT_GRID_SIZE,
    augment_point_view,
    batch_point_views,
    build_distillation_views,
    build_point_view_from_event,
    local_crop_point_view,
    mask_point_view,
    sample_hit_indices,
)


class ViewTests(unittest.TestCase):
    def make_event(self):
        return {
            "calo_hits": {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([4.0, 5.0, 6.0]),
                "z": torch.tensor([7.0, 8.0, 9.0]),
                "detector": torch.tensor([10, 13, 9]),
                "total_energy": torch.tensor([10.0, 11.0, 12.0]),
            }
        }

    def test_build_point_view_from_event_is_calo_only(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )

        self.assertEqual(tuple(view["coord"].shape), (3, 3))
        self.assertEqual(tuple(view["feat"].shape), (3, 2))
        self.assertTrue(torch.equal(view["offset"], torch.tensor([3])))
        self.assertAlmostEqual(view["grid_size"].item(), DEFAULT_POINT_GRID_SIZE)
        self.assertTrue(torch.equal(view["energy"], torch.tensor([10.0, 11.0, 12.0])))
        self.assertTrue(torch.equal(view["calo_type"], torch.tensor([0.0, 1.0, 0.0])))
        self.assertFalse(view["hidden_mask"].any())
        self.assertTrue(view["loss_mask"].all())

    def test_augment_point_view_preserves_shape_and_detector_type(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        augmented = augment_point_view(
            view, coord_noise_scale=0.0, energy_jitter_scale=0.0
        )

        self.assertEqual(tuple(augmented["coord"].shape), (3, 3))
        self.assertEqual(tuple(augmented["feat"].shape), (3, 2))
        self.assertTrue(torch.equal(augmented["calo_type"], view["calo_type"]))
        self.assertTrue(torch.equal(augmented["energy"], view["energy"]))

    def test_mask_point_view_hides_points_and_marks_loss_mask(self):
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        masked_view = mask_point_view(view, mask_fraction=1 / 3)

        self.assertEqual(int(masked_view["hidden_mask"].sum().item()), 1)
        self.assertTrue(
            torch.equal(masked_view["hidden_mask"], masked_view["loss_mask"])
        )
        self.assertEqual(int((masked_view["energy"] == 0.0).sum().item()), 1)

    def test_local_crop_point_view_keeps_subset_for_loss(self):
        torch.manual_seed(0)
        view = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        local_view = local_crop_point_view(view, keep_fraction=2 / 3)

        self.assertEqual(int(local_view["loss_mask"].sum().item()), 2)
        self.assertEqual(int(local_view["hidden_mask"].sum().item()), 1)
        self.assertTrue(
            torch.equal(local_view["hidden_mask"], ~local_view["loss_mask"])
        )

    def test_batch_point_views_builds_cumulative_offsets(self):
        first = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        second = build_point_view_from_event(
            self.make_event(), device=torch.device("cpu")
        )
        batched = batch_point_views([first, second])

        self.assertEqual(tuple(batched["coord"].shape), (6, 3))
        self.assertEqual(tuple(batched["feat"].shape), (6, 2))
        self.assertTrue(torch.equal(batched["offset"], torch.tensor([3, 6])))

    def test_build_distillation_views_batches_multiple_events(self):
        views = build_distillation_views(
            [self.make_event(), self.make_event()],
            device=torch.device("cpu"),
            add_local_view=True,
            add_masked_view=True,
        )

        self.assertEqual(len(views), 4)
        self.assertTrue(torch.equal(views[0]["offset"], torch.tensor([3, 6])))
        self.assertEqual(tuple(views[0]["feat"].shape), (6, 2))
        self.assertIn("hidden_mask", views[2])
        self.assertIn("loss_mask", views[3])

    def test_sample_hit_indices_keeps_all_hits_when_below_limit(self):
        indices = sample_hit_indices(num_hits=5, max_hits=8, device=torch.device("cpu"))
        self.assertTrue(torch.equal(indices, torch.arange(5)))


if __name__ == "__main__":
    unittest.main()
