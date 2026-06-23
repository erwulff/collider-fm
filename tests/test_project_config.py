import unittest

from omegaconf import OmegaConf

from collider_fm.project_config import point_view_kwargs, sonata_batch_kwargs


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "views": {
                    "coord_noise_scale": 0.001,
                    "energy_jitter_scale": 0.01,
                    "phi_rotation_max": 3.141592653589793,
                    "point_dropout": 0.05,
                    "num_global_views": 2,
                    "num_local_views": 4,
                    "global_crop_min_ratio": 0.4,
                    "global_crop_max_ratio": 1.0,
                    "local_crop_min_ratio": 0.1,
                    "local_crop_max_ratio": 0.4,
                    "coord_center": [0, 0, 0],
                    "coord_scale": 5000.0,
                    "energy_transform": "log",
                    "energy_min": 0.01,
                    "energy_max": 20.0,
                    "grid_sample_enabled": True,
                    "grid_sample_size": 0.002,
                },
                "model": {
                    "training": {"grid_size": 0.002},
                    "diagnostics": {"grid_size": 0.004},
                },
            }
        )

    def test_point_view_kwargs_uses_selected_model_flavor(self):
        kwargs = point_view_kwargs(
            self.config,
            "diagnostics",
            max_calo_hits=256,
        )

        self.assertEqual(kwargs["max_calo_hits"], 256)
        self.assertEqual(kwargs["grid_size"], 0.004)
        self.assertEqual(kwargs["coord_center"], [0, 0, 0])
        self.assertEqual(kwargs["coord_scale"], 5000.0)
        self.assertEqual(kwargs["energy_transform"], "log")
        self.assertTrue(kwargs["grid_sample_enabled"])
        self.assertEqual(kwargs["grid_sample_size"], 0.002)

    def test_sonata_batch_kwargs_adds_augmentation_fields(self):
        kwargs = sonata_batch_kwargs(
            self.config,
            "training",
            max_calo_hits=None,
        )

        self.assertIsNone(kwargs["max_calo_hits"])
        self.assertEqual(kwargs["grid_size"], 0.002)
        self.assertEqual(kwargs["coord_noise_scale"], 0.001)
        self.assertEqual(kwargs["feat_noise_scale"], 0.01)
        self.assertEqual(kwargs["phi_rotation_max"], 3.141592653589793)
        self.assertEqual(kwargs["point_dropout"], 0.05)
        self.assertEqual(kwargs["num_global_views"], 2)
        self.assertEqual(kwargs["num_local_views"], 4)
        self.assertEqual(kwargs["global_crop_min_ratio"], 0.4)
        self.assertEqual(kwargs["global_crop_max_ratio"], 1.0)
        self.assertEqual(kwargs["local_crop_min_ratio"], 0.1)
        self.assertEqual(kwargs["local_crop_max_ratio"], 0.4)


if __name__ == "__main__":
    unittest.main()
