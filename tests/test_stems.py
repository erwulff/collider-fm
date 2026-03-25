import unittest

import torch

from collider_fm.stems import CaloStem, ModalityFusion, TrackerStem


class StemTests(unittest.TestCase):
    def test_tracker_stem_emits_expected_shape(self):
        stem = TrackerStem(
            continuous_dim=4,
            embed_dim=3,
            detector_vocab=8,
            volume_vocab=16,
            layer_vocab=32,
            surface_vocab=64,
            output_dim=10,
        )
        continuous = torch.randn(5, 4)
        categorical = {
            "detector": torch.tensor([1, 2, 3, 4, 5]),
            "volume_id": torch.tensor([2, 2, 3, 3, 4]),
            "layer_id": torch.tensor([1, 2, 3, 4, 5]),
            "surface_id": torch.tensor([7, 8, 9, 10, 11]),
        }

        output = stem(continuous, categorical)

        self.assertEqual(tuple(output.shape), (5, 10))

    def test_calo_stem_emits_expected_shape(self):
        stem = CaloStem(continuous_dim=4, embed_dim=4, subsystem_vocab=10, output_dim=12)
        continuous = torch.randn(3, 4)
        categorical = {"detector": torch.tensor([1, 2, 3])}

        output = stem(continuous, categorical)

        self.assertEqual(tuple(output.shape), (3, 12))

    def test_modality_fusion_preserves_requested_order(self):
        fusion = ModalityFusion(feature_dim=6)
        tracker = torch.zeros(2, 6)
        calo = torch.zeros(3, 6)
        modality_id = torch.tensor([1, 0, 1, 0, 1])
        tracker_index = torch.tensor([1, 3])
        calo_index = torch.tensor([0, 2, 4])

        fused = fusion(
            tracker_features=tracker,
            calo_features=calo,
            modality_id=modality_id,
            tracker_index=tracker_index,
            calo_index=calo_index,
        )

        expected = fusion.modality_embedding(modality_id)
        self.assertTrue(torch.allclose(fused, expected))
        self.assertEqual(tuple(fused.shape), (5, 6))


if __name__ == "__main__":
    unittest.main()
