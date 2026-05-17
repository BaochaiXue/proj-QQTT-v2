from __future__ import annotations

import unittest

import numpy as np

from qqtt.tracking.sampling import (
    phystwin_dense_query_count,
    sample_controller_sparse,
    sample_object_dense,
    sample_object_sparse,
    sample_phystwin_dense,
    sample_query_points_from_mask,
)


class Demo3TrackingSamplingSmokeTest(unittest.TestCase):
    def test_query_sampling_returns_deterministic_yx_points(self) -> None:
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 255
        points_a = sample_query_points_from_mask(mask, num_points=4, strategy="grid", seed=0)
        points_b = sample_query_points_from_mask(mask, num_points=4, strategy="grid", seed=999)

        np.testing.assert_array_equal(points_a, points_b)
        self.assertEqual(points_a.shape, (4, 2))
        self.assertTrue(np.all(mask[points_a[:, 0].astype(int), points_a[:, 1].astype(int)] > 0))

    def test_random_sampling_is_seeded_and_yx(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        points_a = sample_query_points_from_mask(mask, num_points=5, strategy="random", seed=42)
        points_b = sample_query_points_from_mask(mask, num_points=5, strategy="random", seed=42)
        np.testing.assert_array_equal(points_a, points_b)
        self.assertEqual(points_a.shape, (5, 2))

    def test_strict_sampling_rejects_too_small_mask(self) -> None:
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[1, 1] = 255
        with self.assertRaises(ValueError):
            sample_query_points_from_mask(mask, num_points=2, strict=True)

    def test_helper_wrappers_return_yx_arrays(self) -> None:
        mask = np.ones((6, 7), dtype=np.uint8)
        self.assertEqual(sample_object_sparse(mask, 3).shape, (3, 2))
        self.assertEqual(sample_object_dense(mask, 5).shape, (5, 2))
        self.assertEqual(sample_controller_sparse(mask, 2).shape, (2, 2))

    def test_phystwin_dense_query_count_is_fixed_at_5000(self) -> None:
        mask_5000 = np.ones((80, 80), dtype=np.uint8)
        self.assertEqual(phystwin_dense_query_count(mask_5000), 5000)
        mask_10000 = np.ones((120, 120), dtype=np.uint8)
        self.assertEqual(phystwin_dense_query_count(mask_10000), 5000)
        with self.assertRaises(ValueError):
            phystwin_dense_query_count(np.ones((10, 10), dtype=np.uint8))

    def test_phystwin_dense_sampling_matches_futurephystwin_torch_randperm(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch is not installed: {exc}")

        mask = np.ones((80, 80), dtype=np.uint8)
        coords = np.argwhere(mask).astype(np.float32)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(44)
        expected_idx = torch.randperm(len(coords), generator=generator)[:5000].numpy()
        points_a = sample_phystwin_dense(mask, seed=42, camera_idx=2, torch_device="cpu")
        points_b = sample_phystwin_dense(mask, seed=42, camera_idx=2, torch_device="cpu")
        points_c = sample_phystwin_dense(mask, seed=42, camera_idx=3, torch_device="cpu")

        self.assertEqual(points_a.shape, (5000, 2))
        np.testing.assert_array_equal(points_a, coords[expected_idx])
        np.testing.assert_array_equal(points_a, points_b)
        self.assertFalse(np.array_equal(points_a, points_c))
        self.assertTrue(np.all(mask[points_a[:, 0].astype(int), points_a[:, 1].astype(int)] > 0))


if __name__ == "__main__":
    unittest.main()
