from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.phystwin_volume_filter import (
    ObjectVoxelBudgetController,
    phystwin_volume_sample_indices,
    phystwin_volume_sample_indices_fast,
    phystwin_volume_sample_points,
)


class PhysTwinVolumeFilterTests(unittest.TestCase):
    def test_one_point_per_occupied_voxel(self) -> None:
        points = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.001, 0.001, 0.001],
                [0.006, 0.000, 0.000],
                [0.000, 0.006, 0.000],
            ],
            dtype=np.float32,
        )

        idx = phystwin_volume_sample_indices(points, voxel_size_m=0.005, origin_world=np.zeros(3))

        self.assertEqual(idx.tolist(), [0, 2, 3])

    def test_fast_indices_match_contract_indices(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.uniform(-0.05, 0.05, size=(200, 3)).astype(np.float32)

        expected = phystwin_volume_sample_indices(
            points,
            voxel_size_m=0.005,
            origin_world=np.zeros(3, dtype=np.float32),
        )
        actual = phystwin_volume_sample_indices_fast(
            points,
            voxel_size_m=0.005,
            origin_world=np.zeros(3, dtype=np.float32),
        )

        np.testing.assert_array_equal(actual, expected)

    def test_output_count_equals_occupied_voxels_for_default(self) -> None:
        points = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.001, 0.001, 0.001],
                [0.005, 0.000, 0.000],
                [0.010, 0.000, 0.000],
            ],
            dtype=np.float32,
        )

        sampled, _colors, stats = phystwin_volume_sample_points(
            points,
            voxel_size_m=0.005,
            origin_world=np.zeros(3, dtype=np.float32),
        )

        self.assertEqual(stats["occupied_voxel_count"], 3)
        self.assertEqual(sampled.shape[0], stats["occupied_voxel_count"])
        self.assertEqual(stats["output_point_count"], 3)
        self.assertIn("object_volume_key_ms", stats)
        self.assertIn("object_volume_unique_ms", stats)
        self.assertIn("object_volume_gather_ms", stats)
        self.assertIn("object_volume_total_ms", stats)
        self.assertEqual(stats["object_volume_sampler_impl"], "numpy-unique")

    def test_points_per_voxel_keeps_multiple_representatives(self) -> None:
        points = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.001, 0.001, 0.001],
                [0.002, 0.002, 0.002],
                [0.006, 0.000, 0.000],
                [0.007, 0.000, 0.000],
            ],
            dtype=np.float32,
        )

        idx = phystwin_volume_sample_indices(
            points,
            voxel_size_m=0.005,
            origin_world=np.zeros(3, dtype=np.float32),
            points_per_voxel=2,
        )

        self.assertEqual(idx.tolist(), [0, 1, 3, 4])

    def test_voxel_size_monotonically_changes_output_count(self) -> None:
        points = np.array([[x, 0.0, 0.0] for x in np.linspace(0.0, 0.019, 20)], dtype=np.float32)

        small, _colors, _stats = phystwin_volume_sample_points(
            points,
            voxel_size_m=0.002,
            origin_world=np.zeros(3, dtype=np.float32),
        )
        large, _colors, _stats = phystwin_volume_sample_points(
            points,
            voxel_size_m=0.010,
            origin_world=np.zeros(3, dtype=np.float32),
        )

        self.assertGreaterEqual(small.shape[0], large.shape[0])

    def test_emergency_cap_triggers_after_voxel_sampling(self) -> None:
        points = np.array([[float(i) * 0.01, 0.0, 0.0] for i in range(10)], dtype=np.float32)

        sampled, _colors, stats = phystwin_volume_sample_points(
            points,
            voxel_size_m=0.005,
            origin_world=np.zeros(3, dtype=np.float32),
            emergency_max_points=4,
        )

        self.assertEqual(stats["occupied_voxel_count"], 10)
        self.assertTrue(stats["safety_cap_triggered"])
        self.assertEqual(sampled.shape[0], 4)

    def test_adaptive_controller_changes_voxel_size_not_point_count(self) -> None:
        controller = ObjectVoxelBudgetController(target_ms=8.0, base_voxel_m=0.005, min_voxel_m=0.005, max_voxel_m=0.012)

        increased = controller.update(20.0)
        lowered = controller.update(1.0)

        self.assertGreater(increased, 0.005)
        self.assertGreaterEqual(lowered, 0.005)


if __name__ == "__main__":
    unittest.main()
