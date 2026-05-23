from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.pcd_postprocess import (
    COMPONENT_SELECTION_LARGEST_N,
    COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
    COMPONENT_SELECTION_MAIN_PLUS_GAP,
    apply_enhanced_phystwin_like_postprocess_with_trace,
)
from qqtt.demo.semantic_surface_filter import filter_semantic_surface_points
from qqtt.demo.three_view_masked_fused_pcd_runtime import (
    FusedLayerCloud,
    POSTPROCESS_ENHANCED_PT,
    apply_semantic_postprocess,
)


def _cluster(center_x: float, count: int, *, spacing: float = 0.0005) -> np.ndarray:
    points = np.zeros((count, 3), dtype=np.float32)
    side = int(np.ceil(np.sqrt(count)))
    for idx in range(count):
        points[idx, 0] = np.float32(center_x + (idx % side) * spacing)
        points[idx, 1] = np.float32((idx // side) * spacing)
        points[idx, 2] = np.float32(1.0)
    return points


def _run_filter(
    points: np.ndarray,
    *,
    policy: str,
    top_n: int,
    gap_m: float = 0.0,
    min_points: int = 1,
) -> tuple[np.ndarray, dict, dict]:
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    filtered, _colors, stats, trace = apply_enhanced_phystwin_like_postprocess_with_trace(
        points=points,
        colors=colors,
        enabled=True,
        radius_m=0.01,
        nb_points=1,
        component_voxel_size_m=0.01,
        keep_near_main_gap_m=gap_m,
        keep_top_n_components=top_n,
        component_selection_policy=policy,
        min_component_points=min_points,
        min_component_ratio=0.0,
    )
    return filtered, stats, trace


class EnhancedPtTopNSurfaceFilterTest(unittest.TestCase):
    def test_largest_n_keeps_exactly_largest_n_3d_components(self) -> None:
        points = np.concatenate([_cluster(0.0, 40), _cluster(1.0, 25), _cluster(2.0, 10)], axis=0)

        filtered, stats, trace = _run_filter(points, policy=COMPONENT_SELECTION_LARGEST_N, top_n=2)

        self.assertEqual(stats["top_n_component_indices"], [0, 1])
        self.assertEqual(stats["top_n_component_point_counts"], [40, 25])
        self.assertEqual(stats["kept_component_indices"], [0, 1])
        self.assertEqual(len(filtered), 65)
        self.assertEqual(int(np.count_nonzero(trace["kept_mask"])), 65)
        self.assertFalse(np.any(filtered[:, 0] > 1.5))

    def test_largest_n_plus_gap_keeps_top_n_plus_nearby_small_component(self) -> None:
        points = np.concatenate([_cluster(0.0, 40), _cluster(1.0, 25), _cluster(0.03, 6)], axis=0)

        filtered, stats, _trace = _run_filter(
            points,
            policy=COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
            top_n=1,
            gap_m=0.05,
        )

        self.assertEqual(stats["top_n_component_indices"], [0])
        self.assertEqual(stats["kept_component_indices"], [0, 2])
        self.assertEqual(len(filtered), 46)
        self.assertTrue(np.any((filtered[:, 0] > 0.02) & (filtered[:, 0] < 0.06)))
        self.assertFalse(np.any(filtered[:, 0] > 0.9))

    def test_main_plus_gap_preserves_old_largest_component_behavior(self) -> None:
        points = np.concatenate([_cluster(0.0, 40), _cluster(1.0, 25), _cluster(0.03, 6)], axis=0)

        filtered, stats, _trace = _run_filter(
            points,
            policy=COMPONENT_SELECTION_MAIN_PLUS_GAP,
            top_n=2,
            gap_m=0.05,
        )

        self.assertEqual(stats["kept_component_indices"], [0, 2])
        self.assertEqual(len(filtered), 46)
        self.assertFalse(np.any(filtered[:, 0] > 0.9))

    def test_min_component_points_blocks_tiny_second_top_n_noise_island(self) -> None:
        points = np.concatenate([_cluster(0.0, 40), _cluster(1.0, 4)], axis=0)

        filtered, stats, _trace = _run_filter(
            points,
            policy=COMPONENT_SELECTION_LARGEST_N,
            top_n=2,
            min_points=8,
        )

        self.assertEqual(stats["top_n_component_indices"], [0])
        self.assertEqual(stats["kept_component_indices"], [0])
        self.assertEqual(len(filtered), 40)

    def test_semantic_surface_filter_exposes_original_survivor_indices(self) -> None:
        points = np.concatenate([_cluster(0.0, 8), _cluster(1.0, 6)], axis=0)
        colors = np.arange(len(points) * 3, dtype=np.uint8).reshape(-1, 3)

        result = filter_semantic_surface_points(
            points_world=points,
            colors=colors,
            enabled=True,
            radius_m=0.01,
            nb_points=1,
            component_voxel_size_m=0.01,
            keep_near_main_gap_m=0.0,
            keep_top_n_components=1,
            component_selection_policy=COMPONENT_SELECTION_LARGEST_N,
            min_component_points=1,
            min_component_ratio=0.0,
        )

        np.testing.assert_array_equal(result.survivor_indices, np.arange(8, dtype=np.int64))
        np.testing.assert_array_equal(result.kept_mask_in_input, np.r_[np.ones(8, dtype=bool), np.zeros(6, dtype=bool)])
        np.testing.assert_array_equal(result.filtered_points, points[result.survivor_indices])
        np.testing.assert_array_equal(result.filtered_colors, colors[result.survivor_indices])

    def test_rendered_pcd_path_removes_same_floating_component_policy(self) -> None:
        points = np.concatenate([_cluster(0.0, 8), _cluster(1.0, 6)], axis=0)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        layer = FusedLayerCloud(
            obj_id=1,
            label="controller",
            postprocess_mode=POSTPROCESS_ENHANCED_PT,
            points_m=points,
            colors_rgb=colors,
            per_camera=(),
        )

        filtered_points, filtered_colors, stats = apply_semantic_postprocess(
            layer,
            filter_cap=0,
            filter_voxel_size_m=0.004,
            phystwin_radius_m=0.01,
            phystwin_nb_points=1,
            enhanced_component_voxel_size_m=0.01,
            enhanced_keep_near_main_gap_m=0.0,
            enhanced_keep_top_n_components=1,
            enhanced_component_selection_policy=COMPONENT_SELECTION_LARGEST_N,
            enhanced_min_component_points=1,
            enhanced_min_component_ratio=0.0,
            apply_enhanced_component_filter_to_pcd=True,
        )

        self.assertEqual(len(filtered_points), 8)
        self.assertEqual(len(filtered_colors), 8)
        self.assertEqual(stats["kept_component_indices"], [0])
        self.assertEqual(stats["removed_component_indices"], [1])
        self.assertFalse(np.any(filtered_points[:, 0] > 0.9))


if __name__ == "__main__":
    unittest.main()
