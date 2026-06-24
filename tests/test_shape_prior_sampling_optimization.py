import unittest
from unittest import mock
import importlib.util

import numpy as np
from scipy.spatial import cKDTree

from qqtt.demo import single_view_shape_prior_sampling as single_view_sampling
from qqtt.demo.single_view_shape_prior_sampling import SimpleShapeMesh
from data_process_sam3d.shape_prior_sampling import (
    ShapePriorBatchSelector,
    effective_shape_prior_max_dist,
)


def _legacy_point_grid_index(point: np.ndarray, min_bound: np.ndarray, grid_size: float) -> tuple[int, int, int]:
    return tuple(np.floor((point - min_bound) / grid_size).astype(int))


def _legacy_dedupe(points: np.ndarray, min_bound: np.ndarray, grid_size: float, limit: int) -> np.ndarray:
    seen: set[tuple[int, int, int]] = set()
    selected: list[np.ndarray] = []
    for point in points:
        grid_index = _legacy_point_grid_index(point, min_bound, grid_size)
        if grid_index in seen:
            continue
        seen.add(grid_index)
        selected.append(point)
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=points.dtype)


def _legacy_select(
    batches: list[np.ndarray],
    reference_points: np.ndarray,
    min_bound: np.ndarray,
    grid_size: float,
    max_dist: float,
    limit: int,
) -> np.ndarray:
    tree = cKDTree(reference_points)
    sorted_batches: list[np.ndarray] = []
    selected = np.empty((0, 3), dtype=np.float64)
    for batch in batches:
        distances, _ = tree.query(batch, k=1)
        if max_dist > 0:
            batch = batch[distances <= max_dist]
            distances = distances[distances <= max_dist]
        batch = batch[np.argsort(distances)]
        sorted_batches.append(batch)
        selected = _legacy_dedupe(np.vstack(sorted_batches), min_bound, grid_size, limit)
        if len(selected) >= limit:
            break
    return selected


def _cube_mesh(size: float = 0.10) -> SimpleShapeMesh:
    half = float(size) * 0.5
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return SimpleShapeMesh(vertices=vertices, faces=faces)


class ShapePriorSamplingOptimizationTest(unittest.TestCase):
    def test_effective_shape_prior_max_dist_is_backend_aware(self) -> None:
        self.assertEqual(effective_shape_prior_max_dist(0.08, "mvsam3d"), 0.035)
        self.assertEqual(effective_shape_prior_max_dist(0.02, "mvsam3d"), 0.02)
        self.assertEqual(effective_shape_prior_max_dist(0.08, "sam3d-single-view"), 0.08)
        self.assertEqual(effective_shape_prior_max_dist(0.05, "legacy"), 0.05)
        self.assertEqual(effective_shape_prior_max_dist(0.0, "mvsam3d"), 0.0)
        self.assertEqual(effective_shape_prior_max_dist(-0.1, "sam3d-single-view"), -0.1)
        with self.assertRaisesRegex(ValueError, "unsupported shape-prior sampling backend"):
            effective_shape_prior_max_dist(0.08, "unknown")

    def test_same_voxel_keeps_nearest_point_within_batch(self) -> None:
        selector = ShapePriorBatchSelector(
            reference_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            min_bound=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            grid_size=0.002,
            max_dist=0.035,
        )

        selected = selector.add_batch(
            np.array(
                [
                    [0.0015, 0.0, 0.0],
                    [0.0010, 0.0, 0.0],
                    [0.0032, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            limit=2,
        )

        np.testing.assert_allclose(selected, [[0.0010, 0.0, 0.0], [0.0032, 0.0, 0.0]])
        np.testing.assert_allclose(selector.points(), selected)
        self.assertEqual(selector.accepted_candidate_count, 3)

    def test_earlier_batch_keeps_voxel_priority_over_later_closer_point(self) -> None:
        selector = ShapePriorBatchSelector(
            reference_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            min_bound=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            grid_size=0.002,
            max_dist=0.035,
        )

        first = selector.add_batch(np.array([[0.0015, 0.0, 0.0]], dtype=np.float64), limit=2)
        second = selector.add_batch(np.array([[0.0001, 0.0, 0.0]], dtype=np.float64), limit=2)

        np.testing.assert_allclose(first, [[0.0015, 0.0, 0.0]])
        self.assertEqual(second.shape, (0, 3))
        np.testing.assert_allclose(selector.points(), [[0.0015, 0.0, 0.0]])

    def test_disabled_max_distance_still_sorts_by_reference_distance(self) -> None:
        selector = ShapePriorBatchSelector(
            reference_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            min_bound=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            grid_size=0.002,
            max_dist=0.0,
        )

        selected = selector.add_batch(
            np.array(
                [
                    [0.010, 0.0, 0.0],
                    [0.004, 0.0, 0.0],
                    [0.020, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            limit=3,
        )

        np.testing.assert_allclose(
            selected,
            [[0.004, 0.0, 0.0], [0.010, 0.0, 0.0], [0.020, 0.0, 0.0]],
        )
        self.assertEqual(selector.accepted_candidate_count, 3)

    def test_incremental_selection_matches_legacy_vstack_sort_dedupe(self) -> None:
        reference = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
        min_bound = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        grid_size = 0.002
        max_dist = 0.035
        limit = 5
        batches = [
            np.array(
                [
                    [0.0065, 0.0, 0.0],
                    [0.0015, 0.0, 0.0],
                    [0.0010, 0.0, 0.0],
                    [0.1000, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [0.0002, 0.0, 0.0],
                    [0.0043, 0.0, 0.0],
                    [0.0092, 0.0, 0.0],
                    [0.0110, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array([[0.0122, 0.0, 0.0], [0.0141, 0.0, 0.0]], dtype=np.float64),
        ]
        expected = _legacy_select(batches, reference, min_bound, grid_size, max_dist, limit)

        selector = ShapePriorBatchSelector(
            reference_points=reference,
            min_bound=min_bound,
            grid_size=grid_size,
            max_dist=max_dist,
        )
        for batch in batches:
            selector.add_batch(batch, limit=limit)
            if len(selector.points()) >= limit:
                break

        np.testing.assert_allclose(selector.points(), expected)
        self.assertEqual(selector.accepted_candidate_count, 7)

    @unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "open3d is required for voxel interior sampling")
    def test_single_view_sampling_uses_voxel_interior_before_random_volume(self) -> None:
        mesh = _cube_mesh()
        reference = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        with mock.patch.object(
            single_view_sampling,
            "_sample_volume",
            side_effect=AssertionError("random volume sampling should not run when voxel candidates are enough"),
        ):
            samples = single_view_sampling.sample_data_process_sam3d_single_view_shape_prior_points(
                mesh,
                reference,
                volume_sample_size_m=0.005,
                shape_prior_max_dist_m=1.0,
                target_surface_points=16,
                target_interior_points=64,
            )

        self.assertEqual(samples.surface_points_m.shape, (16, 3))
        self.assertEqual(samples.interior_points_m.shape, (64, 3))
        self.assertEqual(samples.metadata["shape_prior_interior_points"], 64)


if __name__ == "__main__":
    unittest.main()
