from __future__ import annotations

import unittest

import numpy as np

from qqtt.tracking.backends.cotracker3_adapter import CoTracker3Adapter
from qqtt.tracking.backends.point_tracker_adapter import (
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
    effective_legacy_update_mode,
    normalize_tracker_backend,
    normalize_tracker_batch_query_count_policy,
    normalize_tracker_execution_mode,
    tracker_backend_spec,
)
from qqtt.tracking.base import TrackingResult


class _FakeCoTrackerBackend:
    def __init__(self) -> None:
        self.query_points = np.empty((0, 2), dtype=np.float32)
        self.query_points_by_camera: dict[int, np.ndarray] = {}

    def availability(self):
        from qqtt.tracking.base import BackendAvailability

        return BackendAvailability("fake", True, "ok")

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str:
        return "ok"

    def warmup(self):
        return {"total_ms": 1.0}

    def initialize(self, frames, query_points_yx, masks=None) -> None:
        _ = frames, masks
        self.query_points = np.asarray(query_points_yx, dtype=np.float32)

    def update(self, frame) -> TrackingResult:
        _ = frame
        return TrackingResult(
            tracks_yx=self.query_points[None, :, :],
            visibility=np.ones((1, len(self.query_points)), dtype=np.float32),
            query_points_yx=self.query_points,
            stats={"stream_status": "published"},
        )

    def initialize_batch(self, query_points_yx_by_camera) -> None:
        self.query_points_by_camera = {
            int(camera_idx): np.asarray(points, dtype=np.float32)
            for camera_idx, points in query_points_yx_by_camera.items()
        }

    def update_batch(self, frames_by_camera):
        return {
            int(camera_idx): TrackingResult(
                tracks_yx=points[None, :, :],
                visibility=np.ones((1, len(points)), dtype=np.float32),
                camera_idx=int(camera_idx),
                query_points_yx=points,
                stats={"stream_status": "published", "update_mode": "batch"},
            )
            for camera_idx, points in self.query_points_by_camera.items()
        }


class PointTrackerAdaptersTest(unittest.TestCase):
    def test_backend_normalization_and_specs(self) -> None:
        self.assertEqual(normalize_tracker_backend("co-tracker3"), "cotracker3_online")
        self.assertEqual(normalize_tracker_backend("track_on2"), "trackon2")
        self.assertEqual(normalize_tracker_backend("lite-tracker"), "litetracker")
        self.assertTrue(tracker_backend_spec("cotracker3_online").supports_batch_views)
        self.assertTrue(tracker_backend_spec("trackon2").supports_batch_views)
        self.assertFalse(tracker_backend_spec("litetracker").supports_batch_views)

    def test_execution_mode_and_policy_normalization(self) -> None:
        self.assertEqual(normalize_tracker_execution_mode("batch"), "batch-views")
        self.assertEqual(effective_legacy_update_mode("batch-views"), "batch")
        self.assertEqual(normalize_tracker_batch_query_count_policy("min_common"), "min-common")

    def test_factory_returns_external_adapter_shells(self) -> None:
        trackon = build_point_tracker_adapter_factory(PointTrackerAdapterConfig(backend="trackon2"))(-1)
        lite = build_point_tracker_adapter_factory(PointTrackerAdapterConfig(backend="litetracker"))(-1)

        self.assertEqual(trackon.name, "trackon2")
        self.assertFalse(trackon.availability().available)
        self.assertEqual(lite.name, "litetracker")
        self.assertFalse(lite.availability().available)

    def test_cotracker_adapter_serial_and_batch_shapes(self) -> None:
        adapter = CoTracker3Adapter(backend=_FakeCoTrackerBackend())
        query = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        adapter.initialize([], query)
        serial = adapter.update(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(serial.backend, "cotracker3_online")
        self.assertEqual(serial.tracks_yx.shape, (1, 2, 2))

        batch = CoTracker3Adapter(backend=_FakeCoTrackerBackend())
        batch.initialize_batch({0: query, 1: query[:1]})
        results = batch.update_batch({0: np.zeros((4, 4, 3), dtype=np.uint8), 1: np.zeros((4, 4, 3), dtype=np.uint8)})

        self.assertEqual(set(results), {0, 1})
        self.assertEqual(results[0].backend, "cotracker3_online")
        self.assertEqual(results[1].tracks_yx.shape, (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
