from __future__ import annotations

import unittest

import numpy as np

from qqtt.tracking.backends.cotracker3_adapter import CoTracker3Adapter
from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend
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
        self.assertEqual(tracker_backend_spec("litetracker").batch_support_status, "serial_only")

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
        self.assertIn("--litetracker-weights", lite.availability().reason)

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

    def test_cotracker_online_batch_tensors_are_contiguous(self) -> None:
        frames = [
            np.zeros((3, 4, 5, 3), dtype=np.uint8),
            np.ones((3, 4, 5, 3), dtype=np.uint8),
        ]
        video = CoTracker3OnlineBackend._batch_frames_to_torch_video(frames, device="cpu")
        queries = CoTracker3OnlineBackend._batch_queries_yx_to_torch(
            {
                0: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                1: np.array([[5.0, 6.0]], dtype=np.float32),
                2: np.array([[7.0, 8.0]], dtype=np.float32),
            },
            camera_ids=(0, 1, 2),
            device="cpu",
        )

        self.assertEqual(tuple(video.shape), (3, 2, 3, 4, 5))
        self.assertTrue(video.is_contiguous())
        self.assertEqual(tuple(queries.shape), (3, 2, 3))
        self.assertTrue(queries.is_contiguous())

    def test_cotracker_online_batch_results_preserve_camera_order_and_xy_yx_round_trip(self) -> None:
        backend = CoTracker3OnlineBackend(device="cpu", model=object())
        query_points_by_camera = {
            0: np.array([[100.0, 200.0]], dtype=np.float32),
            1: np.array([[110.0, 210.0]], dtype=np.float32),
            2: np.array([[120.0, 220.0]], dtype=np.float32),
        }
        backend._batch_camera_ids = (0, 1, 2)
        backend._batch_query_points_yx_by_camera = query_points_by_camera
        backend._batch_query_counts_by_camera = {0: 1, 1: 1, 2: 1}
        backend._batch_total_frames = 1

        tracks_xy = np.array(
            [
                [[[200.0, 100.0]]],
                [[[210.0, 110.0]]],
                [[[220.0, 120.0]]],
            ],
            dtype=np.float32,
        )
        visibility = np.ones((3, 1, 1), dtype=np.float32)

        results = backend._tracks_to_batch_results(
            tracks_xy=tracks_xy,
            visibility=visibility,
            run_ms=1.0,
            step=1,
            window_len=2,
        )

        self.assertEqual(tuple(results), (0, 1, 2))
        for camera_idx, expected_yx in query_points_by_camera.items():
            result = results[camera_idx]
            self.assertEqual(result.camera_idx, camera_idx)
            np.testing.assert_allclose(result.tracks_yx[0, 0], expected_yx[0])
            np.testing.assert_allclose(result.query_points_yx[0], expected_yx[0])
            self.assertEqual(result.stats["batch_camera_ids"], [0, 1, 2])

    def test_cotracker_online_forward_window_patch_makes_expanded_coords_contiguous(self) -> None:
        import torch

        class _Core:
            def __init__(self) -> None:
                self.contiguous_seen: dict[str, bool] = {}

            def forward_window(self, *, coords, track_feat_support_pyramid, vis=None, conf=None, attention_mask=None):
                self.contiguous_seen = {
                    "coords": bool(coords.is_contiguous()),
                    "track_feat_support": bool(track_feat_support_pyramid[0].is_contiguous()),
                    "vis": bool(vis.is_contiguous()),
                    "conf": bool(conf.is_contiguous()),
                    "attention_mask": bool(attention_mask.is_contiguous()),
                }
                return "ok"

        class _Predictor:
            def __init__(self) -> None:
                self.model = _Core()

        predictor = _Predictor()
        CoTracker3OnlineBackend._patch_online_model_for_batch_views(predictor)
        coords = torch.zeros((2, 1, 3, 2)).expand(2, 4, 3, 2)
        track_feat = torch.zeros((2, 4, 3, 5)).transpose(1, 2)
        vis = torch.zeros((2, 1, 3, 1)).expand(2, 4, 3, 1)
        conf = torch.zeros((2, 1, 3, 1)).expand(2, 4, 3, 1)
        attention_mask = torch.zeros((2, 1, 3)).expand(2, 4, 3)

        self.assertFalse(coords.is_contiguous())
        self.assertEqual(
            predictor.model.forward_window(
                coords=coords,
                track_feat_support_pyramid=[track_feat],
                vis=vis,
                conf=conf,
                attention_mask=attention_mask,
            ),
            "ok",
        )
        self.assertEqual(
            predictor.model.contiguous_seen,
            {
                "coords": True,
                "track_feat_support": True,
                "vis": True,
                "conf": True,
                "attention_mask": True,
            },
        )

    def test_cotracker_online_cuda_cache_is_reserved_for_oom_recovery(self) -> None:
        self.assertTrue(CoTracker3OnlineBackend._is_cuda_oom_error(RuntimeError("CUDA out of memory.")))
        self.assertTrue(CoTracker3OnlineBackend._is_cuda_oom_error(RuntimeError("CUBLAS out of memory.")))
        self.assertFalse(CoTracker3OnlineBackend._is_cuda_oom_error(RuntimeError("shape mismatch")))


if __name__ == "__main__":
    unittest.main()
