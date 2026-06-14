from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from qqtt.demo import realtime_masked_edgetam_pcd as demo
from qqtt.demo.realtime_single_camera_pointcloud import CameraIntrinsics
from qqtt.tracking.base import TrackingResult


class _FakeTapNextAdapter:
    name = "tapnextpp"

    def __init__(self) -> None:
        self.query_points_yx: np.ndarray | None = None
        self.initialized = False

    def initialize(self, frames: list[np.ndarray], query_points_yx: np.ndarray, masks: list[np.ndarray] | None = None) -> None:
        self.query_points_yx = np.ascontiguousarray(query_points_yx, dtype=np.float32)
        self.initialized = True

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self.query_points_yx is None:
            raise RuntimeError("fake adapter was not initialized")
        return TrackingResult(
            tracks_yx=self.query_points_yx[None, :, :],
            visibility=np.ones((1, len(self.query_points_yx)), dtype=np.float32),
            backend=self.name,
            camera_idx=0,
            query_points_yx=self.query_points_yx,
            stats={"model_run_ms": 1.25},
        )


class SingleDemoTapNextOverlayTest(unittest.TestCase):
    def _tracker_args(self):
        args = demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--tracker-backend",
                "tapnextpp",
                "--tracker-query-count",
                "4",
                "--tracker-overlay-max-points",
                "4",
                "--tracker-display-scope",
                "union",
            ]
        )
        demo.apply_demo_preset(args)
        demo.validate_args(args)
        return args

    def _mask_packet(self) -> demo.MaskPacket:
        controller = np.zeros((4, 4), dtype=bool)
        obj = np.zeros((4, 4), dtype=bool)
        controller[2:, :] = True
        obj[:2, :] = True
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.full((4, 4), 1000, dtype=np.uint16)
        return demo.MaskPacket(
            seq=0,
            color_bgr=color,
            depth_source="realsense",
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=time.perf_counter(),
            process_done_perf_s=time.perf_counter(),
            dropped_capture_frames=0,
            timing=demo.PipelineTiming(),
            controller_mask=controller,
            object_mask=obj,
            depth_u16=depth,
        )

    def test_display_visibility_scopes_query_labels(self) -> None:
        visibility = np.ones((4,), dtype=np.float32)
        query_is_object = np.array([True, False, True, False])
        query_is_controller = np.array([False, True, False, True])

        np.testing.assert_array_equal(
            demo._tracker_display_visibility(
                visibility,
                query_is_object=query_is_object,
                query_is_controller=query_is_controller,
                display_scope=demo.TRACKER_DISPLAY_SCOPE_OBJECT,
            ),
            np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            demo._tracker_display_visibility(
                visibility,
                query_is_object=query_is_object,
                query_is_controller=query_is_controller,
                display_scope=demo.TRACKER_DISPLAY_SCOPE_UNION,
            ),
            visibility,
        )

    def test_visible_spread_selection_filters_nonfinite_tracks(self) -> None:
        tracks = np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        visibility = np.ones((4,), dtype=np.float32)

        selected = demo._select_visible_spread_indices(tracks, visibility, max_points=2)

        self.assertEqual(len(selected), 2)
        self.assertNotIn(1, selected.tolist())

    def test_tracker_worker_publishes_lifted_marker_packet_with_fake_adapter(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        fake_adapter = _FakeTapNextAdapter()
        runtime._build_tracker_adapter = lambda: fake_adapter  # type: ignore[method-assign]
        runtime.mask_slot.put(self._mask_packet())

        thread = threading.Thread(target=runtime._tracker_worker, daemon=True)
        thread.start()
        deadline = time.time() + 3.0
        while time.time() < deadline and runtime.tracker_marker_slot.latest_seq() < 0:
            time.sleep(0.01)
        runtime.stop_event.set()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        packet = runtime.tracker_marker_slot.get_latest_after(-1)
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertTrue(fake_adapter.initialized)
        self.assertEqual(packet.backend, "tapnextpp")
        self.assertEqual(packet.query_count, 4)
        self.assertEqual(packet.display_scope, "union")
        self.assertGreater(packet.marker_count, 0)
        self.assertLessEqual(packet.marker_count, 4)
        self.assertEqual(packet.marker_colors_rgb_u8.shape, (packet.marker_count, 3))
        self.assertTrue(np.all(packet.marker_xyz_m[:, 2] > 0.0))

    def test_fatal_worker_error_records_once_and_requests_render_update(self) -> None:
        args = demo.build_parser().parse_args(["--render-mode", "none", "--track-mode", "none", "--pcd-mode", "none"])
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        render_requests: list[int] = []
        runtime._request_render_update = lambda: render_requests.append(1)

        first = runtime._record_fatal_worker_error("EdgeTAM segmentation", RuntimeError("cuda oom"))
        second = runtime._record_fatal_worker_error("TAPNext++ tracker", RuntimeError("later failure"))

        self.assertIs(first, second)
        self.assertTrue(runtime.stop_event.is_set())
        self.assertEqual(render_requests, [1])
        self.assertEqual(first.stage, "EdgeTAM segmentation")
        self.assertIn("cuda oom", runtime._format_fatal_hud(first))

    def test_worker_thread_wrapper_records_unhandled_fatal_error(self) -> None:
        args = demo.build_parser().parse_args(["--render-mode", "none", "--track-mode", "none", "--pcd-mode", "none"])
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)

        def fail_capture() -> None:
            raise RuntimeError("synthetic capture failure")

        runtime._capture_worker = fail_capture  # type: ignore[method-assign]
        runtime._start_threads()
        deadline = time.time() + 1.0
        while time.time() < deadline and runtime._fatal_error_snapshot() is None:
            time.sleep(0.01)
        runtime.stop()

        fatal = runtime._fatal_error_snapshot()
        self.assertIsNotNone(fatal)
        assert fatal is not None
        self.assertEqual(fatal.stage, "capture worker")
        self.assertIn("synthetic capture failure", fatal.message)


if __name__ == "__main__":
    unittest.main()
