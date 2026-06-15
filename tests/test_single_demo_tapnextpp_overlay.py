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


class _FakeFfsRunner:
    def __init__(self) -> None:
        self.call_count = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
    ) -> dict[str, np.ndarray]:
        _ = right, baseline_m
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.call_count += 1
            call_idx = self.call_count
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return {
            "depth_ir_left_m": np.full(left.shape, 0.25 + float(call_idx), dtype=np.float32),
            "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
        }


class _FakeFfsAligner:
    def align(self, depth_ir_left_m: np.ndarray) -> np.ndarray:
        return np.asarray(depth_ir_left_m, dtype=np.float32) + np.float32(1.0)


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

    def _ffs_mask_packet(self, seq: int) -> demo.MaskPacket:
        controller = np.zeros((4, 4), dtype=bool)
        obj = np.zeros((4, 4), dtype=bool)
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        ir_left = np.full((4, 4), 32 + int(seq), dtype=np.uint8)
        ir_right = np.full((4, 4), 64 + int(seq), dtype=np.uint8)
        return demo.MaskPacket(
            seq=int(seq),
            color_bgr=color,
            depth_source="ffs",
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=1.0,
            receive_perf_s=time.perf_counter(),
            process_done_perf_s=time.perf_counter(),
            dropped_capture_frames=0,
            timing=demo.PipelineTiming(),
            controller_mask=controller,
            object_mask=obj,
            depth_u16=None,
            ir_left_u8=ir_left,
            ir_right_u8=ir_right,
            k_ir_left=np.eye(3, dtype=np.float32),
            t_ir_left_to_color=np.eye(4, dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            ir_baseline_m=0.05,
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

    def test_render_point_cap_uses_deterministic_spatial_spread(self) -> None:
        left = np.array([[float(idx) * 0.001, 0.0, 0.5] for idx in range(10)], dtype=np.float32)
        right = left + np.array([1.0, 0.0, 0.0], dtype=np.float32)
        points = np.vstack([left, right])
        colors = np.arange(points.shape[0] * 3, dtype=np.uint8).reshape(-1, 3)

        capped_points, capped_colors = demo.cap_render_points(points, colors, max_points=6)
        capped_points_again, capped_colors_again = demo.cap_render_points(points, colors, max_points=6)

        self.assertEqual(capped_points.shape, (6, 3))
        self.assertEqual(capped_colors.shape, (6, 3))
        self.assertGreater(np.count_nonzero(capped_points[:, 0] < 0.5), 0)
        self.assertGreater(np.count_nonzero(capped_points[:, 0] > 0.5), 0)
        np.testing.assert_array_equal(capped_points, capped_points_again)
        np.testing.assert_array_equal(capped_colors, capped_colors_again)

    def test_render_point_cap_balances_sparse_separated_region(self) -> None:
        dense = np.array(
            [
                [
                    0.001 * float(sample % 10),
                    0.02 * float(y_bin) + 0.0001 * float(sample),
                    0.50 + 0.02 * float(z_bin),
                ]
                for y_bin in range(8)
                for z_bin in range(8)
                for sample in range(100)
            ],
            dtype=np.float32,
        )
        sparse = np.array(
            [
                [
                    1.0 + 0.01 * float(sample % 5),
                    0.02 * float(y_bin) + 0.0002 * float(sample),
                    0.50 + 0.02 * float(z_bin),
                ]
                for y_bin in range(4)
                for z_bin in range(4)
                for sample in range(5)
            ],
            dtype=np.float32,
        )
        points = np.vstack([dense, sparse])
        colors = np.arange(points.shape[0] * 3, dtype=np.uint8).reshape(-1, 3)

        capped_points, capped_colors = demo.cap_render_points(points, colors, max_points=72)
        capped_points_again, capped_colors_again = demo.cap_render_points(points, colors, max_points=72)

        self.assertEqual(capped_points.shape, (72, 3))
        self.assertEqual(capped_colors.shape, (72, 3))
        self.assertGreaterEqual(np.count_nonzero(capped_points[:, 0] > 0.5), 8)
        np.testing.assert_array_equal(capped_points, capped_points_again)
        np.testing.assert_array_equal(capped_colors, capped_colors_again)

    def test_render_point_cap_zero_keeps_original_arrays(self) -> None:
        points = np.zeros((8, 3), dtype=np.float32)
        colors = np.zeros((8, 3), dtype=np.uint8)

        capped_points, capped_colors = demo.cap_render_points(points, colors, max_points=0)

        self.assertIs(capped_points, points)
        self.assertIs(capped_colors, colors)

    def test_tracker_lift_mask_uses_current_union_mask_and_erosion(self) -> None:
        packet = self._mask_packet()

        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        mask = runtime._tracker_lift_mask(packet)
        expected_union = np.logical_or(packet.controller_mask, packet.object_mask)
        self.assertIsNotNone(mask)
        np.testing.assert_array_equal(mask, expected_union)

        erode_args = self._tracker_args()
        erode_args.pcd_mask_erode_pixels = 1
        erode_runtime = demo.RealtimeMaskedEdgeTamPcdDemo(erode_args)
        eroded = erode_runtime._tracker_lift_mask(packet)
        expected_eroded = np.zeros_like(expected_union)
        expected_eroded[1:3, 1:3] = True
        self.assertIsNotNone(eroded)
        np.testing.assert_array_equal(eroded, expected_eroded)

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

    def test_local_ffs_depth_is_cached_per_sequence(self) -> None:
        args = demo.build_parser().parse_args(["--depth-source", "ffs"])
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runner = _FakeFfsRunner()
        runtime.ffs_runner = runner
        runtime._get_ir_to_color_aligner = lambda **_kwargs: _FakeFfsAligner()  # type: ignore[method-assign]
        packet = self._ffs_mask_packet(seq=7)

        first_depth, first_ffs_ms, first_align_ms = runtime._compute_ffs_depth_color_m(packet)
        second_depth, second_ffs_ms, second_align_ms = runtime._compute_ffs_depth_color_m(packet)

        self.assertEqual(runner.call_count, 1)
        self.assertIs(first_depth, second_depth)
        self.assertEqual(first_ffs_ms, second_ffs_ms)
        self.assertEqual(first_align_ms, second_align_ms)
        np.testing.assert_allclose(first_depth, np.full((4, 4), 2.25, dtype=np.float32))

    def test_local_ffs_runner_is_serialized_across_sequences(self) -> None:
        args = demo.build_parser().parse_args(["--depth-source", "ffs"])
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runner = _FakeFfsRunner()
        runtime.ffs_runner = runner
        runtime._get_ir_to_color_aligner = lambda **_kwargs: _FakeFfsAligner()  # type: ignore[method-assign]
        packets = [self._ffs_mask_packet(seq=11), self._ffs_mask_packet(seq=12)]
        outputs: list[np.ndarray] = []

        def compute(packet: demo.MaskPacket) -> None:
            outputs.append(runtime._compute_ffs_depth_color_m(packet)[0])

        threads = [threading.Thread(target=compute, args=(packet,)) for packet in packets]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1.0)

        self.assertEqual(runner.call_count, 2)
        self.assertEqual(runner.max_active, 1)
        self.assertEqual(len(outputs), 2)


if __name__ == "__main__":
    unittest.main()
