from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from qqtt.demo import realtime_masked_edgetam_pcd as demo
from qqtt.demo.query_rainbow import query_rainbow_colors_from_points_yx_rgb_u8, query_rainbow_colors_rgb_u8
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


class _StaticTrackingAdapter:
    name = "tapnextpp"

    def __init__(self, tracks_yx: np.ndarray, visibility: np.ndarray | None = None) -> None:
        self.tracks_yx = np.ascontiguousarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
        if visibility is None:
            visibility = np.ones((len(self.tracks_yx),), dtype=np.float32)
        self.visibility = np.ascontiguousarray(visibility, dtype=np.float32).reshape(-1)

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        return TrackingResult(
            tracks_yx=self.tracks_yx[None, :, :],
            visibility=self.visibility[None, :],
            backend=self.name,
            camera_idx=0,
            query_points_yx=self.tracks_yx,
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
    def test_transform_points_c2w_matches_phystwin_homogeneous_lift(self) -> None:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        points = np.array([[0.0, 0.0, 0.5], [0.1, 0.2, 0.3]], dtype=np.float32)

        transformed = demo._transform_points_c2w(points, c2w)

        expected = points + np.array([0.25, -0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(transformed, expected, atol=1e-6)

    def test_pcd_packet_build_applies_table_c2w_before_render_packet(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        runtime.table_c2w = np.eye(4, dtype=np.float32)
        runtime.table_c2w[:3, 3] = np.array([0.25, -0.5, 1.0], dtype=np.float32)

        result = runtime._build_pcd_packet_from_mask(self._mask_packet(), rng=np.random.default_rng(0))

        self.assertEqual(result.packet.coordinate_frame, "table_world_z0")
        self.assertGreater(result.packet.controller_point_count, 0)
        self.assertGreater(result.packet.object_point_count, 0)
        np.testing.assert_allclose(
            result.packet.controller_xyz_m,
            np.tile(np.array([[0.25, -0.5, 2.0]], dtype=np.float32), (result.packet.controller_point_count, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.packet.object_xyz_m,
            np.tile(np.array([[0.25, -0.5, 2.0]], dtype=np.float32), (result.packet.object_point_count, 1)),
            atol=1e-6,
        )

    def test_world_z_diagnostics_reports_quantiles_and_threshold_candidates(self) -> None:
        diagnostics = demo.build_world_z_diagnostics(
            object_xyz_m=np.array(
                [
                    [0.0, 0.0, -0.20],
                    [0.0, 0.0, -0.01],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.012],
                    [0.0, 0.0, 0.05],
                ],
                dtype=np.float32,
            ),
            controller_xyz_m=np.array([[0.0, 0.0, -0.03]], dtype=np.float32),
            table_z_m=0.0,
            thresholds_m=(0.005, 0.02),
        )

        self.assertEqual(diagnostics["table_z_m"], 0.0)
        self.assertEqual(diagnostics["table_z_above_direction"], demo.TABLE_Z_ABOVE_DIRECTION_NEGATIVE)
        self.assertEqual(diagnostics["thresholds_m"], [0.005, 0.02])
        object_stats = diagnostics["classes"]["object"]
        self.assertEqual(object_stats["count"], 5)
        self.assertAlmostEqual(object_stats["z_m"]["p50"], 0.0, places=6)
        self.assertEqual(object_stats["table_thresholds"][0]["candidate_count"], 3)
        self.assertEqual(object_stats["table_thresholds"][1]["candidate_count"], 4)
        self.assertAlmostEqual(object_stats["table_thresholds"][1]["candidate_ratio"], 0.8, places=6)
        self.assertEqual(diagnostics["classes"]["controller"]["table_thresholds"][1]["candidate_count"], 0)

    def test_table_z_filter_uses_negative_above_table_direction_by_default(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, -0.20],
                [0.0, 0.0, -0.01],
                [0.0, 0.0, 0.01],
            ],
            dtype=np.float32,
        )
        colors = np.arange(9, dtype=np.uint8).reshape(3, 3)

        filtered_points, filtered_colors, stats = demo.apply_table_z_filter(
            points,
            colors,
            enabled=True,
            threshold_m=0.02,
            table_z_m=0.0,
        )

        self.assertEqual(stats["table_z_above_direction"], demo.TABLE_Z_ABOVE_DIRECTION_NEGATIVE)
        self.assertEqual(stats["removed_points"], 2)
        np.testing.assert_allclose(filtered_points, np.array([[0.0, 0.0, -0.20]], dtype=np.float32))
        np.testing.assert_array_equal(filtered_colors, colors[:1])

    def test_table_z_filter_applies_after_world_transform_when_enabled(self) -> None:
        args = demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--tracker-backend",
                "tapnextpp",
                "--tracker-query-count",
                "4",
                "--enable-table-z-filter",
                "--table-z-filter-threshold-m",
                "0.02",
                "--table-z-filter-classes",
                "both",
            ]
        )
        demo.apply_demo_preset(args)
        demo.validate_args(args)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        runtime.table_c2w = np.eye(4, dtype=np.float32)
        runtime.table_c2w[2, 3] = -1.0

        result = runtime._build_pcd_packet_from_mask(self._mask_packet(), rng=np.random.default_rng(0))

        self.assertEqual(result.packet.coordinate_frame, "table_world_z0")
        self.assertEqual(result.packet.object_point_count, 0)
        self.assertEqual(result.packet.controller_point_count, 0)
        self.assertEqual(result.world_z_diagnostics["classes"]["object"]["table_thresholds"][0]["candidate_count"], 8)
        self.assertEqual(result.world_z_diagnostics["classes"]["controller"]["table_thresholds"][0]["candidate_count"], 8)

    def test_world_z_diagnostics_includes_hand_instances_when_available(self) -> None:
        args = demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--tracker-backend",
                "tapnextpp",
                "--tracker-query-count",
                "4",
            ]
        )
        demo.apply_demo_preset(args)
        demo.validate_args(args)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        runtime.table_c2w = np.eye(4, dtype=np.float32)
        hand_a = np.zeros((4, 4), dtype=bool)
        hand_b = np.zeros((4, 4), dtype=bool)
        hand_a[2, :] = True
        hand_b[3, :] = True

        result = runtime._build_pcd_packet_from_mask(
            self._mask_packet(hand_a_mask=hand_a, hand_b_mask=hand_b),
            rng=np.random.default_rng(0),
        )

        self.assertEqual(result.world_z_diagnostics["classes"]["hand_a"]["count"], 4)
        self.assertEqual(result.world_z_diagnostics["classes"]["hand_b"]["count"], 4)
        self.assertAlmostEqual(result.world_z_diagnostics["classes"]["hand_a"]["z_m"]["p50"], 1.0, places=6)

    def test_panel_render_mode_requires_fake_live_lossless_tracking(self) -> None:
        args = demo.build_parser().parse_args(
            [
                "--render-mode",
                "panel",
                "--input-source",
                "live",
                "--track-mode",
                "controller-object",
                "--pcd-mode",
                "masked",
                "--tracker-backend",
                "tapnextpp",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--render-mode panel requires --input-source fake-live"):
            demo.validate_args(args)

    def test_panel_hud_from_runtime_pair_uses_latest_rgb_and_paired_seq(self) -> None:
        args = self._tracker_args()
        args.input_source = demo.INPUT_SOURCE_FAKE_LIVE
        args.render_mode = "panel"
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        rgb_frame = self._frame_packet(seq=5, source_timestamp_s=105.5)
        pcd_packet = self._pcd_packet(seq=3, source_timestamp_s=103.0)
        pair = demo.PairedRenderPacket(
            seq=3,
            pcd_packet=pcd_packet,
            tracker_packet=self._tracker_packet(seq=3),
            mask_packet=self._mask_packet(seq=3),
        )

        hud = runtime._build_panel_hud(
            rgb_frame=rgb_frame,
            pair=pair,
            display_time_s=pair.pcd_packet.process_done_perf_s + 0.1,
        )

        self.assertEqual(hud.rgb_seq, 5)
        self.assertEqual(hud.paired_seq, 3)
        self.assertEqual(hud.rgb_ahead_frames, 2)
        self.assertEqual(hud.marker_count, pair.tracker_packet.marker_count)
        self.assertEqual(hud.input_time_s, 105.5)

    def test_marker_residual_audit_accepts_markers_inside_union_residual(self) -> None:
        object_residual = np.zeros((4, 4), dtype=bool)
        controller_residual = np.zeros((4, 4), dtype=bool)
        object_residual[1, 1] = True
        controller_residual[2, 3] = True

        audit = demo._audit_marker_residual_subset(
            np.array([[1.2, 0.8], [2.0, 3.0]], dtype=np.float32),
            object_residual_mask=object_residual,
            controller_residual_mask=controller_residual,
        )

        self.assertEqual(audit.checked_count, 2)
        self.assertEqual(audit.violation_count, 0)
        np.testing.assert_array_equal(audit.valid, np.array([True, True], dtype=bool))
        np.testing.assert_array_equal(audit.violation, np.array([False, False], dtype=bool))
        np.testing.assert_array_equal(audit.pixels_yx, np.array([[1, 1], [2, 3]], dtype=np.int64))

    def test_marker_residual_audit_counts_outside_nonfinite_and_out_of_bounds_markers(self) -> None:
        object_residual = np.zeros((4, 4), dtype=bool)
        controller_residual = np.zeros((4, 4), dtype=bool)
        object_residual[1, 1] = True

        audit = demo._audit_marker_residual_subset(
            np.array(
                [
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [np.nan, 1.0],
                    [4.0, 0.0],
                ],
                dtype=np.float32,
            ),
            object_residual_mask=object_residual,
            controller_residual_mask=controller_residual,
        )

        self.assertEqual(audit.checked_count, 4)
        self.assertEqual(audit.violation_count, 3)
        np.testing.assert_array_equal(audit.valid, np.array([True, False, False, False], dtype=bool))
        np.testing.assert_array_equal(audit.violation, np.array([False, True, True, True], dtype=bool))

    def test_paired_render_packet_rejects_mask_seq_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, r"mask=4"):
            demo.PairedRenderPacket(
                seq=3,
                pcd_packet=self._pcd_packet(seq=3),
                tracker_packet=self._tracker_packet(seq=3),
                mask_packet=self._mask_packet(seq=4),
            )

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

    def _tracker_residual_table_z_args(self, *, query_count: int = 4):
        args = demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--tracker-backend",
                "tapnextpp",
                "--tracker-query-count",
                str(int(query_count)),
                "--tracker-overlay-max-points",
                "0",
                "--tracker-display-scope",
                "union",
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--pcd-filter-preset",
                "original",
                "--object-filter",
                "none",
                "--controller-filter",
                "none",
                "--object-filter-cap",
                "0",
                "--controller-filter-cap",
                "0",
                "--enable-table-z-filter",
                "--table-z-filter-threshold-m",
                "0.02",
                "--table-z-filter-classes",
                "both",
            ]
        )
        demo.validate_args(args)
        return args

    def _mask_packet(
        self,
        seq: int = 0,
        *,
        hand_a_mask: np.ndarray | None = None,
        hand_b_mask: np.ndarray | None = None,
        depth_u16: np.ndarray | None = None,
    ) -> demo.MaskPacket:
        controller = np.zeros((4, 4), dtype=bool)
        obj = np.zeros((4, 4), dtype=bool)
        controller[2:, :] = True
        obj[:2, :] = True
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.full((4, 4), 1000, dtype=np.uint16) if depth_u16 is None else depth_u16
        return demo.MaskPacket(
            seq=int(seq),
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
            hand_a_mask=hand_a_mask,
            hand_b_mask=hand_b_mask,
            depth_u16=np.ascontiguousarray(depth, dtype=np.uint16),
        )

    def _frame_packet(
        self,
        seq: int = 0,
        *,
        source_timestamp_s: float | None = None,
        source_frame_index: int | None = None,
        source_step: int | None = None,
    ) -> demo.FramePacket:
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        return demo.FramePacket(
            seq=int(seq),
            color_bgr=color,
            depth_source="realsense",
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=time.perf_counter(),
            timing=demo.PipelineTiming(),
            depth_u16=np.full((4, 4), 1000, dtype=np.uint16),
            source_timestamp_s=source_timestamp_s,
            source_frame_index=source_frame_index,
            source_step=source_step,
        )

    def _pcd_packet(
        self,
        seq: int = 0,
        *,
        coordinate_frame: str = demo.COORDINATE_FRAME,
        source_timestamp_s: float | None = None,
        source_frame_index: int | None = None,
        source_step: int | None = None,
    ) -> demo.MaskedPcdPacket:
        controller_points = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
        object_points = np.array([[0.05, 0.0, 0.6]], dtype=np.float32)
        controller_colors = np.array([[255, 0, 0]], dtype=np.uint8)
        object_colors = np.array([[0, 255, 0]], dtype=np.uint8)
        now = time.perf_counter()
        return demo.MaskedPcdPacket(
            seq=int(seq),
            controller_xyz_m=controller_points,
            controller_colors_rgb_u8=controller_colors,
            object_xyz_m=object_points,
            object_colors_rgb_u8=object_colors,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0),
            receive_perf_s=now,
            process_done_perf_s=now,
            dropped_capture_frames=0,
            dropped_seg_frames=0,
            timing=demo.PipelineTiming(),
            filter_telemetry=demo.PcdFilterTelemetry(
                enabled=True,
                mode="sync",
                render_using_filtered=True,
                object_output_points=1,
                controller_output_points=1,
            ),
            coordinate_frame=coordinate_frame,
            source_timestamp_s=source_timestamp_s,
            source_frame_index=source_frame_index,
            source_step=source_step,
        )

    def _tracker_packet(self, seq: int = 0) -> demo.TrackerMarkerPacket:
        now = time.perf_counter()
        return demo.TrackerMarkerPacket(
            seq=int(seq),
            marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            marker_colors_rgb_u8=query_rainbow_colors_rgb_u8(1),
            query_rgb_u8=query_rainbow_colors_rgb_u8(1),
            query_points_yx=np.array([[1.0, 1.0]], dtype=np.float32),
            tracks_yx=np.array([[1.0, 1.0]], dtype=np.float32),
            visibility=np.ones((1,), dtype=np.float32),
            query_is_object=np.array([True], dtype=bool),
            query_is_controller=np.array([False], dtype=bool),
            receive_perf_s=now,
            process_done_perf_s=now,
            query_count=1,
            consistent_visible_count=1,
            model_ms=1.25,
            lift_ms=0.5,
            e2e_ms=2.0,
            backend="tapnextpp",
            display_scope="union",
            query_indices=np.array([0], dtype=np.int64),
        )

    def _seed_three_tracker_queries(self, runtime: demo.RealtimeMaskedEdgeTamPcdDemo) -> None:
        runtime._tracker_query_points_yx = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32)
        runtime._tracker_query_rgb_u8 = query_rainbow_colors_rgb_u8(3)
        runtime._tracker_query_is_object = np.array([True, False, False], dtype=bool)
        runtime._tracker_query_is_controller = np.array([False, True, True], dtype=bool)
        runtime._tracker_query_target_id = np.array([demo.OBJECT_ID, demo.CONTROLLER_ID, demo.CONTROLLER_ID], dtype=np.int64)
        runtime._tracker_query_controller_instance_id = np.array(
            [
                demo.QUERY_CONTROLLER_INSTANCE_NONE,
                demo.QUERY_CONTROLLER_INSTANCE_HAND_A,
                demo.QUERY_CONTROLLER_INSTANCE_HAND_B,
            ],
            dtype=np.int64,
        )

    def _install_three_query_residual_masks(
        self,
        runtime: demo.RealtimeMaskedEdgeTamPcdDemo,
        *,
        object_points: list[tuple[int, int]],
        controller_points: list[tuple[int, int]],
    ) -> None:
        object_residual = np.zeros((4, 4), dtype=bool)
        controller_residual = np.zeros((4, 4), dtype=bool)
        for y, x in object_points:
            object_residual[int(y), int(x)] = True
        for y, x in controller_points:
            controller_residual[int(y), int(x)] = True
        runtime._tracker_pcd_filter_residual_masks = lambda packet: (object_residual, controller_residual)  # type: ignore[method-assign]

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

    def test_query_rainbow_colors_are_deterministic_identity_colors(self) -> None:
        points_yx = np.array(
            [[0.0, 0.0], [3.0, 1.0], [6.0, 2.0], [9.0, 3.0], [12.0, 4.0]],
            dtype=np.float32,
        )
        first = query_rainbow_colors_from_points_yx_rgb_u8(points_yx)
        second = query_rainbow_colors_from_points_yx_rgb_u8(points_yx)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (5, 3))
        self.assertGreater(np.unique(first, axis=0).shape[0], 4)
        np.testing.assert_array_equal(first, query_rainbow_colors_from_points_yx_rgb_u8(points_yx + np.array([0.0, 99.0], dtype=np.float32)))

    def test_split_controller_hand_instances_sorts_masks_by_initial_x(self) -> None:
        right = np.zeros((5, 8), dtype=bool)
        left = np.zeros((5, 8), dtype=bool)
        right[1:4, 5:7] = True
        left[1:4, 1:3] = True

        hand_a, hand_b = demo.split_controller_hand_instances([right, left], label="human hand")

        np.testing.assert_array_equal(hand_a, left)
        np.testing.assert_array_equal(hand_b, right)

    def test_split_controller_hand_instances_splits_union_connected_components(self) -> None:
        union = np.zeros((5, 8), dtype=bool)
        union[1:4, 1:3] = True
        union[1:4, 5:7] = True

        hand_a, hand_b = demo.split_controller_hand_instances([union], label="human hand")

        self.assertEqual(int(np.count_nonzero(hand_a)), 6)
        self.assertEqual(int(np.count_nonzero(hand_b)), 6)
        self.assertLess(np.argwhere(hand_a)[:, 1].mean(), np.argwhere(hand_b)[:, 1].mean())
        np.testing.assert_array_equal(np.logical_or(hand_a, hand_b), union)

    def test_split_controller_hand_instances_fails_fast_for_one_component(self) -> None:
        one_hand = np.zeros((5, 8), dtype=bool)
        one_hand[1:4, 1:4] = True

        with self.assertRaisesRegex(RuntimeError, "requires two visible hands"):
            demo.split_controller_hand_instances([one_hand], label="human hand")

    def test_query_target_labels_and_per_target_visibility_are_instance_gated(self) -> None:
        hand_a = np.zeros((4, 6), dtype=bool)
        obj = np.zeros((4, 6), dtype=bool)
        hand_b = np.zeros((4, 6), dtype=bool)
        hand_a[:, 0:2] = True
        obj[:, 2:4] = True
        hand_b[:, 4:6] = True
        controller = np.logical_or(hand_a, hand_b)
        query_points = np.array([[1.0, 1.0], [1.0, 4.0], [1.0, 2.0]], dtype=np.float32)

        query_is_object, query_is_controller, target_id, controller_instance_id = demo._classify_query_targets_yx(
            query_points,
            object_mask=obj,
            hand_a_mask=hand_a,
            hand_b_mask=hand_b,
            controller_mask=controller,
        )

        np.testing.assert_array_equal(target_id, np.array([demo.HAND_A_ID, demo.HAND_B_ID, demo.OBJECT_ID]))
        np.testing.assert_array_equal(
            controller_instance_id,
            np.array(
                [
                    demo.QUERY_CONTROLLER_INSTANCE_HAND_A,
                    demo.QUERY_CONTROLLER_INSTANCE_HAND_B,
                    demo.QUERY_CONTROLLER_INSTANCE_NONE,
                ],
                dtype=np.int64,
            ),
        )
        np.testing.assert_array_equal(query_is_object, np.array([False, False, True]))
        np.testing.assert_array_equal(query_is_controller, np.array([True, True, False]))

        mask_packet = self._mask_packet()
        mask_packet = demo.MaskPacket(
            seq=mask_packet.seq,
            color_bgr=np.zeros((4, 6, 3), dtype=np.uint8),
            depth_source=mask_packet.depth_source,
            intrinsics=mask_packet.intrinsics,
            depth_scale_m_per_unit=mask_packet.depth_scale_m_per_unit,
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=mask_packet.process_done_perf_s,
            dropped_capture_frames=mask_packet.dropped_capture_frames,
            timing=mask_packet.timing,
            controller_mask=controller,
            object_mask=obj,
            hand_a_mask=hand_a,
            hand_b_mask=hand_b,
            depth_u16=np.full((4, 6), 1000, dtype=np.uint16),
        )
        swapped_tracks = np.array([[1.0, 4.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        visibility = demo._tracker_per_target_visibility(
            swapped_tracks,
            np.ones((3,), dtype=np.float32),
            mask_packet=mask_packet,
            query_target_id=target_id,
        )

        np.testing.assert_array_equal(visibility, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    def test_three_identity_segmentation_prompt_and_controller_union(self) -> None:
        class _FakeTensor:
            def to(self, *, device, dtype):
                _ = device, dtype
                return self

        class _FakeInputs:
            pixel_values = [_FakeTensor()]
            original_sizes = [(4, 6)]

        class _FakeProcessor:
            def __init__(self) -> None:
                self.prompt_obj_ids: list[int] | None = None
                self.prompt_masks: list[np.ndarray] | None = None

            def __call__(self, *, images, device, return_tensors):
                _ = images, device, return_tensors
                return _FakeInputs()

            def add_inputs_to_inference_session(self, *, inference_session, frame_idx, obj_ids, input_masks):
                _ = inference_session, frame_idx
                self.prompt_obj_ids = list(obj_ids)
                self.prompt_masks = [np.asarray(mask, dtype=bool) for mask in input_masks]
                return None

            def post_process_masks(self, pred_masks, *, original_sizes, binarize):
                _ = pred_masks, original_sizes, binarize
                return [
                    [
                        np.asarray(self.prompt_masks[0], dtype=bool),
                        np.asarray(self.prompt_masks[1], dtype=bool),
                        np.asarray(self.prompt_masks[2], dtype=bool),
                    ]
                ]

        class _FakeModelOutput:
            frame_idx = 0
            object_ids = [demo.HAND_A_ID, demo.OBJECT_ID, demo.HAND_B_ID]
            pred_masks = object()

        class _FakeModel:
            def __call__(self, *, inference_session, frame, frame_idx):
                _ = inference_session, frame, frame_idx
                return _FakeModelOutput()

        class _FakeTorch:
            bfloat16 = object()
            float16 = object()

            @staticmethod
            def inference_mode():
                class _Ctx:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                return _Ctx()

        args = self._tracker_args()
        args.controller_instance_mode = demo.CONTROLLER_INSTANCE_MODE_TWO_HANDS
        args.device = "cpu"
        args.dtype = "float32"
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        processor = _FakeProcessor()
        hand_a = np.zeros((4, 6), dtype=bool)
        obj = np.zeros((4, 6), dtype=bool)
        hand_b = np.zeros((4, 6), dtype=bool)
        hand_a[:, 0:2] = True
        obj[:, 2:4] = True
        hand_b[:, 4:6] = True
        frame = self._mask_packet()
        packet = runtime._run_segmentation_frame(
            hf_stream=object(),
            torch_module=_FakeTorch(),
            dtype=np.float32,
            model=_FakeModel(),
            processor=processor,
            session=object(),
            frame=demo.FramePacket(
                seq=0,
                color_bgr=np.zeros((4, 6, 3), dtype=np.uint8),
                depth_source=frame.depth_source,
                intrinsics=frame.intrinsics,
                depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
                receive_perf_s=frame.receive_perf_s,
                timing=demo.PipelineTiming(),
                depth_u16=np.full((4, 6), 1000, dtype=np.uint16),
            ),
            initial_masks=demo.InitialMaskBundle(
                controller_mask=np.logical_or(hand_a, hand_b),
                object_mask=obj,
                hand_a_mask=hand_a,
                hand_b_mask=hand_b,
            ),
            add_prompt=True,
        )

        self.assertEqual(processor.prompt_obj_ids, [demo.HAND_A_ID, demo.OBJECT_ID, demo.HAND_B_ID])
        np.testing.assert_array_equal(packet.hand_a_mask, hand_a)
        np.testing.assert_array_equal(packet.hand_b_mask, hand_b)
        np.testing.assert_array_equal(packet.object_mask, obj)
        np.testing.assert_array_equal(packet.controller_mask, np.logical_or(hand_a, hand_b))

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

    def test_tracker_hud_shows_consistent_visible_count(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        now = time.perf_counter()
        tracker_packet = demo.TrackerMarkerPacket(
            seq=0,
            marker_xyz_m=np.zeros((2, 3), dtype=np.float32),
            marker_colors_rgb_u8=np.zeros((2, 3), dtype=np.uint8),
            query_rgb_u8=np.zeros((4, 3), dtype=np.uint8),
            query_points_yx=np.zeros((4, 2), dtype=np.float32),
            tracks_yx=np.zeros((2, 2), dtype=np.float32),
            visibility=np.ones((2,), dtype=np.float32),
            query_is_object=np.array([True, False], dtype=bool),
            query_is_controller=np.array([False, True], dtype=bool),
            receive_perf_s=now,
            process_done_perf_s=now,
            query_count=4,
            consistent_visible_count=3,
            backend="tapnextpp",
            display_scope="union",
        )

        text = runtime._format_hud(
            packet=self._pcd_packet(),
            timing=demo.PipelineTiming(),
            tracker_packet=tracker_packet,
        )

        self.assertIn("consistent=3/4", text)

    def test_ordered_lossless_queue_rejects_gaps_and_backlog_overflow(self) -> None:
        queue = demo.OrderedPacketQueue[demo.MaskPacket](name="unit", max_backlog_frames=2)
        stop_event = threading.Event()

        queue.put(self._mask_packet(seq=0))
        queue.put(self._mask_packet(seq=1))
        with self.assertRaisesRegex(demo.LosslessPipelineError, "backlog exceeded"):
            queue.put(self._mask_packet(seq=2))

        self.assertEqual(queue.get(stop_event=stop_event).seq, 0)
        queue.put(self._mask_packet(seq=2))
        self.assertEqual(queue.get(stop_event=stop_event).seq, 1)
        self.assertEqual(queue.get(stop_event=stop_event).seq, 2)
        queue.close()
        self.assertIsNone(queue.get(stop_event=stop_event))

        queue.reset()
        queue.put(self._mask_packet(seq=0))
        with self.assertRaisesRegex(demo.LosslessPipelineError, "expected seq 1, got 2"):
            queue.put(self._mask_packet(seq=2))

    def test_publish_mask_packet_waits_for_tracker_queue_capacity(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime._reset_lossless_state()
        runtime.lossless_pcd_mask_queue.max_backlog_frames = 2
        runtime.lossless_tracker_mask_queue.max_backlog_frames = 1
        runtime.lossless_pcd_mask_queue.put(self._mask_packet(seq=0))
        runtime.lossless_tracker_mask_queue.put(self._mask_packet(seq=0))

        thread = threading.Thread(target=lambda: runtime._publish_mask_packet(self._mask_packet(seq=1)), daemon=True)
        thread.start()

        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        self.assertIsNone(runtime._fatal_error_snapshot())
        self.assertEqual(runtime.lossless_pcd_mask_queue.pending_count(), 1)
        self.assertEqual(runtime.lossless_tracker_mask_queue.pending_count(), 1)

        drained = runtime.lossless_tracker_mask_queue.get(stop_event=runtime.stop_event)
        self.assertIsNotNone(drained)
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertIsNone(runtime._fatal_error_snapshot())
        self.assertEqual(runtime.lossless_pcd_mask_queue.pending_count(), 2)
        self.assertEqual(runtime.lossless_tracker_mask_queue.pending_count(), 1)
        self.assertEqual(runtime.lossless_tracker_mask_queue.latest_seq(), 1)
        self.assertEqual(runtime._lossless_segmented_frames, 1)
        runtime.stop_event.set()
        runtime._close_lossless_queues()

    def test_lossless_controller_filter_budget_can_drop_below_default_min_cap(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)

        self.assertEqual(runtime.object_filter_budget.min_cap, int(args.filter_min_cap))
        self.assertLess(runtime.controller_filter_budget.min_cap, int(args.filter_min_cap))
        self.assertEqual(runtime.controller_filter_budget.min_cap, demo.DEFAULT_LOSSLESS_CONTROLLER_FILTER_MIN_CAP)

    def test_same_seq_pairer_holds_later_complete_pair_until_missing_seq_arrives(self) -> None:
        pairer = demo.SameSeqPairer(max_backlog_frames=4)

        for seq in range(10):
            pcd_result = demo.PcdBuildResult(
                packet=self._pcd_packet(seq=seq),
                depth_m=None,
                mask_packet=self._mask_packet(seq=seq),
            )
            self.assertEqual(pairer.add_pcd_result(pcd_result), [])
            pairs = pairer.add_tracker_packet(self._tracker_packet(seq=seq))
            self.assertEqual([pair.seq for pair in pairs], [seq])

        pcd10 = demo.PcdBuildResult(
            packet=self._pcd_packet(seq=10),
            depth_m=None,
            mask_packet=self._mask_packet(seq=10),
        )
        pcd11 = demo.PcdBuildResult(
            packet=self._pcd_packet(seq=11),
            depth_m=None,
            mask_packet=self._mask_packet(seq=11),
        )
        self.assertEqual(pairer.add_pcd_result(pcd10), [])
        self.assertEqual(pairer.add_pcd_result(pcd11), [])
        self.assertEqual(pairer.add_tracker_packet(self._tracker_packet(seq=11)), [])

        pairs = pairer.add_tracker_packet(self._tracker_packet(seq=10))

        self.assertEqual([pair.seq for pair in pairs], [10, 11])

    def test_same_seq_pairer_waits_for_fast_side_capacity_until_opposite_side_flushes(self) -> None:
        pairer = demo.SameSeqPairer(max_backlog_frames=2)
        stop_event = threading.Event()
        wait_results: list[bool] = []
        errors: list[BaseException] = []
        waiter_entered = threading.Event()

        self.assertEqual(
            pairer.add_pcd_result(
                demo.PcdBuildResult(
                    packet=self._pcd_packet(seq=0),
                    depth_m=None,
                    mask_packet=self._mask_packet(seq=0),
                )
            ),
            [],
        )
        self.assertEqual(
            pairer.add_pcd_result(
                demo.PcdBuildResult(
                    packet=self._pcd_packet(seq=1),
                    depth_m=None,
                    mask_packet=self._mask_packet(seq=1),
                )
            ),
            [],
        )

        def wait_for_pcd_capacity() -> None:
            waiter_entered.set()
            try:
                wait_results.append(
                    pairer.wait_for_side_capacity(
                        "pcd",
                        stop_event=stop_event,
                        timeout_s=0.01,
                    )
                )
            except BaseException as exc:  # pragma: no cover - re-raised below
                errors.append(exc)

        thread = threading.Thread(target=wait_for_pcd_capacity, daemon=True)
        thread.start()

        self.assertTrue(waiter_entered.wait(timeout=1.0))
        time.sleep(0.05)
        self.assertEqual(wait_results, [])

        pairs = pairer.add_tracker_packet(self._tracker_packet(seq=0))

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        if errors:
            raise errors[0]
        self.assertEqual([pair.seq for pair in pairs], [0])
        self.assertEqual(wait_results, [True])

    def test_lossless_pcd_worker_waits_for_pairer_capacity_when_tracker_lags(self) -> None:
        args = self._tracker_args()
        args.render_mode = "none"
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime._reset_lossless_state()
        runtime.same_seq_pairer.max_backlog_frames = 1
        runtime.lossless_pcd_mask_queue.put(self._mask_packet(seq=0))
        runtime.lossless_pcd_mask_queue.put(self._mask_packet(seq=1))
        runtime.lossless_pcd_mask_queue.close()

        def build_pcd(
            mask_packet: demo.MaskPacket,
            *,
            rng: np.random.Generator,
            require_filter_seq: bool = False,
        ) -> demo.PcdBuildResult:
            _ = rng, require_filter_seq
            return demo.PcdBuildResult(
                packet=self._pcd_packet(seq=mask_packet.seq),
                depth_m=None,
                mask_packet=mask_packet,
            )

        runtime._build_pcd_packet_from_mask = build_pcd  # type: ignore[method-assign]
        thread = threading.Thread(target=runtime._lossless_pcd_worker, daemon=True)
        thread.start()

        deadline = time.time() + 1.0
        while time.time() < deadline and runtime.same_seq_pairer.stats.pending_pcd < 1:
            time.sleep(0.01)
        self.assertEqual(runtime.same_seq_pairer.stats.pending_pcd, 1)
        time.sleep(0.05)
        self.assertIsNone(runtime._fatal_error_snapshot())
        self.assertTrue(thread.is_alive())

        acquired = runtime._lossless_pairer_lock.acquire(timeout=1.0)
        self.assertTrue(acquired)
        try:
            pairs = runtime.same_seq_pairer.add_tracker_packet(self._tracker_packet(seq=0))
            runtime._publish_pairer_outputs(pairs)
        finally:
            runtime._lossless_pairer_lock.release()

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(runtime._fatal_error_snapshot())
        self.assertEqual(runtime._lossless_pcd_results, 2)
        runtime.stop_event.set()
        runtime._close_lossless_queues()

    def test_strict_pair_rejects_mismatched_pcd_and_tracker_seq(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        pcd_packet = self._pcd_packet(seq=10)
        tracker_packet = self._tracker_packet(seq=9)
        pcd_result = demo.PcdBuildResult(
            packet=pcd_packet,
            depth_m=None,
            mask_packet=self._mask_packet(seq=10),
        )

        with self.assertRaisesRegex(ValueError, "same-seq render packet mismatch"):
            runtime._publish_strict_render_pair(pcd_result, tracker_packet)

        self.assertEqual(runtime.paired_render_slot.latest_seq(), -1)
        self.assertIsNone(runtime.paired_render_slot.get_latest_after(-1))

    def test_strict_pair_publish_updates_pcd_and_tracker_together(self) -> None:
        args = self._tracker_args()
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        render_requests: list[int] = []
        runtime._request_render_update = lambda: render_requests.append(1)
        pcd_result = demo.PcdBuildResult(
            packet=self._pcd_packet(seq=11),
            depth_m=None,
            mask_packet=self._mask_packet(seq=11),
        )
        tracker_packet = self._tracker_packet(seq=11)

        pair = runtime._publish_strict_render_pair(pcd_result, tracker_packet)

        self.assertEqual(pair.seq, 11)
        self.assertEqual(pair.pcd_packet.seq, 11)
        self.assertEqual(pair.tracker_packet.seq, 11)
        self.assertEqual(render_requests, [1])
        self.assertIs(runtime.paired_render_slot.get_latest_after(-1), pair)
        self.assertEqual(runtime.render_slot.latest_seq(), -1)
        self.assertEqual(runtime.tracker_marker_slot.latest_seq(), -1)

    def test_lossless_headless_pair_publish_does_not_fill_unconsumed_render_queue(self) -> None:
        args = self._tracker_args()
        args.render_mode = "none"
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime._reset_lossless_state()
        pcd_result = demo.PcdBuildResult(
            packet=self._pcd_packet(seq=0),
            depth_m=None,
            mask_packet=self._mask_packet(seq=0),
        )

        pair = runtime._publish_strict_render_pair(pcd_result, self._tracker_packet(seq=0))

        self.assertEqual(pair.seq, 0)
        self.assertEqual(runtime.paired_render_slot.latest_seq(), 0)
        self.assertEqual(runtime.lossless_paired_render_queue.latest_seq(), -1)
        self.assertEqual(runtime.lossless_paired_render_queue.pending_count(), 0)

    def test_slow_lossless_output_worker_does_not_block_pairer_submission(self) -> None:
        args = self._tracker_args()
        args.render_mode = "none"
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime._reset_lossless_state()
        entered_seq0_publish = threading.Event()
        release_seq0_publish = threading.Event()
        published: list[int] = []
        errors: list[BaseException] = []

        def pair_for(seq: int) -> demo.PairedBuildResult:
            return demo.PairedBuildResult(
                seq=seq,
                pcd_result=demo.PcdBuildResult(
                    packet=self._pcd_packet(seq=seq),
                    depth_m=None,
                    mask_packet=self._mask_packet(seq=seq),
                ),
                tracker_packet=self._tracker_packet(seq=seq),
            )

        def fake_publish(
            pcd_result: demo.PcdBuildResult,
            tracker_packet: demo.TrackerMarkerPacket,
        ) -> demo.PairedRenderPacket:
            seq = int(pcd_result.packet.seq)
            if seq == 0:
                entered_seq0_publish.set()
                self.assertTrue(release_seq0_publish.wait(timeout=1.0))
            published.append(seq)
            return demo.PairedRenderPacket(
                seq=seq,
                pcd_packet=pcd_result.packet,
                tracker_packet=tracker_packet,
            )

        def publish(pair: demo.PairedBuildResult) -> None:
            try:
                runtime._publish_pairer_outputs([pair])
            except BaseException as exc:  # pragma: no cover - re-raised below
                errors.append(exc)

        runtime._publish_strict_render_pair = fake_publish  # type: ignore[method-assign]

        output_thread = threading.Thread(target=runtime._lossless_pair_output_worker, daemon=True)
        output_thread.start()
        publish(pair_for(0))
        self.assertTrue(entered_seq0_publish.wait(timeout=1.0))

        acquired = runtime._lossless_pairer_lock.acquire(timeout=0.1)
        self.assertTrue(acquired)
        if acquired:
            runtime._lossless_pairer_lock.release()

        publish(pair_for(1))
        self.assertEqual(published, [])
        self.assertEqual(runtime.lossless_pair_output_queue.pending_count(), 1)

        runtime.lossless_pair_output_queue.close()
        release_seq0_publish.set()
        output_thread.join(timeout=1.0)

        self.assertFalse(output_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(published, [0, 1])
        self.assertEqual(runtime._lossless_next_publish_seq, 2)

    def test_fake_live_lossless_capture_waits_for_first_pair_before_replay_clock(self) -> None:
        class FakeRecordingSource:
            frame_count = 100
            recording_fps = 30.0
            effective_fps = 5.0

            def __init__(self) -> None:
                self.first_receive_s = time.perf_counter() - 2.0

            def source_index_for_recording_elapsed_s(self, elapsed_s: float) -> int:
                source_index = int(float(elapsed_s) * self.recording_fps)
                return max(0, min(source_index, self.frame_count - 1))

            def read_packet(
                self,
                *,
                seq: int,
                frame_index: int | None = None,
                wait_ms: float = 0.0,
            ) -> demo.FramePacket:
                _ = frame_index
                receive_s = self.first_receive_s if int(seq) == 0 else time.perf_counter()
                return demo.FramePacket(
                    seq=int(seq),
                    color_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
                    depth_source="realsense",
                    intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0),
                    depth_scale_m_per_unit=0.001,
                    receive_perf_s=receive_s,
                    timing=demo.PipelineTiming(wait_ms=float(wait_ms)),
                    depth_u16=np.ones((4, 4), dtype=np.uint16),
                )

        args = self._tracker_args()
        args.input_source = demo.INPUT_SOURCE_FAKE_LIVE
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime._reset_lossless_state()
        runtime.recording_source = FakeRecordingSource()
        published: list[int] = []
        first_published = threading.Event()

        def publish(packet: demo.FramePacket, *, record_s: float | None = None) -> None:
            _ = record_s
            published.append(int(packet.seq))
            if int(packet.seq) == 0:
                first_published.set()

        runtime._publish_capture_packet = publish  # type: ignore[method-assign]
        thread = threading.Thread(target=runtime._capture_recording_worker, daemon=True)
        thread.start()

        self.assertTrue(first_published.wait(timeout=1.0))
        runtime._recording_first_frame_segmented.set()
        time.sleep(0.25)
        try:
            self.assertEqual(published, [0])
            first_pair_ready = getattr(runtime, "_lossless_first_pair_published", None)
            self.assertIsNotNone(first_pair_ready)
            first_pair_ready.set()
            deadline = time.time() + 1.0
            while time.time() < deadline and len(published) <= 1:
                time.sleep(0.01)
            self.assertGreater(len(published), 1)
            self.assertEqual(published[1], 1)
        finally:
            runtime.stop_event.set()
            thread.join(timeout=1.0)
            runtime._close_lossless_queues()

    def test_pcd_visual_mode_with_tracker_start_threads_uses_parallel_lossless_workers(self) -> None:
        args = self._tracker_args()
        args.demo_visual_mode = demo.DEMO_VISUAL_MODE_PCD
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        started: list[str] = []

        def capture_worker() -> None:
            started.append("capture")

        def seg_worker() -> None:
            started.append("seg")

        def strict_pair_worker() -> None:
            started.append("strict-pair")
            runtime.stop_event.set()

        def tracker_worker() -> None:
            started.append("old-tracker")

        def pcd_worker() -> None:
            started.append("old-pcd")

        def lossless_pcd_worker() -> None:
            started.append("pcd")

        def lossless_tracker_worker() -> None:
            started.append("tracker")
            runtime.stop_event.set()

        def lossless_pair_output_worker() -> None:
            started.append("pair-output")

        runtime._capture_worker = capture_worker  # type: ignore[method-assign]
        runtime._seg_worker = seg_worker  # type: ignore[method-assign]
        runtime._strict_paired_worker = strict_pair_worker  # type: ignore[method-assign]
        runtime._tracker_worker = tracker_worker  # type: ignore[method-assign]
        runtime._pcd_worker = pcd_worker  # type: ignore[method-assign]
        runtime._lossless_pcd_worker = lossless_pcd_worker  # type: ignore[method-assign]
        runtime._lossless_tracker_worker = lossless_tracker_worker  # type: ignore[method-assign]
        runtime._lossless_pair_output_worker = lossless_pair_output_worker  # type: ignore[method-assign]

        runtime._start_threads()
        deadline = time.time() + 1.0
        while time.time() < deadline and "tracker" not in started:
            time.sleep(0.01)
        runtime.stop()

        self.assertIn("pcd", started)
        self.assertIn("tracker", started)
        self.assertIn("pair-output", started)
        self.assertNotIn("strict-pair", started)
        self.assertNotIn("old-tracker", started)
        self.assertNotIn("old-pcd", started)

    def test_headless_capture_writer_saves_filtered_pcd_depth_and_query_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "capture"
            writer = demo.HeadlessCaptureWriter(
                output_dir,
                metadata={
                    "width": 4,
                    "height": 4,
                    "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0},
                    "saved_pcd_source": "pt_filter_filtered",
                    "pcd_coordinate_frame": "table_world_z0",
                    "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
                },
            )
            input_packet = self._frame_packet(
                seq=0,
                source_timestamp_s=12.5,
                source_frame_index=7,
                source_step=42,
            )
            writer.write_input_frame(input_packet)
            now = time.perf_counter()
            query_points_yx = np.array([[1.0, 1.0]], dtype=np.float32)
            query_rgb_u8 = query_rainbow_colors_from_points_yx_rgb_u8(query_points_yx)
            tracker_packet = demo.TrackerMarkerPacket(
                seq=0,
                marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                marker_colors_rgb_u8=query_rgb_u8,
                query_rgb_u8=query_rgb_u8,
                query_points_yx=query_points_yx,
                tracks_yx=np.array([[1.0, 1.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                receive_perf_s=now,
                process_done_perf_s=now,
                query_count=1,
                consistent_visible_count=1,
                query_indices=np.array([0], dtype=np.int64),
                query_target_id=np.array([demo.OBJECT_ID], dtype=np.int64),
                query_controller_instance_id=np.array([demo.QUERY_CONTROLLER_INSTANCE_NONE], dtype=np.int64),
                query_all_target_id=np.array([demo.OBJECT_ID], dtype=np.int64),
                query_all_controller_instance_id=np.array([demo.QUERY_CONTROLLER_INSTANCE_NONE], dtype=np.int64),
                object_query_count=1,
                marker_pixels_yx=np.array([[1, 1]], dtype=np.int64),
                marker_residual_valid=np.array([True], dtype=bool),
                marker_residual_violation=np.array([False], dtype=bool),
                marker_residual_checked_count=1,
                marker_residual_violation_count=0,
                marker_residual_gate=demo.TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z,
                coordinate_frame="table_world_z0",
            )

            mask_packet = self._mask_packet()
            pcd_mask = np.ones((2, 2), dtype=bool)
            writer.write_tracker(tracker_packet)
            pcd_packet = replace(
                self._pcd_packet(
                    coordinate_frame="table_world_z0",
                    source_timestamp_s=12.5,
                    source_frame_index=7,
                    source_step=42,
                ),
                receive_perf_s=10.0,
                process_done_perf_s=10.2,
            )
            tracker_packet = replace(tracker_packet, receive_perf_s=10.0, process_done_perf_s=10.4)
            writer.write_pcd(
                pcd_packet,
                depth_m=np.ones((4, 4), dtype=np.float32),
                mask_packet=mask_packet,
                controller_pcd_mask=pcd_mask,
                object_pcd_mask=~pcd_mask,
                pcd_stride=2,
                pcd_mask_erode_pixels=1,
                object_pcd_mask_erode_pixels=3,
                controller_pcd_mask_erode_pixels=0,
                tracker_packet=tracker_packet,
                world_z_diagnostics={
                    "seq": 0,
                    "table_z_m": 0.0,
                    "thresholds_m": [0.005, 0.02],
                    "classes": {"object": {"count": 1}, "controller": {"count": 1}},
                },
            )

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["saved_pcd_source"], "pt_filter_filtered")
            self.assertEqual(metadata["pcd_coordinate_frame"], "table_world_z0")
            self.assertEqual(metadata["camera_to_world_c2w"][3], [0.0, 0.0, 0.0, 1.0])
            self.assertEqual(metadata["saved_mask_source"], "edgetam_binary_masks")
            self.assertEqual(metadata["saved_rgb_source"], "segmentation_color_bgr")
            self.assertTrue(metadata["panel_supported"])
            self.assertEqual(metadata["panel_sync_policy"], "left_latest_rgb_right_strict_same_seq")
            self.assertEqual(metadata["panel_backend"], "open3d_multi_viewport")
            self.assertEqual(metadata["input_rgb_timeline"], "input_frames.jsonl")
            rows = [json.loads(line) for line in (output_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
            input_rows = [
                json.loads(line)
                for line in (output_dir / "input_frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(input_rows), 1)
            self.assertEqual(input_rows[0]["seq"], 0)
            self.assertEqual(input_rows[0]["input_rgb_path"], "input_rgb/000000.png")
            self.assertEqual(input_rows[0]["source_timestamp_s"], 12.5)
            self.assertEqual(input_rows[0]["source_frame_index"], 7)
            self.assertEqual(input_rows[0]["source_step"], 42)
            self.assertTrue((output_dir / input_rows[0]["input_rgb_path"]).is_file())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["world_z_stats_path"], "world_z_stats.jsonl")

            self.assertEqual(rows[0]["source_timestamp_s"], 12.5)
            self.assertEqual(rows[0]["source_frame_index"], 7)
            self.assertEqual(rows[0]["source_step"], 42)
            z_rows = [
                json.loads(line)
                for line in (output_dir / "world_z_stats.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(z_rows[0]["seq"], 0)
            self.assertEqual(z_rows[0]["classes"]["object"]["count"], 1)
            self.assertEqual(rows[0]["filter_telemetry"]["mode"], "sync")
            self.assertTrue((output_dir / rows[0]["pcd_path"]).is_file())
            self.assertTrue((output_dir / rows[0]["ffs_depth_path"]).is_file())
            self.assertTrue((output_dir / rows[0]["rgb_path"]).is_file())
            self.assertTrue((output_dir / rows[0]["query_trajectory_path"]).is_file())
            self.assertTrue((output_dir / rows[0]["mask_path"]).is_file())
            self.assertEqual(rows[0]["controller_mask_pixels"], int(np.count_nonzero(mask_packet.controller_mask)))
            self.assertEqual(rows[0]["object_mask_pixels"], int(np.count_nonzero(mask_packet.object_mask)))
            self.assertEqual(rows[0]["hand_a_mask_pixels"], int(np.count_nonzero(mask_packet.controller_mask)))
            self.assertEqual(rows[0]["hand_b_mask_pixels"], 0)
            self.assertEqual(rows[0]["controller_pcd_mask_pixels"], int(np.count_nonzero(pcd_mask)))
            self.assertEqual(rows[0]["object_pcd_mask_pixels"], 0)
            self.assertEqual(rows[0]["hand_a_query_count"], 0)
            self.assertEqual(rows[0]["hand_b_query_count"], 0)
            self.assertEqual(rows[0]["object_query_count"], 1)
            self.assertEqual(rows[0]["marker_count"], 1)
            self.assertEqual(rows[0]["marker_residual_checked_count"], 1)
            self.assertEqual(rows[0]["marker_residual_violation_count"], 0)
            self.assertEqual(rows[0]["marker_residual_gate"], demo.TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z)
            self.assertEqual(rows[0]["filter_preset"], "pt_filter_filtered")
            self.assertAlmostEqual(rows[0]["process_done_perf_s"], 10.2)
            self.assertAlmostEqual(rows[0]["pair_process_done_perf_s"], 10.4)
            self.assertAlmostEqual(rows[0]["pipeline_latency_ms"], 400.0)
            self.assertEqual(rows[0]["pcd_mask_erode_pixels"], 1)
            self.assertEqual(rows[0]["object_pcd_mask_erode_pixels"], 3)
            self.assertEqual(rows[0]["controller_pcd_mask_erode_pixels"], 0)
            pcd = np.load(output_dir / rows[0]["pcd_path"], allow_pickle=False)
            self.assertEqual(str(pcd["saved_pcd_source"][0]), "pt_filter_filtered")
            self.assertEqual(str(pcd["coordinate_frame"][0]), "table_world_z0")
            mask_payload = np.load(output_dir / rows[0]["mask_path"], allow_pickle=False)
            np.testing.assert_array_equal(mask_payload["controller_mask"], mask_packet.controller_mask)
            np.testing.assert_array_equal(mask_payload["object_mask"], mask_packet.object_mask)
            np.testing.assert_array_equal(mask_payload["hand_a_mask"], mask_packet.controller_mask)
            np.testing.assert_array_equal(mask_payload["hand_b_mask"], np.zeros_like(mask_packet.controller_mask))
            np.testing.assert_array_equal(mask_payload["controller_pcd_mask"], pcd_mask)
            self.assertEqual(int(mask_payload["pcd_stride"][0]), 2)
            self.assertEqual(int(mask_payload["pcd_mask_erode_pixels"][0]), 1)
            self.assertEqual(int(mask_payload["object_pcd_mask_erode_pixels"][0]), 3)
            self.assertEqual(int(mask_payload["controller_pcd_mask_erode_pixels"][0]), 0)
            trajectory = np.load(output_dir / rows[0]["query_trajectory_path"], allow_pickle=False)
            self.assertEqual(str(trajectory["coordinate_frame"][0]), "table_world_z0")
            np.testing.assert_array_equal(trajectory["query_indices"], np.array([0], dtype=np.int64))
            np.testing.assert_array_equal(trajectory["query_rgb_u8"], query_rgb_u8)
            np.testing.assert_array_equal(trajectory["marker_rgb_u8"], query_rgb_u8)
            np.testing.assert_array_equal(trajectory["marker_pixels_yx"], np.array([[1, 1]], dtype=np.int64))
            np.testing.assert_array_equal(trajectory["marker_residual_valid"], np.array([True], dtype=bool))
            np.testing.assert_array_equal(trajectory["marker_residual_violation"], np.array([False], dtype=bool))
            self.assertEqual(int(trajectory["marker_residual_checked_count"][0]), 1)
            self.assertEqual(int(trajectory["marker_residual_violation_count"][0]), 0)
            self.assertEqual(
                str(trajectory["marker_residual_gate"][0]),
                demo.TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z,
            )
            np.testing.assert_array_equal(trajectory["query_target_id"], np.array([demo.OBJECT_ID], dtype=np.int64))
            np.testing.assert_array_equal(
                trajectory["query_controller_instance_id"],
                np.array([demo.QUERY_CONTROLLER_INSTANCE_NONE], dtype=np.int64),
            )
            self.assertEqual(int(trajectory["object_query_count"][0]), 1)

    def test_open3d_panel_viewport_layer_plan_separates_pcd_and_tracking(self) -> None:
        plan = demo.open3d_panel_viewport_layer_plan()

        self.assertEqual(plan["middle"]["kind"], "filtered_pcd")
        self.assertEqual(
            plan["middle"]["layers"],
            [demo.GEOMETRY_CONTROLLER, demo.GEOMETRY_OBJECT],
        )
        self.assertEqual(plan["right"]["kind"], "filtered_pcd_with_tracking")
        self.assertEqual(
            plan["right"]["layers"],
            [
                demo.GEOMETRY_CONTROLLER,
                demo.GEOMETRY_OBJECT,
                demo.GEOMETRY_TRACKER_OBJECT,
                demo.GEOMETRY_TRACKER_CONTROLLER,
            ],
        )

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
        self.assertEqual(packet.consistent_visible_count, 4)
        self.assertGreater(packet.marker_count, 0)
        self.assertLessEqual(packet.marker_count, 4)
        self.assertEqual(packet.marker_colors_rgb_u8.shape, (packet.marker_count, 3))
        self.assertEqual(packet.query_rgb_u8.shape, (packet.query_count, 3))
        self.assertFalse(np.all(packet.marker_colors_rgb_u8 == np.array([255, 0, 0], dtype=np.uint8)))
        np.testing.assert_array_equal(packet.marker_colors_rgb_u8, packet.query_rgb_u8[packet.query_indices])
        self.assertTrue(np.all(packet.marker_xyz_m[:, 2] > 0.0))

    def test_tracker_queries_use_pcd_filter_residual_pixels_with_stride(self) -> None:
        args = demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--tracker-backend",
                "tapnextpp",
                "--tracker-query-count",
                "4",
                "--pcd-stride",
                "2",
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--pcd-filter-preset",
                "original",
                "--object-filter",
                "none",
                "--controller-filter",
                "none",
                "--object-filter-cap",
                "0",
                "--controller-filter-cap",
                "0",
            ]
        )
        demo.validate_args(args)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        adapter = _FakeTapNextAdapter()

        query_points = runtime._ensure_tracker_queries(self._mask_packet(), adapter)

        self.assertIsNotNone(query_points)
        assert query_points is not None
        expected = {(0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0)}
        self.assertEqual({tuple(point) for point in query_points.tolist()}, expected)
        np.testing.assert_array_equal(adapter.query_points_yx, query_points)

    def test_tracker_residual_masks_apply_table_z_filter(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=2)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        runtime.table_c2w = np.eye(4, dtype=np.float32)
        runtime.table_c2w[2, 3] = -1.1
        depth = np.full((4, 4), 1100, dtype=np.uint16)
        depth[0, 0] = 1000
        depth[2, 0] = 1000

        object_residual, controller_residual = runtime._tracker_pcd_filter_residual_masks(
            self._mask_packet(depth_u16=depth)
        )

        expected_object = np.zeros((4, 4), dtype=bool)
        expected_object[0, 0] = True
        expected_controller = np.zeros((4, 4), dtype=bool)
        expected_controller[2, 0] = True
        np.testing.assert_array_equal(object_residual, expected_object)
        np.testing.assert_array_equal(controller_residual, expected_controller)

    def test_tracker_query_initialization_fails_when_table_z_leaves_too_few_candidates(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=3)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        runtime.table_c2w = np.eye(4, dtype=np.float32)
        runtime.table_c2w[2, 3] = -1.1
        depth = np.full((4, 4), 1100, dtype=np.uint16)
        depth[0, 0] = 1000
        depth[2, 0] = 1000

        with self.assertRaisesRegex(RuntimeError, "not enough residual query candidates"):
            runtime._ensure_tracker_queries(self._mask_packet(depth_u16=depth), _FakeTapNextAdapter())

    def test_tracker_marker_display_hides_tracks_outside_current_residual_class_masks(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=2)
        args.enable_table_z_filter = False
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime._tracker_query_points_yx = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        runtime._tracker_query_rgb_u8 = query_rainbow_colors_rgb_u8(2)
        runtime._tracker_query_is_object = np.array([True, False], dtype=bool)
        runtime._tracker_query_is_controller = np.array([False, True], dtype=bool)
        runtime._tracker_query_target_id = np.array([demo.OBJECT_ID, demo.CONTROLLER_ID], dtype=np.int64)
        runtime._tracker_query_controller_instance_id = np.array(
            [demo.QUERY_CONTROLLER_INSTANCE_NONE, demo.QUERY_CONTROLLER_INSTANCE_HAND_A],
            dtype=np.int64,
        )
        object_residual = np.zeros((4, 4), dtype=bool)
        object_residual[0, 0] = True
        controller_residual = np.zeros((4, 4), dtype=bool)
        controller_residual[2, 0] = True
        runtime._tracker_pcd_filter_residual_masks = lambda packet: (object_residual, controller_residual)  # type: ignore[method-assign]

        packet = runtime._build_tracker_marker_packet(
            self._mask_packet(),
            _StaticTrackingAdapter(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)),
        )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.marker_count, 1)
        np.testing.assert_array_equal(packet.query_indices, np.array([1], dtype=np.int64))
        self.assertEqual(packet.marker_residual_checked_count, 1)
        self.assertEqual(packet.marker_residual_violation_count, 0)
        self.assertEqual(packet.marker_residual_gate, demo.TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z)
        np.testing.assert_array_equal(packet.marker_residual_valid, np.array([True], dtype=bool))
        np.testing.assert_array_equal(packet.marker_residual_violation, np.array([False], dtype=bool))
        np.testing.assert_array_equal(packet.marker_pixels_yx, np.array([[2, 0]], dtype=np.int64))
        self.assertEqual(packet.hand_a_query_count, 1)
        self.assertEqual(packet.object_query_count, 0)

    def test_tracker_marker_retirement_keeps_filtered_query_hidden_after_it_passes_again(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=3)
        args.enable_table_z_filter = False
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        self._seed_three_tracker_queries(runtime)
        self._install_three_query_residual_masks(
            runtime,
            object_points=[(0, 0)],
            controller_points=[(2, 0), (2, 1)],
        )

        first = runtime._build_tracker_marker_packet(
            self._mask_packet(),
            _StaticTrackingAdapter(np.array([[0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32)),
        )
        second = runtime._build_tracker_marker_packet(
            self._mask_packet(seq=1),
            _StaticTrackingAdapter(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32)),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        np.testing.assert_array_equal(first.query_alive_mask, np.array([False, True, True], dtype=bool))
        np.testing.assert_array_equal(second.query_alive_mask, np.array([False, True, True], dtype=bool))
        np.testing.assert_array_equal(second.query_indices, np.array([1, 2], dtype=np.int64))
        self.assertEqual(second.marker_count, 2)
        self.assertEqual(second.remaining_query_count, 2)
        self.assertEqual(second.retired_query_count, 1)
        self.assertEqual(second.remaining_object_query_count, 0)
        self.assertEqual(second.remaining_controller_query_count, 2)
        self.assertEqual(second.remaining_hand_a_query_count, 1)
        self.assertEqual(second.remaining_hand_b_query_count, 1)

    def test_tracker_marker_retirement_can_be_disabled_for_old_per_frame_gate_behavior(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=3)
        args.enable_table_z_filter = False
        args.tracker_retire_filtered_markers = False
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        self._seed_three_tracker_queries(runtime)
        self._install_three_query_residual_masks(
            runtime,
            object_points=[(0, 0)],
            controller_points=[(2, 0), (2, 1)],
        )

        first = runtime._build_tracker_marker_packet(
            self._mask_packet(),
            _StaticTrackingAdapter(np.array([[0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32)),
        )
        second = runtime._build_tracker_marker_packet(
            self._mask_packet(seq=1),
            _StaticTrackingAdapter(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32)),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        np.testing.assert_array_equal(first.query_alive_mask, np.array([True, True, True], dtype=bool))
        np.testing.assert_array_equal(second.query_indices, np.array([0, 1, 2], dtype=np.int64))
        self.assertEqual(second.marker_count, 3)
        self.assertEqual(second.remaining_query_count, 3)
        self.assertEqual(second.retired_query_count, 0)

    def test_tracker_marker_retirement_ignores_tracker_visibility_drop_and_overlay_cap(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=3)
        args.enable_table_z_filter = False
        args.tracker_overlay_max_points = 1
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        self._seed_three_tracker_queries(runtime)
        self._install_three_query_residual_masks(
            runtime,
            object_points=[(0, 0)],
            controller_points=[(2, 0), (2, 1)],
        )

        first = runtime._build_tracker_marker_packet(
            self._mask_packet(),
            _StaticTrackingAdapter(
                np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32),
                visibility=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            ),
        )
        second = runtime._build_tracker_marker_packet(
            self._mask_packet(seq=1),
            _StaticTrackingAdapter(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32)),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(first.marker_count, 1)
        self.assertEqual(second.marker_count, 1)
        np.testing.assert_array_equal(second.query_alive_mask, np.array([True, True, True], dtype=bool))
        self.assertEqual(second.remaining_query_count, 3)
        self.assertEqual(second.retired_query_count, 0)

    def test_tracker_marker_display_hides_tracks_removed_by_table_z(self) -> None:
        args = self._tracker_residual_table_z_args(query_count=2)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)
        runtime.table_c2w = np.eye(4, dtype=np.float32)
        runtime.table_c2w[2, 3] = -1.1
        runtime._tracker_query_points_yx = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        runtime._tracker_query_rgb_u8 = query_rainbow_colors_rgb_u8(2)
        runtime._tracker_query_is_object = np.array([True, False], dtype=bool)
        runtime._tracker_query_is_controller = np.array([False, True], dtype=bool)
        runtime._tracker_query_target_id = np.array([demo.OBJECT_ID, demo.CONTROLLER_ID], dtype=np.int64)
        runtime._tracker_query_controller_instance_id = np.array(
            [demo.QUERY_CONTROLLER_INSTANCE_NONE, demo.QUERY_CONTROLLER_INSTANCE_HAND_A],
            dtype=np.int64,
        )
        depth = np.full((4, 4), 1100, dtype=np.uint16)
        depth[0, 0] = 1000
        depth[2, 0] = 1000

        packet = runtime._build_tracker_marker_packet(
            self._mask_packet(depth_u16=depth),
            _StaticTrackingAdapter(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)),
        )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.marker_count, 1)
        np.testing.assert_array_equal(packet.query_indices, np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(packet.marker_colors_rgb_u8, packet.query_rgb_u8[packet.query_indices])
        self.assertEqual(packet.marker_residual_checked_count, 1)
        self.assertEqual(packet.marker_residual_violation_count, 0)
        np.testing.assert_array_equal(packet.marker_residual_valid, np.array([True], dtype=bool))
        np.testing.assert_array_equal(packet.marker_residual_violation, np.array([False], dtype=bool))
        self.assertEqual(packet.hand_a_query_count, 1)
        self.assertEqual(packet.object_query_count, 0)
        self.assertEqual(packet.query_count, 2)

    def test_headless_writer_saves_tracker_alive_mask_and_remaining_counts(self) -> None:
        now = time.perf_counter()
        packet = demo.TrackerMarkerPacket(
            seq=7,
            marker_xyz_m=np.zeros((1, 3), dtype=np.float32),
            marker_colors_rgb_u8=query_rainbow_colors_rgb_u8(1),
            query_rgb_u8=query_rainbow_colors_rgb_u8(3),
            query_points_yx=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32),
            tracks_yx=np.array([[2.0, 0.0]], dtype=np.float32),
            visibility=np.ones((1,), dtype=np.float32),
            query_is_object=np.array([False], dtype=bool),
            query_is_controller=np.array([True], dtype=bool),
            receive_perf_s=now,
            process_done_perf_s=now,
            query_count=3,
            query_indices=np.array([1], dtype=np.int64),
            query_target_id=np.array([demo.CONTROLLER_ID], dtype=np.int64),
            query_controller_instance_id=np.array([demo.QUERY_CONTROLLER_INSTANCE_HAND_A], dtype=np.int64),
            query_all_target_id=np.array([demo.OBJECT_ID, demo.CONTROLLER_ID, demo.CONTROLLER_ID], dtype=np.int64),
            query_all_controller_instance_id=np.array(
                [
                    demo.QUERY_CONTROLLER_INSTANCE_NONE,
                    demo.QUERY_CONTROLLER_INSTANCE_HAND_A,
                    demo.QUERY_CONTROLLER_INSTANCE_HAND_B,
                ],
                dtype=np.int64,
            ),
            query_alive_mask=np.array([False, True, True], dtype=bool),
            remaining_query_count=2,
            remaining_object_query_count=0,
            remaining_controller_query_count=2,
            remaining_hand_a_query_count=1,
            remaining_hand_b_query_count=1,
            retired_query_count=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            writer = demo.HeadlessCaptureWriter(tmp, metadata={"saved_pcd_source": "none_filtered"})
            writer.write_tracker(packet)
            payload = np.load(Path(tmp) / "query_trajectory" / "000007.npz", allow_pickle=False)

            np.testing.assert_array_equal(payload["query_alive_mask"], np.array([False, True, True], dtype=bool))
            self.assertEqual(int(payload["remaining_query_count"][0]), 2)
            self.assertEqual(int(payload["remaining_object_query_count"][0]), 0)
            self.assertEqual(int(payload["remaining_controller_query_count"][0]), 2)
            self.assertEqual(int(payload["remaining_hand_a_query_count"][0]), 1)
            self.assertEqual(int(payload["remaining_hand_b_query_count"][0]), 1)
            self.assertEqual(int(payload["retired_query_count"][0]), 1)

    def test_tracker_query_initialization_fails_when_residual_candidates_are_too_few(self) -> None:
        args = demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--tracker-backend",
                "tapnextpp",
                "--tracker-query-count",
                "5",
                "--pcd-stride",
                "2",
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--pcd-filter-preset",
                "original",
                "--object-filter",
                "none",
                "--controller-filter",
                "none",
                "--object-filter-cap",
                "0",
                "--controller-filter-cap",
                "0",
            ]
        )
        demo.validate_args(args)
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        runtime.ray_x = np.zeros((4, 4), dtype=np.float32)
        runtime.ray_y = np.zeros((4, 4), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "not enough residual query candidates"):
            runtime._ensure_tracker_queries(self._mask_packet(), _FakeTapNextAdapter())

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


def __getattr__(name: str) -> object:
    if name == "RealtimeMaskedEdgeTamPcdTest":
        return SingleDemoTapNextOverlayTest
    raise AttributeError(name)


if __name__ == "__main__":
    unittest.main()
