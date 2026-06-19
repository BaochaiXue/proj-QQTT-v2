from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.harness.diagnostics.demo.render_demo32_headless_capture import (
    PANEL_MODE_SIDE_BY_SIDE,
    TRACKING_BACKGROUND_MASK_RGB,
    TRACKING_BACKGROUND_MASK_TARGET_UNION,
    _apply_tracking_background_mask,
    _project_points,
    _read_target_union_mask,
    render_capture_to_video,
    render_table_z_filter_overlay_sweep,
)


class Demo32HeadlessRenderHelperTest(unittest.TestCase):
    def _write_minimal_tracking_capture(
        self,
        capture_dir: Path,
        *,
        metadata_extra: dict[str, object] | None = None,
        row_extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        (capture_dir / "pcd").mkdir(parents=True)
        (capture_dir / "ffs_depth").mkdir()
        (capture_dir / "rgb").mkdir()
        (capture_dir / "query_trajectory").mkdir()
        metadata = {
            "width": 8,
            "height": 6,
            "saved_pcd_source": "enhanced_pt_filtered",
            "intrinsics": {"fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 3.0},
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.savez(
            capture_dir / "pcd" / "000000.npz",
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
        )
        np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((6, 8), dtype=np.float32))
        Image.fromarray(np.full((6, 8, 3), 50, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
        np.savez(
            capture_dir / "query_trajectory" / "000000.npz",
            tracks_yx=np.empty((0, 2), dtype=np.float32),
            visibility=np.empty((0,), dtype=np.float32),
            query_indices=np.empty((0,), dtype=np.int64),
        )
        row = {
            "seq": 0,
            "pcd_path": "pcd/000000.npz",
            "ffs_depth_path": "ffs_depth/000000.npy",
            "rgb_path": "rgb/000000.png",
            "query_trajectory_path": "query_trajectory/000000.npz",
        }
        if row_extra:
            row.update(row_extra)
        (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return row

    def test_apply_tracking_background_mask_blacks_pixels_outside_union(self) -> None:
        image = np.full((4, 5, 3), 80, dtype=np.uint8)
        image[1, 2] = np.array([10, 20, 30], dtype=np.uint8)
        mask = np.zeros((4, 5), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True

        kept = _apply_tracking_background_mask(image, mask)

        self.assertEqual(kept, 2)
        np.testing.assert_array_equal(image[1, 2], np.array([10, 20, 30], dtype=np.uint8))
        np.testing.assert_array_equal(image[3, 4], np.array([80, 80, 80], dtype=np.uint8))
        np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 0], dtype=np.uint8))

    def test_read_target_union_mask_uses_object_or_controller_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            controller_mask = np.zeros((4, 5), dtype=bool)
            object_mask = np.zeros((4, 5), dtype=bool)
            controller_mask[1, 2] = True
            object_mask[3, 4] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            frame = {"seq": 0, "mask_path": "masks/000000.npz"}

            union = _read_target_union_mask(capture_dir=capture_dir, frame=frame, width=5, height=4)

            expected = np.logical_or(controller_mask, object_mask)
            np.testing.assert_array_equal(union, expected)

    def test_tracking_target_union_requires_mask_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            self._write_minimal_tracking_capture(capture_dir)

            with self.assertRaisesRegex(RuntimeError, "requires mask_path"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

    def test_tracking_target_union_requires_existing_mask_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            self._write_minimal_tracking_capture(capture_dir, row_extra={"mask_path": "masks/000000.npz"})

            with self.assertRaisesRegex(RuntimeError, "mask file missing"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

    def test_tracking_target_union_requires_object_and_controller_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(capture_dir, row_extra={"mask_path": "masks/000000.npz"})
            np.savez(capture_dir / "masks" / "000000.npz", object_mask=np.zeros((6, 8), dtype=bool))

            with self.assertRaisesRegex(RuntimeError, "controller_mask"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

    def test_tracking_target_union_rejects_wrong_mask_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(capture_dir, row_extra={"mask_path": "masks/000000.npz"})
            np.savez(
                capture_dir / "masks" / "000000.npz",
                object_mask=np.zeros((5, 8), dtype=bool),
                controller_mask=np.zeros((5, 8), dtype=bool),
            )

            with self.assertRaisesRegex(RuntimeError, "does not match render shape"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

    def test_render_synthetic_capture_to_video_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5], [0.05, 0.0, 0.6]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32], [32, 255, 255]], dtype=np.uint8),
                query_rgb_u8=np.array([[255, 32, 32], [32, 255, 255]], dtype=np.uint8),
                tracks_yx=np.array([[12.0, 16.0], [12.0, 18.0]], dtype=np.float32),
                visibility=np.ones((2,), dtype=np.float32),
                query_indices=np.array([0, 1], dtype=np.int64),
                query_is_object=np.array([False, True], dtype=bool),
                query_is_controller=np.array([True, False], dtype=bool),
                query_controller_instance_id=np.array([1, 0], dtype=np.int64),
                query_count=np.array([2], dtype=np.int64),
                marker_pixels_yx=np.array([[12, 16], [12, 18]], dtype=np.int64),
                marker_residual_valid=np.array([True, True], dtype=bool),
                marker_residual_violation=np.array([False, False], dtype=bool),
                marker_residual_checked_count=np.array([2], dtype=np.int64),
                marker_residual_violation_count=np.array([0], dtype=np.int64),
                marker_residual_gate=np.array(["pcd_filter_residual_table_z"]),
            )
            (capture_dir / "masks").mkdir()
            controller_mask = np.zeros((24, 32), dtype=bool)
            object_mask = np.zeros((24, 32), dtype=bool)
            controller_mask[11:14, 15:18] = True
            object_mask[11:14, 17:20] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
                "marker_residual_checked_count": 2,
                "marker_residual_violation_count": 0,
                "marker_residual_gate": "pcd_filter_residual_table_z",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            output = capture_dir / "video.mp4"
            summary = render_capture_to_video(capture_dir=capture_dir, output=output, fps=30.0)

            self.assertTrue(output.is_file())
            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["saved_pcd_source"], "enhanced_pt_filtered")
            self.assertEqual(summary["query_overlay"], "phystwin_rgb_current_points_only")
            self.assertEqual(summary["query_color_mode"], "phystwin_rainbow_identity")
            self.assertEqual(summary["query_match_policy"], "exact_same_seq_only")
            self.assertEqual(summary["missing_query_frames"], 0)
            self.assertEqual(summary["rendered_counts"][0]["controller_points"], 0)
            self.assertEqual(summary["rendered_counts"][0]["object_points"], 0)
            self.assertEqual(summary["rendered_counts"][0]["query_controller_points"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_object_points"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_hand_a_points"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_hand_b_points"], 0)
            self.assertEqual(summary["query_count_totals"]["hand_a"], 1)
            self.assertEqual(summary["query_count_totals"]["hand_b"], 0)
            self.assertEqual(summary["query_count_totals"]["object"], 1)
            self.assertEqual(summary["tracking_marker_residual_checked_total"], 2)
            self.assertEqual(summary["tracking_marker_residual_violation_total"], 0)
            self.assertEqual(summary["tracking_marker_residual_violation_frames"], 0)
            self.assertEqual(summary["tracking_marker_residual_audit_missing_frames"], 0)
            self.assertTrue(summary["tracking_marker_residual_target_met"])
            self.assertEqual(summary["rendered_counts"][0]["marker_residual_checked_count"], 2)
            self.assertEqual(summary["rendered_counts"][0]["marker_residual_violation_count"], 0)
            self.assertTrue((capture_dir / "video.render_summary.json").is_file())
            self.assertEqual(summary["tracking_background_mask"], TRACKING_BACKGROUND_MASK_TARGET_UNION)
            self.assertEqual(summary["tracking_background_mask_source"], "object_mask|controller_mask")
            self.assertEqual(
                summary["rendered_counts"][0]["tracking_background_mask_pixels"],
                int(np.count_nonzero(np.logical_or(controller_mask, object_mask))),
            )
            self.assertEqual(
                summary["tracking_background_mask_pixel_total"],
                int(np.count_nonzero(np.logical_or(controller_mask, object_mask))),
            )

    def test_tracking_marker_residual_summary_reports_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 8,
                "height": 6,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 3.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.empty((0, 3), dtype=np.float32),
                controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((6, 8), dtype=np.float32))
            Image.fromarray(np.full((6, 8, 3), 50, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                tracks_yx=np.array([[2.0, 3.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                marker_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                marker_residual_checked_count=np.array([1], dtype=np.int64),
                marker_residual_violation_count=np.array([1], dtype=np.int64),
                marker_residual_gate=np.array(["pcd_filter_residual_table_z"]),
            )
            mask = np.ones((6, 8), dtype=bool)
            np.savez(capture_dir / "masks" / "000000.npz", object_mask=mask, controller_mask=np.zeros_like(mask))
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
                "marker_residual_checked_count": 1,
                "marker_residual_violation_count": 1,
                "marker_residual_gate": "pcd_filter_residual_table_z",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

            self.assertEqual(summary["tracking_marker_residual_checked_total"], 1)
            self.assertEqual(summary["tracking_marker_residual_violation_total"], 1)
            self.assertEqual(summary["tracking_marker_residual_violation_frames"], 1)
            self.assertEqual(summary["tracking_marker_residual_audit_missing_frames"], 0)
            self.assertFalse(summary["tracking_marker_residual_target_met"])

    def test_tracking_marker_residual_summary_treats_missing_audit_as_unproven(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(
                capture_dir,
                row_extra={"mask_path": "masks/000000.npz"},
            )
            mask = np.ones((6, 8), dtype=bool)
            np.savez(capture_dir / "masks" / "000000.npz", object_mask=mask, controller_mask=np.zeros_like(mask))

            summary = render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

            self.assertEqual(summary["tracking_marker_residual_checked_total"], 0)
            self.assertEqual(summary["tracking_marker_residual_violation_total"], 0)
            self.assertEqual(summary["tracking_marker_residual_violation_frames"], 0)
            self.assertEqual(summary["tracking_marker_residual_audit_missing_frames"], 1)
            self.assertFalse(summary["tracking_marker_residual_target_met"])

    def test_tracking_marker_residual_summary_treats_wrong_gate_as_unproven(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(
                capture_dir,
                row_extra={
                    "mask_path": "masks/000000.npz",
                    "marker_count": 1,
                    "marker_residual_checked_count": 1,
                    "marker_residual_violation_count": 0,
                    "marker_residual_gate": "target_mask_depth",
                },
            )
            mask = np.ones((6, 8), dtype=bool)
            np.savez(capture_dir / "masks" / "000000.npz", object_mask=mask, controller_mask=np.zeros_like(mask))

            summary = render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

            self.assertEqual(summary["tracking_marker_residual_audit_missing_frames"], 1)
            self.assertFalse(summary["tracking_marker_residual_target_met"])

    def test_tracking_marker_residual_summary_treats_partial_audit_as_unproven(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(
                capture_dir,
                row_extra={
                    "mask_path": "masks/000000.npz",
                    "marker_count": 1,
                    "marker_residual_checked_count": 0,
                    "marker_residual_violation_count": 0,
                    "marker_residual_gate": "pcd_filter_residual_table_z",
                },
            )
            mask = np.ones((6, 8), dtype=bool)
            np.savez(capture_dir / "masks" / "000000.npz", object_mask=mask, controller_mask=np.zeros_like(mask))

            summary = render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

            self.assertEqual(summary["tracking_marker_residual_audit_missing_frames"], 1)
            self.assertFalse(summary["tracking_marker_residual_target_met"])

    def test_render_side_by_side_panel_prefers_input_rgb_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "input_rgb").mkdir()
            (capture_dir / "custom_input_rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "pcd_filter_preset": "enhanced-pt",
                "replay_fps": 5,
                "startup_hold_s": 2,
                "input_rgb_timeline": "input_frames.jsonl",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            Image.fromarray(np.full((24, 32, 3), 80, dtype=np.uint8)).save(capture_dir / "input_rgb" / "000000.png")
            Image.fromarray(np.full((24, 32, 3), 160, dtype=np.uint8)).save(capture_dir / "input_rgb" / "000002.png")
            Image.fromarray(np.full((24, 32, 3), 220, dtype=np.uint8)).save(
                capture_dir / "custom_input_rgb" / "chosen.png"
            )
            Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32]], dtype=np.uint8),
                tracks_yx=np.array([[12.0, 16.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([False], dtype=bool),
                query_is_controller=np.array([True], dtype=bool),
                query_controller_instance_id=np.array([1], dtype=np.int64),
                query_count=np.array([1], dtype=np.int64),
            )
            controller_mask = np.zeros((24, 32), dtype=bool)
            object_mask = np.zeros((24, 32), dtype=bool)
            controller_mask[11:14, 15:18] = True
            object_mask[11:14, 17:20] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            input_rows = [
                {"seq": 0, "receive_perf_s": 10.0},
                {
                    "seq": 2,
                    "receive_perf_s": 10.4,
                    "source_timestamp_s": 123.456,
                    "input_rgb_path": "custom_input_rgb/chosen.png",
                },
            ]
            (capture_dir / "input_frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in input_rows) + "\n",
                encoding="utf-8",
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
                "receive_perf_s": 10.0,
                "process_done_perf_s": 10.4,
                "pipeline_latency_ms": 400,
                "filter_preset": "enhanced-pt",
                "marker_count": 1,
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            output = capture_dir / "video.mp4"
            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=output,
                fps=30.0,
                panel_mode=PANEL_MODE_SIDE_BY_SIDE,
            )

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".panel_summary.json").is_file())
            self.assertEqual(summary["panel_mode"], PANEL_MODE_SIDE_BY_SIDE)
            self.assertEqual(summary["left_rgb_policy"], "latest_input_rgb")
            self.assertEqual(summary["sync_policy"], "latest_receive_perf_s_lte_pair_process_done_perf_s")
            self.assertEqual(summary["missing_rgb_frames"], 0)
            self.assertEqual(summary["rendered_counts"][0]["rgb_seq"], 2)
            self.assertEqual(summary["rendered_counts"][0]["paired_seq"], 0)
            self.assertEqual(summary["rendered_counts"][0]["rgb_ahead_frames"], 2)
            self.assertAlmostEqual(summary["rendered_counts"][0]["input_time_s"], 123.456)
            self.assertEqual(summary["rendered_counts"][0]["input_rgb_source_path"], "custom_input_rgb/chosen.png")
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 1)

    def test_render_side_by_side_panel_reports_paired_fallback_without_input_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.empty((0, 3), dtype=np.float32),
                controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                tracks_yx=np.array([[12.0, 16.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                query_count=np.array([1], dtype=np.int64),
            )
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=np.zeros((24, 32), dtype=bool),
                object_mask=np.zeros((24, 32), dtype=bool),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
                "source_timestamp_s": 45.25,
                "receive_perf_s": 99.5,
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                panel_mode=PANEL_MODE_SIDE_BY_SIDE,
            )

            self.assertEqual(summary["input_rgb_frame_count"], 0)
            self.assertEqual(summary["left_rgb_policy"], "same_seq_fallback")
            self.assertEqual(summary["sync_policy"], "paired_seq_fallback")
            self.assertEqual(summary["missing_rgb_frames"], 1)
            self.assertEqual(summary["rendered_counts"][0]["rgb_seq"], 0)
            self.assertEqual(summary["rendered_counts"][0]["paired_seq"], 0)
            self.assertAlmostEqual(summary["rendered_counts"][0]["input_time_s"], 45.25)

    def test_render_side_by_side_panel_does_not_select_future_input_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            (capture_dir / "input_rgb").mkdir()
            self._write_minimal_tracking_capture(
                capture_dir,
                metadata_extra={"input_rgb_timeline": "input_frames.jsonl"},
                row_extra={
                    "mask_path": "masks/000000.npz",
                    "source_timestamp_s": 45.25,
                    "receive_perf_s": 9.5,
                    "process_done_perf_s": 10.0,
                },
            )
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=np.zeros((6, 8), dtype=bool),
                object_mask=np.zeros((6, 8), dtype=bool),
            )
            Image.fromarray(np.full((6, 8, 3), 220, dtype=np.uint8)).save(
                capture_dir / "input_rgb" / "000007.png"
            )
            input_rows = [
                {
                    "seq": 7,
                    "receive_perf_s": 10.25,
                    "source_timestamp_s": 777.0,
                },
            ]
            (capture_dir / "input_frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in input_rows) + "\n",
                encoding="utf-8",
            )

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                panel_mode=PANEL_MODE_SIDE_BY_SIDE,
            )

            self.assertEqual(summary["input_rgb_frame_count"], 1)
            self.assertEqual(summary["left_rgb_policy"], "latest_input_rgb")
            self.assertEqual(summary["sync_policy"], "latest_receive_perf_s_lte_pair_process_done_perf_s")
            self.assertEqual(summary["missing_rgb_frames"], 1)
            self.assertEqual(summary["rendered_counts"][0]["rgb_seq"], 0)
            self.assertEqual(summary["rendered_counts"][0]["paired_seq"], 0)
            self.assertEqual(summary["rendered_counts"][0]["rgb_ahead_frames"], 0)
            self.assertAlmostEqual(summary["rendered_counts"][0]["input_time_s"], 45.25)
            self.assertEqual(summary["rendered_counts"][0]["input_rgb_source_path"], "rgb/000000.png")

    def test_render_side_by_side_panel_uses_strict_pair_completion_for_input_rgb_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            (capture_dir / "input_rgb").mkdir()
            self._write_minimal_tracking_capture(
                capture_dir,
                metadata_extra={"input_rgb_timeline": "input_frames.jsonl"},
                row_extra={
                    "mask_path": "masks/000000.npz",
                    "receive_perf_s": 10.0,
                    "process_done_perf_s": 10.2,
                    "pair_process_done_perf_s": 10.4,
                    "pipeline_latency_ms": 400.0,
                },
            )
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=np.zeros((6, 8), dtype=bool),
                object_mask=np.zeros((6, 8), dtype=bool),
            )
            for seq, value in [(0, 80), (1, 160), (2, 220)]:
                Image.fromarray(np.full((6, 8, 3), value, dtype=np.uint8)).save(
                    capture_dir / "input_rgb" / f"{seq:06d}.png"
                )
            input_rows = [
                {"seq": 0, "receive_perf_s": 10.1, "source_timestamp_s": 100.0},
                {"seq": 1, "receive_perf_s": 10.3, "source_timestamp_s": 101.0},
                {"seq": 2, "receive_perf_s": 10.5, "source_timestamp_s": 102.0},
            ]
            (capture_dir / "input_frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in input_rows) + "\n",
                encoding="utf-8",
            )

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                panel_mode=PANEL_MODE_SIDE_BY_SIDE,
            )

            rendered = summary["rendered_counts"][0]
            self.assertEqual(summary["sync_policy"], "latest_receive_perf_s_lte_pair_process_done_perf_s")
            self.assertEqual(rendered["rgb_seq"], 1)
            self.assertEqual(rendered["paired_seq"], 0)
            self.assertEqual(rendered["rgb_ahead_frames"], 1)
            self.assertAlmostEqual(rendered["input_time_s"], 101.0)
            self.assertAlmostEqual(rendered["pipeline_latency_ms"], 400.0)
            self.assertAlmostEqual(rendered["display_latency_ms"], 400.0)

    def test_side_by_side_pcd_visual_mode_reports_tracking_panel_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                tracks_yx=np.array([[12.0, 16.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([False], dtype=bool),
                query_is_controller=np.array([True], dtype=bool),
                query_controller_instance_id=np.array([1], dtype=np.int64),
                query_count=np.array([1], dtype=np.int64),
            )
            controller_mask = np.zeros((24, 32), dtype=bool)
            object_mask = np.zeros((24, 32), dtype=bool)
            controller_mask[11:14, 15:18] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                demo_visual_mode="pcd",
                panel_mode=PANEL_MODE_SIDE_BY_SIDE,
            )

            self.assertEqual(summary["demo_visual_mode"], "pcd")
            self.assertEqual(summary["panel_mode"], PANEL_MODE_SIDE_BY_SIDE)
            self.assertEqual(summary["query_overlay"], "phystwin_rgb_current_points_only")
            self.assertEqual(summary["query_color_mode"], "phystwin_rainbow_identity")
            self.assertEqual(summary["tracking_background_mask_source"], "object_mask|controller_mask")
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 1)
            self.assertEqual(summary["query_count_totals"]["controller"], 1)

    def test_render_does_not_fallback_to_previous_query_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            for seq in (0, 1):
                np.savez(
                    capture_dir / "pcd" / f"{seq:06d}.npz",
                    controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                    controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                    object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                    object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
                )
                np.save(capture_dir / "ffs_depth" / f"{seq:06d}.npy", np.ones((24, 32), dtype=np.float32))
                Image.fromarray(np.full((24, 32, 3), 32 + seq, dtype=np.uint8)).save(
                    capture_dir / "rgb" / f"{seq:06d}.png"
                )
                controller_mask = np.zeros((24, 32), dtype=bool)
                object_mask = np.zeros((24, 32), dtype=bool)
                controller_mask[10:13, 14:18] = True
                object_mask[10:13, 18:21] = True
                np.savez(
                    capture_dir / "masks" / f"{seq:06d}.npz",
                    controller_mask=controller_mask,
                    object_mask=object_mask,
                )
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32]], dtype=np.uint8),
                tracks_yx=np.array([[12.0, 16.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([False], dtype=bool),
                query_is_controller=np.array([True], dtype=bool),
                query_count=np.array([1], dtype=np.int64),
            )
            rows = [
                {
                    "seq": 0,
                    "pcd_path": "pcd/000000.npz",
                    "ffs_depth_path": "ffs_depth/000000.npy",
                    "rgb_path": "rgb/000000.png",
                    "query_trajectory_path": "query_trajectory/000000.npz",
                    "mask_path": "masks/000000.npz",
                },
                {
                    "seq": 1,
                    "pcd_path": "pcd/000001.npz",
                    "ffs_depth_path": "ffs_depth/000001.npy",
                    "rgb_path": "rgb/000001.png",
                    "query_trajectory_path": "query_trajectory/000001.npz",
                    "mask_path": "masks/000001.npz",
                },
            ]
            (capture_dir / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            summary = render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

            self.assertEqual(summary["missing_query_frames"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_trajectory_exact"], 1)
            self.assertEqual(summary["rendered_counts"][1]["query_trajectory_exact"], 0)
            self.assertEqual(summary["rendered_counts"][1]["query_points"], 0)

    def test_tracking_rgb_background_mask_does_not_require_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 16,
                "height": 12,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.empty((0, 3), dtype=np.float32),
                controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((12, 16), dtype=np.float32))
            Image.fromarray(np.full((12, 16, 3), 90, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                tracks_yx=np.array([[6.0, 8.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                query_count=np.array([1], dtype=np.int64),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                tracking_background_mask=TRACKING_BACKGROUND_MASK_RGB,
            )

            self.assertEqual(summary["tracking_background_mask"], TRACKING_BACKGROUND_MASK_RGB)
            self.assertEqual(summary["tracking_background_mask_source"], "full_rgb")
            self.assertEqual(summary["tracking_background_mask_pixel_total"], 0)
            self.assertEqual(summary["rendered_counts"][0]["tracking_background_mask_pixels"], 0)
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 1)

    def test_project_points_applies_world_to_camera_for_table_world_pcd(self) -> None:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        camera_points = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
        world_points = camera_points + c2w[:3, 3]

        uv, valid = _project_points(
            world_points,
            {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            width=32,
            height=24,
            coordinate_frame="table_world_z0",
            camera_to_world_c2w=c2w,
        )

        self.assertTrue(bool(valid[0]))
        np.testing.assert_array_equal(uv[0], np.array([16, 12], dtype=np.int32))

    def test_table_z_filter_overlay_sweep_renders_removed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "rgb").mkdir()
            (capture_dir / "ffs_depth").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "pcd_coordinate_frame": "table_world_z0",
                "table_z_m": 0.0,
                "table_z_above_direction": "negative",
                "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.empty((0, 3), dtype=np.float32),
                controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                object_xyz_m=np.array([[0.0, 0.0, -0.01], [0.01, 0.0, -0.1]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0], [0, 128, 255]], dtype=np.uint8),
                coordinate_frame=np.asarray(["table_world_z0"]),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_table_z_filter_overlay_sweep(
                capture_dir=capture_dir,
                output_dir=capture_dir / "z_overlay",
                fps=30.0,
                thresholds_m=(0.02,),
            )

            self.assertEqual(summary["pcd_coordinate_frame"], "table_world_z0")
            self.assertEqual(summary["thresholds_m"], [0.02])
            threshold_summary = summary["thresholds"][0]
            self.assertTrue(Path(threshold_summary["output"]).is_file())
            self.assertEqual(threshold_summary["removed_total"], 1)
            self.assertEqual(threshold_summary["kept_total"], 1)
            self.assertEqual(threshold_summary["frames"][0]["object_removed_points"], 1)

    def test_pcd_visual_mode_suppresses_query_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32]], dtype=np.uint8),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([False], dtype=bool),
                query_is_controller=np.array([True], dtype=bool),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "query_trajectory_path": "query_trajectory/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                demo_visual_mode="pcd",
            )

            self.assertEqual(summary["demo_visual_mode"], "pcd")
            self.assertEqual(summary["query_overlay"], "none")
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 0)


if __name__ == "__main__":
    unittest.main()
