from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data_process.visualization.triplet_video_compare import run_triplet_video_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class TripletVideoCompareWorkflowSmokeTest(unittest.TestCase):
    def test_two_case_workflow_writes_three_videos_and_on_the_fly_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=2)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=2)

            output_dir = tmp_root / "triplet_video_output"
            summary = run_triplet_video_compare_workflow(
                aligned_root=aligned_root,
                realsense_case="native_case",
                ffs_case="ffs_case",
                output_dir=output_dir,
                render_frame_fn=lambda points, colors, **kwargs: np.full((72, 96, 3), 90, dtype=np.uint8),
            )

            self.assertTrue((output_dir / "native_open3d.mp4").is_file())
            self.assertTrue((output_dir / "ffs_raw_open3d.mp4").is_file())
            self.assertTrue((output_dir / "ffs_postprocess_open3d.mp4").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())

            on_disk_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["same_case_mode"])
            self.assertEqual(on_disk_summary["frame_count"], 2)
            self.assertEqual(on_disk_summary["render_contract"]["render_mode"], "color_by_rgb")
            self.assertEqual(on_disk_summary["render_contract"]["image_flip"], "vertical")
            self.assertEqual(on_disk_summary["variants"]["native"]["frame_count"], 2)
            self.assertEqual(on_disk_summary["variants"]["ffs_raw"]["frame_count"], 2)
            self.assertEqual(on_disk_summary["variants"]["ffs_postprocess"]["frame_count"], 2)
            self.assertEqual(on_disk_summary["variants"]["ffs_postprocess"]["ffs_native_like_postprocess_origin"], "on_the_fly")

    def test_same_case_workflow_prefers_aligned_postprocess_auxiliary_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "both_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_native_like_postprocess=True,
                include_depth_ffs_native_like_postprocess_float_m=True,
                frame_num=2,
            )

            output_dir = tmp_root / "triplet_video_output"
            summary = run_triplet_video_compare_workflow(
                aligned_root=aligned_root,
                case_name="both_case",
                output_dir=output_dir,
                render_frame_fn=lambda points, colors, **kwargs: np.full((72, 96, 3), 120, dtype=np.uint8),
            )

            self.assertTrue(summary["same_case_mode"])
            self.assertEqual(summary["variants"]["ffs_postprocess"]["ffs_native_like_postprocess_origin"], "aligned_auxiliary")
            self.assertEqual(
                summary["variants"]["ffs_postprocess"]["depth_dirs_used"],
                ["depth_ffs_native_like_postprocess_float_m"],
            )

    def test_two_case_workflow_prefers_archived_raw_depth_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=2)
            make_visualization_case(
                ffs_case,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_original=True,
                include_depth_ffs_float_m_original=True,
                frame_num=2,
                depth_backend_used="both",
                depth_source_for_depth_dir="realsense",
            )

            output_dir = tmp_root / "triplet_video_output"
            summary = run_triplet_video_compare_workflow(
                aligned_root=aligned_root,
                realsense_case="native_case",
                ffs_case="ffs_case",
                output_dir=output_dir,
                render_frame_fn=lambda points, colors, **kwargs: np.full((72, 96, 3), 90, dtype=np.uint8),
            )

            self.assertEqual(summary["variants"]["ffs_raw"]["depth_dirs_used"], ["depth_ffs_float_m_original"])

    def test_workflow_uses_rgb_colors_and_vertical_flip(self) -> None:
        seen_colorful = {"value": False}

        def _fake_renderer(points, colors, **kwargs):
            channel_equal = bool(np.all(colors[:, 0] == colors[:, 1]) and np.all(colors[:, 1] == colors[:, 2]))
            seen_colorful["value"] = not channel_equal
            image = np.zeros((80, 120, 3), dtype=np.uint8)
            image[:40, :, :] = np.array([0, 255, 0], dtype=np.uint8)
            image[40:, :, :] = np.array([255, 0, 0], dtype=np.uint8)
            return image

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "triplet_video_output"
            summary = run_triplet_video_compare_workflow(
                aligned_root=aligned_root,
                realsense_case="native_case",
                ffs_case="ffs_case",
                output_dir=output_dir,
                render_frame_fn=_fake_renderer,
            )

            self.assertTrue(seen_colorful["value"])
            native_frame = cv2.imread(str(output_dir / "native_frames" / "000000.png"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(native_frame)
            self.assertTrue(np.array_equal(native_frame[10, 10], np.array([255, 0, 0], dtype=np.uint8)))
            self.assertTrue(np.array_equal(native_frame[70, 10], np.array([0, 255, 0], dtype=np.uint8)))
            self.assertEqual(summary["render_contract"]["render_mode"], "color_by_rgb")
            self.assertEqual(summary["render_contract"]["image_flip"], "vertical")


if __name__ == "__main__":
    unittest.main()
