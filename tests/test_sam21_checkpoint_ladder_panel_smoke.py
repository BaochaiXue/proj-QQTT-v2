from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments import sam21_checkpoint_ladder_panel as ladder
from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


class Sam21CheckpointLadderPanelSmokeTest(unittest.TestCase):
    def test_bbox_xyxy_from_mask_uses_tight_frame0_union(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 3:7] = True
        self.assertEqual(ladder.bbox_xyxy_from_mask(mask), [3.0, 2.0, 6.0, 4.0])
        self.assertEqual(ladder.bbox_xyxy_from_mask(mask, padding_px=2), [1.0, 0.0, 8.0, 6.0])
        with self.assertRaises(ValueError):
            ladder.bbox_xyxy_from_mask(np.zeros((4, 4), dtype=bool))

    def test_sam2_mask_writer_schema_loads_as_union_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=2)
            mask0 = np.zeros((8, 10), dtype=bool)
            mask1 = np.zeros((8, 10), dtype=bool)
            mask0[1:4, 2:5] = True
            mask1[2:5, 3:6] = True
            mask_root = root / "sam21_masks"
            summary = ladder.write_single_object_masks(
                mask_root=mask_root,
                camera_idx=0,
                object_label="object",
                masks_by_frame_token={"0": mask0, "1": mask1},
                overwrite=False,
            )

            self.assertEqual(summary["saved_frame_count"], 2)
            loaded = ladder.load_union_mask(
                mask_root=mask_root,
                case_dir=case_dir,
                camera_idx=0,
                frame_token="1",
                text_prompt="object",
            )
            self.assertEqual(int(loaded.sum()), int(mask1.sum()))
            self.assertTrue(np.array_equal(loaded, mask1))

    def test_matched_mask_labels_handles_multi_prompt_rope_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=1)
            mask_root = make_sam31_masks(
                case_dir,
                prompt_labels_by_object={
                    0: "white thick twisted rope on top of the blue box",
                },
                camera_ids=[2],
                frame_tokens=["0"],
            )
            labels = ladder.matched_mask_labels(
                mask_root=mask_root,
                camera_idx=2,
                text_prompt="white twisted rope on the blue box, white thick twisted rope on top of the blue box",
            )
            self.assertEqual(labels, ["white thick twisted rope on top of the blue box"])

    def test_timing_aggregate_uses_inference_ms_without_warmup(self) -> None:
        records = [
            {"case_key": "a", "camera_idx": 0, "checkpoint_key": "tiny", "inference_ms_per_frame": 10.0},
            {"case_key": "a", "camera_idx": 1, "checkpoint_key": "tiny", "inference_ms_per_frame": 20.0},
            {"case_key": "a", "camera_idx": 0, "checkpoint_key": "large", "inference_ms_per_frame": 50.0},
        ]
        aggregate = ladder.aggregate_timing_records(records)
        self.assertEqual(aggregate["record_count"], 3)
        self.assertAlmostEqual(
            aggregate["by_checkpoint"]["tiny"]["mean_inference_ms_per_frame"],
            15.0,
        )
        self.assertAlmostEqual(aggregate["by_checkpoint"]["tiny"]["mean_fps"], 1000.0 / 15.0)

    def test_stable_report_describes_no_output_timing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.md"
            ladder.write_benchmark_report(
                markdown_path=report_path,
                benchmark_payload={
                    "sam21_timing_protocol": "stable_throughput",
                    "stable_warmup_runs": 5,
                    "stable_speed_uses_cudagraph_step_marker": True,
                    "timing_aggregate": {
                        "by_checkpoint": {
                            "tiny": {
                                "mean_inference_ms_per_frame": 8.0,
                                "mean_fps": 125.0,
                                "sample_count": 18,
                            }
                        }
                    },
                    "stable_checkpoint_summaries": [
                        {
                            "checkpoint_key": "tiny",
                            "aggregate_ms_per_frame": 8.0,
                            "aggregate_fps": 125.0,
                            "speed_phase_wall_fps_including_state_setup": 80.0,
                            "total_timed_frames": 540,
                        }
                    ],
                    "gif_summaries": [],
                    "environment": {},
                },
                quality_payload={"cases": {}},
            )
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("no-output propagation timing with per-step cudagraph markers", text)
            self.assertIn("separate mask collection are excluded", text)
            self.assertIn("| tiny | 8.00 | 125.00 | 80.00 | 540 |", text)

    def test_compose_3x5_panel_shape(self) -> None:
        rows = [
            [np.full((12, 16, 3), 30 + row * 20 + col, dtype=np.uint8) for col in range(5)]
            for row in range(3)
        ]
        board = ladder.compose_3x5_panel(
            title_lines=["title", "subtitle"],
            row_headers=["cam0", "cam1", "cam2"],
            column_headers=["sam31", "large", "base+", "small", "tiny"],
            image_rows=rows,
            row_label_width=20,
        )
        self.assertEqual(board.shape, (84 + 38 + 3 * 12, 20 + 5 * 16, 3))

    def test_pinhole_renderer_uses_original_camera_z_buffer(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        colors = np.asarray(
            [
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0],
            ],
            dtype=np.uint8,
        )
        K = np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        image = ladder.render_pinhole_point_cloud(
            points,
            colors,
            intrinsic_matrix=K,
            extrinsic_matrix=np.eye(4, dtype=np.float32),
            width=12,
            height=12,
            point_radius_px=0,
        )
        self.assertTrue(np.array_equal(image[5, 5], np.asarray([0, 255, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(image[5, 7], np.asarray([255, 0, 0], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
