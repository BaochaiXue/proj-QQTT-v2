from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.workflows.ffs_confidence_pcd_panels import (
    build_confidence_pcd_board,
    build_static_confidence_pcd_round_specs,
    run_ffs_static_confidence_pcd_panels_workflow,
)
from tests.visualization_test_utils import make_rerun_compare_cases, make_sam31_masks


class _FakeConfidenceRunner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)

    def run_pair_with_confidence(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float]:
        height, width = np.asarray(left_image).shape[:2]
        disparity = np.full((height, width), 2.0, dtype=np.float32)
        depth_ir_left_m = np.full((height, width), 0.75, dtype=np.float32)
        margin = np.linspace(0.0, 1.0, num=height * width, dtype=np.float32).reshape(height, width)
        max_softmax = np.flipud(margin).astype(np.float32)
        return {
            "disparity": disparity,
            "depth_ir_left_m": depth_ir_left_m,
            "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
            "baseline_m": float(baseline_m),
            "scale": 1.0,
            "resize_scale_x": 1.0,
            "resize_scale_y": 1.0,
            "valid_iters": 8,
            "max_disp": 192,
            "confidence_margin_ir_left": margin,
            "confidence_max_softmax_ir_left": max_softmax,
        }


class _RenderCollector:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, points: np.ndarray, colors: np.ndarray, **kwargs) -> np.ndarray:
        points_array = np.asarray(points, dtype=np.float32)
        colors_array = np.asarray(colors, dtype=np.uint8)
        self.calls.append(
            {
                "point_count": int(len(points_array)),
                "color_count": int(len(colors_array)),
                "width": int(kwargs["width"]),
                "height": int(kwargs["height"]),
                "render_kind": str(kwargs.get("render_kind", "")),
                "metric_name": None if kwargs.get("metric_name") is None else str(kwargs.get("metric_name")),
                "camera_idx": None if kwargs.get("camera_idx") is None else int(kwargs.get("camera_idx")),
                "intrinsic_matrix": np.asarray(kwargs.get("intrinsic_matrix"), dtype=np.float32).copy(),
                "extrinsic_matrix": np.asarray(kwargs.get("extrinsic_matrix"), dtype=np.float32).copy(),
                "colors": colors_array.copy(),
            }
        )
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 90, dtype=np.uint8)


class FfsConfidencePcdPanelsSmokeTest(unittest.TestCase):
    def test_build_static_confidence_pcd_round_specs_matches_expected_round_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            specs = build_static_confidence_pcd_round_specs(aligned_root=Path(tmp_dir) / "data")

        self.assertEqual([item["round_id"] for item in specs], ["round1", "round2", "round3"])
        self.assertTrue(str(specs[2]["mask_root"]).endswith("masked_pointcloud_compare_round3_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks"))

    def test_build_confidence_pcd_board_returns_nonempty_matrix_board(self) -> None:
        rgb_images = [np.full((40, 60, 3), 40, dtype=np.uint8) for _ in range(3)]
        pcd_images = [np.full((40, 60, 3), 80, dtype=np.uint8) for _ in range(3)]
        confidence_pcd_images = [np.full((40, 60, 3), 120, dtype=np.uint8) for _ in range(3)]

        board = build_confidence_pcd_board(
            round_label="Round 1",
            frame_idx=0,
            metric_name="margin",
            model_config={"scale": 1.0, "valid_iters": 8, "max_disp": 192},
            column_headers=["Cam0", "Cam1", "Cam2"],
            rgb_images=rgb_images,
            pcd_images=pcd_images,
            confidence_pcd_images=confidence_pcd_images,
        )

        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 120)
        self.assertGreater(board.shape[1], 180)
        self.assertGreater(int(np.count_nonzero(board)), 0)

    def test_workflow_writes_metric_boards_and_uses_fused_masked_ffs_pcd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            static_root = aligned_root / "static"
            static_root.mkdir(parents=True, exist_ok=True)
            _, ffs_case_dir = make_rerun_compare_cases(
                static_root,
                native_case_name="unused_native_case",
                ffs_case_name="ffs_case_round1",
                frame_num=1,
            )
            mask_root = make_sam31_masks(ffs_case_dir, prompt_labels_by_object={1: "stuffed animal"})
            output_root = tmp_root / "ffs_confidence_pcd_output"
            render_collector = _RenderCollector()

            summary = run_ffs_static_confidence_pcd_panels_workflow(
                aligned_root=aligned_root,
                output_root=output_root,
                ffs_repo="/tmp/fake_ffs_repo",
                model_path="/tmp/fake_model_path.pth",
                metrics="both",
                round_specs=[
                    {
                        "round_id": "round1",
                        "round_label": "Round 1",
                        "case_ref": "static/ffs_case_round1",
                        "mask_root": mask_root,
                    }
                ],
                runner_factory=lambda **kwargs: _FakeConfidenceRunner(**kwargs),
                render_frame_fn=render_collector,
            )

            round_dir = output_root / "round1"
            self.assertTrue((round_dir / "margin_board.png").is_file())
            self.assertTrue((round_dir / "max_softmax_board.png").is_file())
            self.assertTrue((round_dir / "summary.json").is_file())
            self.assertTrue((output_root / "summary.json").is_file())
            self.assertEqual(summary["metrics"], ["margin", "max_softmax"])
            self.assertEqual(len(summary["rounds"]), 1)
            round_summary = summary["rounds"][0]
            self.assertEqual(round_summary["row_headers"], ["RGB", "Masked FFS PCD", "Confidence PCD"])
            self.assertEqual(round_summary["pcd_render_contract"]["row2_source"], "fused_masked_ffs")
            self.assertEqual(round_summary["pcd_render_contract"]["row3_source"], "fused_masked_ffs_confidence")
            self.assertEqual(len(round_summary["column_views"]), 3)
            self.assertGreater(int(round_summary["fused_masked_point_count"]), 0)
            self.assertGreater(int(round_summary["cropped_fused_masked_point_count"]), 0)
            self.assertEqual(sorted(round_summary["board_paths"].keys()), ["margin", "max_softmax"])
            self.assertEqual(len(render_collector.calls), 9)
            rgb_calls = [call for call in render_collector.calls if call["render_kind"] == "rgb_pcd"]
            confidence_calls = [call for call in render_collector.calls if call["render_kind"] == "confidence_pcd"]
            self.assertEqual(len(rgb_calls), 3)
            self.assertEqual(len(confidence_calls), 6)
            for call in render_collector.calls:
                self.assertGreater(int(call["point_count"]), 0)
                self.assertEqual(int(call["point_count"]), int(call["color_count"]))
                self.assertEqual(np.asarray(call["intrinsic_matrix"]).shape, (3, 3))
                self.assertEqual(np.asarray(call["extrinsic_matrix"]).shape, (4, 4))
            rgb_calls_by_camera = {int(call["camera_idx"]): call for call in rgb_calls}
            for call in confidence_calls:
                camera_idx = int(call["camera_idx"])
                self.assertIn(camera_idx, rgb_calls_by_camera)
                self.assertEqual(int(call["point_count"]), int(rgb_calls_by_camera[camera_idx]["point_count"]))
                self.assertFalse(np.array_equal(np.asarray(call["colors"]), np.asarray(rgb_calls_by_camera[camera_idx]["colors"])))
                self.assertIn(str(call["metric_name"]), ("margin", "max_softmax"))


if __name__ == "__main__":
    unittest.main()
