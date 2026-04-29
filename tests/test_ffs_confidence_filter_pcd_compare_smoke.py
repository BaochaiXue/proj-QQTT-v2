from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
    build_confidence_filter_pcd_board,
    build_static_confidence_filter_round_specs,
    run_ffs_confidence_filter_pcd_compare_workflow,
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
        yy, xx = np.indices((height, width), dtype=np.float32)
        depth_ir_left_m = np.full((height, width), 0.75, dtype=np.float32)
        confidence_left_high = np.full((height, width), 0.9, dtype=np.float32)
        confidence_right_high = np.full((height, width), 0.9, dtype=np.float32)
        confidence_top_high = np.full((height, width), 0.9, dtype=np.float32)
        confidence_checker = np.full((height, width), 0.9, dtype=np.float32)
        low_confidence_object_pixels = ((xx >= 1) & (xx <= 2) & (yy >= 1) & (yy <= 2))
        confidence_left_high[low_confidence_object_pixels] = 0.1
        confidence_right_high[low_confidence_object_pixels] = 0.1
        confidence_top_high[low_confidence_object_pixels] = 0.1
        confidence_checker[low_confidence_object_pixels] = 0.1
        return {
            "disparity": np.full((height, width), 2.0, dtype=np.float32),
            "depth_ir_left_m": depth_ir_left_m,
            "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
            "baseline_m": float(baseline_m),
            "scale": 1.0,
            "resize_scale_x": 1.0,
            "resize_scale_y": 1.0,
            "valid_iters": 4,
            "max_disp": 192,
            "confidence_margin_ir_left": confidence_left_high,
            "confidence_max_softmax_ir_left": confidence_right_high,
            "confidence_entropy_ir_left": confidence_top_high,
            "confidence_variance_ir_left": confidence_checker,
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
                "metric_name": str(kwargs.get("metric_name", "")),
                "camera_idx": int(kwargs["camera_idx"]),
                "intrinsic_matrix": np.asarray(kwargs.get("intrinsic_matrix"), dtype=np.float32).copy(),
                "extrinsic_matrix": np.asarray(kwargs.get("extrinsic_matrix"), dtype=np.float32).copy(),
            }
        )
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 80, dtype=np.uint8)


class FfsConfidenceFilterPcdCompareSmokeTest(unittest.TestCase):
    def test_static_round_specs_match_expected_case_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            specs = build_static_confidence_filter_round_specs(aligned_root=Path(tmp_dir) / "data")

        self.assertEqual([item["round_id"] for item in specs], ["round1", "round2", "round3"])
        self.assertTrue(str(specs[0]["native_case_ref"]).startswith("static/native_30_static_round1"))
        self.assertTrue(str(specs[0]["ffs_case_ref"]).startswith("static/ffs_30_static_round1"))
        self.assertTrue(str(specs[0]["mask_root"]).endswith("masked_pointcloud_compare_round1_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks"))

    def test_build_confidence_filter_pcd_board_returns_6x3_matrix(self) -> None:
        rendered_rows = [[np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(3)] for _ in range(6)]
        variant_rows = [
            {"key": f"row{idx}", "row_header": f"Row {idx}", "summary_label": f"row_{idx}"}
            for idx in range(6)
        ]

        board = build_confidence_filter_pcd_board(
            round_label="Round 1",
            frame_idx=0,
            confidence_threshold=0.10,
            model_config={
                "scale": 1.0,
                "valid_iters": 4,
                "max_disp": 192,
                "depth_min_m": 0.2,
                "depth_max_m": 1.5,
                "object_mask_enabled": True,
            },
            column_headers=["Cam0", "Cam1", "Cam2"],
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )

        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 240)
        self.assertGreater(board.shape[1], 180)
        self.assertGreater(int(np.count_nonzero(board)), 0)

    def test_workflow_writes_round_board_and_filters_all_four_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            static_root = aligned_root / "static"
            static_root.mkdir(parents=True, exist_ok=True)
            _, ffs_case_dir = make_rerun_compare_cases(
                static_root,
                native_case_name="native_case_round1",
                ffs_case_name="ffs_case_round1",
                frame_num=1,
            )
            mask_root = make_sam31_masks(ffs_case_dir, prompt_labels_by_object={1: "stuffed animal"})
            output_root = tmp_root / "output"
            render_collector = _RenderCollector()

            summary = run_ffs_confidence_filter_pcd_compare_workflow(
                aligned_root=aligned_root,
                output_root=output_root,
                ffs_repo="/tmp/fake_ffs_repo",
                model_path="/tmp/fake_model_path.pth",
                frame_idx=0,
                confidence_threshold=0.50,
                tile_width=80,
                tile_height=60,
                max_points_per_camera=None,
                round_specs=[
                    {
                        "round_id": "round1",
                        "round_label": "Round 1",
                        "native_case_ref": "static/native_case_round1",
                        "ffs_case_ref": "static/ffs_case_round1",
                        "mask_root": mask_root,
                    }
                ],
                runner_factory=lambda **kwargs: _FakeConfidenceRunner(**kwargs),
                render_frame_fn=render_collector,
            )

            round_summary = summary["rounds"][0]
            board_path = Path(round_summary["board_path"])
            self.assertTrue(board_path.is_file())
            self.assertTrue((output_root / "summary.json").is_file())
            self.assertTrue((output_root / "round1" / "summary.json").is_file())
            self.assertEqual(
                round_summary["row_headers"],
                [
                    "RealSense native depth",
                    "Fast-FoundationStereo depth\nno confidence filter",
                    "Fast-FoundationStereo depth\nmargin confidence >= 0.50",
                    "Fast-FoundationStereo depth\nmaximum softmax confidence >= 0.50",
                    "Fast-FoundationStereo depth\nentropy confidence >= 0.50",
                    "Fast-FoundationStereo depth\nvariance confidence >= 0.50",
                ],
            )
            self.assertEqual(round_summary["confidence_modes"], ["margin", "max_softmax", "entropy", "variance"])
            self.assertFalse(round_summary["render_contract"]["formal_depth_written"])
            self.assertTrue(round_summary["render_contract"]["object_masked"])
            self.assertFalse(round_summary["postprocess"]["enabled"])
            self.assertGreater(round_summary["mask_debug"]["0"]["mask_pixel_count"], 0)
            self.assertEqual(len(render_collector.calls), 18)
            for call in render_collector.calls:
                self.assertEqual(int(call["width"]), 80)
                self.assertEqual(int(call["height"]), 60)
                self.assertEqual(np.asarray(call["intrinsic_matrix"]).shape, (3, 3))
                self.assertEqual(np.asarray(call["extrinsic_matrix"]).shape, (4, 4))
            raw_count = int(round_summary["fused_point_counts"]["ffs_original"])
            self.assertGreater(raw_count, 0)
            for variant_key in ("ffs_margin", "ffs_max_softmax", "ffs_entropy", "ffs_variance"):
                self.assertIn(variant_key, round_summary["fused_point_counts"])
                self.assertLess(int(round_summary["fused_point_counts"][variant_key]), raw_count)
                for camera_stats in round_summary["per_variant_camera"][variant_key]:
                    self.assertIn("filter_stats", camera_stats)
                    self.assertEqual(float(camera_stats["filter_stats"]["confidence_threshold"]), 0.5)

    def test_workflow_records_phystwin_like_postprocess_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            static_root = aligned_root / "static"
            static_root.mkdir(parents=True, exist_ok=True)
            _, ffs_case_dir = make_rerun_compare_cases(
                static_root,
                native_case_name="native_case_round1",
                ffs_case_name="ffs_case_round1",
                frame_num=1,
            )
            mask_root = make_sam31_masks(ffs_case_dir, prompt_labels_by_object={1: "stuffed animal"})
            render_collector = _RenderCollector()

            summary = run_ffs_confidence_filter_pcd_compare_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                ffs_repo="/tmp/fake_ffs_repo",
                model_path="/tmp/fake_model_path.pth",
                frame_idx=0,
                confidence_threshold=0.50,
                tile_width=80,
                tile_height=60,
                max_points_per_camera=None,
                phystwin_like_postprocess=True,
                phystwin_radius_m=10.0,
                phystwin_nb_points=1,
                round_specs=[
                    {
                        "round_id": "round1",
                        "round_label": "Round 1",
                        "native_case_ref": "static/native_case_round1",
                        "ffs_case_ref": "static/ffs_case_round1",
                        "mask_root": mask_root,
                    }
                ],
                runner_factory=lambda **kwargs: _FakeConfidenceRunner(**kwargs),
                render_frame_fn=render_collector,
            )

            round_summary = summary["rounds"][0]
            self.assertTrue(round_summary["postprocess"]["enabled"])
            self.assertEqual(round_summary["render_contract"]["display_postprocess"], "phystwin_like_radius_neighbor_filter")
            self.assertEqual(len(render_collector.calls), 18)
            for variant_key, stats in round_summary["postprocess_stats_by_variant"].items():
                self.assertIn(variant_key, round_summary["fused_point_counts_before_postprocess"])
                self.assertGreaterEqual(int(stats["input_point_count"]), int(stats["inlier_point_count"]))
                self.assertEqual(int(stats["inlier_point_count"]), int(round_summary["fused_point_counts"][variant_key]))
                self.assertEqual(float(stats["radius_m"]), 10.0)
                self.assertEqual(int(stats["nb_points"]), 1)


if __name__ == "__main__":
    unittest.main()
