from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments.ffs_confidence_threshold_sweep_pcd_compare import (
    DEFAULT_CONFIDENCE_SWEEP_THRESHOLDS,
    build_confidence_threshold_sweep_pcd_board,
    parse_confidence_thresholds,
    run_ffs_confidence_threshold_sweep_pcd_workflow,
)
from tests.visualization_test_utils import make_rerun_compare_cases, make_sam31_masks


class _FakeConfidenceRunner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.call_count = 0

    def run_pair_with_confidence(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float]:
        self.call_count += 1
        height, width = np.asarray(left_image).shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        depth_ir_left_m = np.full((height, width), 0.75, dtype=np.float32)
        low_confidence_object_pixels = (xx >= 1) & (xx <= 4) & (yy >= 1) & (yy <= 4)
        confidence_maps = {}
        for mode in ("margin", "max_softmax", "entropy", "variance"):
            confidence = np.full((height, width), 0.9, dtype=np.float32)
            confidence[low_confidence_object_pixels] = 0.1
            confidence_maps[f"confidence_{mode}_ir_left"] = confidence
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
            **confidence_maps,
        }


class _RenderCollector:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, points: np.ndarray, colors: np.ndarray, **kwargs) -> np.ndarray:
        self.calls.append(
            {
                "point_count": int(len(np.asarray(points))),
                "width": int(kwargs["width"]),
                "height": int(kwargs["height"]),
                "render_kind": str(kwargs.get("render_kind", "")),
                "metric_name": str(kwargs.get("metric_name", "")),
                "camera_idx": int(kwargs["camera_idx"]),
                "confidence_threshold": float(kwargs.get("confidence_threshold", -1.0)),
            }
        )
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 90, dtype=np.uint8)


class FfsConfidenceThresholdSweepPcdCompareSmokeTest(unittest.TestCase):
    def test_parse_confidence_thresholds_uses_default_percent_sweep(self) -> None:
        self.assertEqual(
            parse_confidence_thresholds(DEFAULT_CONFIDENCE_SWEEP_THRESHOLDS),
            [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
        )
        self.assertEqual(parse_confidence_thresholds("0.01,0.50"), [0.01, 0.50])
        with self.assertRaises(ValueError):
            parse_confidence_thresholds("0.01,1.25")

    def test_build_confidence_threshold_sweep_pcd_board_returns_6x3_matrix(self) -> None:
        rendered_rows = [[np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(3)] for _ in range(6)]
        variant_rows = [
            {"key": f"row{idx}", "row_header": f"Row {idx}", "summary_label": f"row_{idx}"}
            for idx in range(6)
        ]

        board = build_confidence_threshold_sweep_pcd_board(
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
                "mask_erode_pixels": 1,
                "phystwin_like_postprocess_enabled": True,
                "phystwin_radius_m": 0.01,
                "phystwin_nb_points": 40,
            },
            column_headers=["Cam0", "Cam1", "Cam2"],
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )

        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 240)
        self.assertGreater(board.shape[1], 180)
        self.assertGreater(int(np.count_nonzero(board)), 0)

    def test_workflow_writes_one_result_folder_for_threshold_sweep(self) -> None:
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
            fake_runner = _FakeConfidenceRunner()

            summary = run_ffs_confidence_threshold_sweep_pcd_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                ffs_repo="/tmp/fake_ffs_repo",
                model_path="/tmp/fake_model_path.pth",
                frame_idx=0,
                thresholds=[0.01, 0.50],
                tile_width=80,
                tile_height=60,
                max_points_per_camera=None,
                mask_erode_pixels=1,
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
                runner_factory=lambda **kwargs: fake_runner,
                render_frame_fn=render_collector,
            )

            self.assertTrue((tmp_root / "output" / "summary.json").is_file())
            round_summary = summary["rounds"][0]
            self.assertEqual(round_summary["thresholds"], [0.01, 0.50])
            self.assertEqual(len(round_summary["threshold_summaries"]), 2)
            self.assertEqual(round_summary["mask_contract"]["erode_pixels"], 1)
            self.assertTrue(round_summary["postprocess"]["enabled"])
            self.assertEqual(fake_runner.call_count, 3)
            self.assertEqual(len(render_collector.calls), 36)
            self.assertLess(
                round_summary["mask_debug"]["0"]["mask_pixel_count_after_erode"],
                round_summary["mask_debug"]["0"]["mask_pixel_count_before_erode"],
            )

            for threshold_summary in round_summary["threshold_summaries"]:
                board_path = Path(threshold_summary["board_path"])
                self.assertTrue(board_path.is_file())
                self.assertTrue((Path(threshold_summary["output_dir"]) / "summary.json").is_file())
                self.assertTrue(threshold_summary["render_contract"]["object_masked"])
                self.assertEqual(threshold_summary["render_contract"]["mask_erode_pixels"], 1)
                self.assertEqual(threshold_summary["postprocess"]["mode"], "phystwin_like_radius_neighbor_filter")

            high_threshold = round_summary["threshold_summaries"][1]
            self.assertEqual(
                high_threshold["row_headers"],
                [
                    "RealSense native depth",
                    "Fast-FoundationStereo depth\nno confidence filter",
                    "Fast-FoundationStereo depth\nmargin confidence >= 0.50",
                    "Fast-FoundationStereo depth\nmaximum softmax confidence >= 0.50",
                    "Fast-FoundationStereo depth\nentropy confidence >= 0.50",
                    "Fast-FoundationStereo depth\nvariance confidence >= 0.50",
                ],
            )


if __name__ == "__main__":
    unittest.main()
