from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.depth_backends.fast_foundation_stereo import compute_confidence_proxies_from_logits
from data_process.depth_backends.geometry import align_ir_scalar_to_color
from data_process.visualization.workflows.ffs_confidence_panels import (
    build_confidence_board,
    build_static_confidence_round_specs,
    run_ffs_static_confidence_panels_workflow,
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


class FfsConfidencePanelsSmokeTest(unittest.TestCase):
    def test_compute_confidence_proxies_from_logits_returns_margin_and_max_softmax(self) -> None:
        logits = np.array(
            [
                [
                    [[3.0, 1.0], [0.0, 0.0]],
                    [[1.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [5.0, -1.0]],
                ]
            ],
            dtype=np.float32,
        )
        confidence = compute_confidence_proxies_from_logits(logits)

        self.assertEqual(set(confidence.keys()), {"margin", "max_softmax"})
        self.assertEqual(confidence["margin"].shape, (1, 2, 2))
        self.assertEqual(confidence["max_softmax"].shape, (1, 2, 2))
        self.assertTrue(np.all(confidence["margin"] >= 0.0))
        self.assertTrue(np.all(confidence["margin"] <= 1.0))
        self.assertTrue(np.all(confidence["max_softmax"] >= 0.0))
        self.assertTrue(np.all(confidence["max_softmax"] <= 1.0))
        self.assertGreater(float(confidence["margin"][0, 0, 0]), 0.0)
        self.assertGreaterEqual(
            float(confidence["max_softmax"][0, 0, 0]),
            float(confidence["margin"][0, 0, 0]),
        )

    def test_align_ir_scalar_to_color_uses_nearest_depth_winner(self) -> None:
        depth_ir = np.array([[1.0, 0.5]], dtype=np.float32)
        scalar_ir = np.array([[0.2, 0.9]], dtype=np.float32)
        K_ir_left = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T_ir_left_to_color = np.eye(4, dtype=np.float32)
        K_color = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        scalar_color = align_ir_scalar_to_color(
            depth_ir,
            scalar_ir,
            K_ir_left,
            T_ir_left_to_color,
            K_color,
            output_shape=(1, 1),
            invalid_value=0.0,
        )

        self.assertEqual(scalar_color.shape, (1, 1))
        self.assertAlmostEqual(float(scalar_color[0, 0]), 0.9, places=6)

    def test_build_static_confidence_round_specs_matches_expected_round_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            specs = build_static_confidence_round_specs(aligned_root=Path(tmp_dir) / "data")

        self.assertEqual([item["round_id"] for item in specs], ["round1", "round2", "round3"])
        self.assertTrue(str(specs[0]["mask_root"]).endswith("masked_pointcloud_compare_round1_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks"))

    def test_build_confidence_board_returns_matrix_board_with_legends(self) -> None:
        rgb_images = [np.full((40, 60, 3), 50, dtype=np.uint8) for _ in range(3)]
        depth_images = [np.full((40, 60, 3), 0, dtype=np.uint8) for _ in range(3)]
        confidence_images = [np.full((40, 60, 3), 0, dtype=np.uint8) for _ in range(3)]

        board = build_confidence_board(
            round_label="Round 1",
            frame_idx=0,
            metric_name="margin",
            model_config={"scale": 1.0, "valid_iters": 8, "max_disp": 192, "depth_min_m": 0.0, "depth_max_m": 1.5},
            column_headers=["Cam0", "Cam1", "Cam2"],
            rgb_images=rgb_images,
            depth_images=depth_images,
            confidence_images=confidence_images,
        )

        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 120)
        self.assertGreater(board.shape[1], 180)
        self.assertGreater(int(np.count_nonzero(board)), 0)

    def test_workflow_writes_margin_and_max_softmax_boards_for_one_round(self) -> None:
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
            output_root = tmp_root / "ffs_confidence_output"

            summary = run_ffs_static_confidence_panels_workflow(
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
            )

            round_dir = output_root / "round1"
            self.assertTrue((round_dir / "margin_board.png").is_file())
            self.assertTrue((round_dir / "max_softmax_board.png").is_file())
            self.assertTrue((round_dir / "summary.json").is_file())
            self.assertTrue((output_root / "summary.json").is_file())
            self.assertEqual(summary["metrics"], ["margin", "max_softmax"])
            self.assertEqual(len(summary["rounds"]), 1)
            self.assertEqual(summary["rounds"][0]["round_id"], "round1")
            self.assertEqual(sorted(summary["rounds"][0]["board_paths"].keys()), ["margin", "max_softmax"])
            self.assertEqual(len(summary["rounds"][0]["per_camera"]), 3)


if __name__ == "__main__":
    unittest.main()
