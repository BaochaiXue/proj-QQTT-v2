from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments.ffs_mask_erode_sweep_pcd_compare import (
    DEFAULT_MASK_ERODE_PIXELS,
    DEFAULT_ROW_LABEL_WIDTH,
    build_ffs_mask_erode_sweep_pcd_board,
    parse_mask_erode_pixels,
    run_ffs_mask_erode_sweep_pcd_workflow,
)
from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


class _RenderCollector:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, points: np.ndarray, colors: np.ndarray, **kwargs) -> np.ndarray:
        self.calls.append(
            {
                "point_count": int(len(np.asarray(points))),
                "color_count": int(len(np.asarray(colors))),
                "width": int(kwargs["width"]),
                "height": int(kwargs["height"]),
                "render_kind": str(kwargs.get("render_kind", "")),
                "metric_name": str(kwargs.get("metric_name", "")),
                "camera_idx": int(kwargs["camera_idx"]),
                "intrinsic_matrix": np.asarray(kwargs.get("intrinsic_matrix"), dtype=np.float32).copy(),
                "extrinsic_matrix": np.asarray(kwargs.get("extrinsic_matrix"), dtype=np.float32).copy(),
            }
        )
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 82, dtype=np.uint8)


class FfsMaskErodeSweepPcdCompareSmokeTest(unittest.TestCase):
    def test_parse_mask_erode_pixels_uses_default_1_to_8_sweep(self) -> None:
        self.assertEqual(parse_mask_erode_pixels(DEFAULT_MASK_ERODE_PIXELS), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(parse_mask_erode_pixels("1,8"), [1, 8])
        with self.assertRaises(ValueError):
            parse_mask_erode_pixels("0,1")
        with self.assertRaises(ValueError):
            parse_mask_erode_pixels("1,1")

    def test_build_board_returns_10x3_matrix_for_default_rows(self) -> None:
        rendered_rows = [[np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(3)] for _ in range(10)]
        variant_rows = [
            {
                "key": "native",
                "row_header": "RealSense native depth\nobject mask unchanged",
                "summary_label": "native_depth_mask_0px",
            },
            {
                "key": "ffs_original",
                "row_header": "Fast-FoundationStereo depth\nobject mask unchanged",
                "summary_label": "ffs_original_mask_0px",
            },
            *[
                {
                    "key": f"ffs_erode_{idx}px",
                    "row_header": f"Fast-FoundationStereo depth\nobject mask eroded inward {idx}px",
                    "summary_label": f"ffs_mask_erode_{idx}px",
                }
                for idx in range(1, 9)
            ],
        ]

        board = build_ffs_mask_erode_sweep_pcd_board(
            round_label="Round 1",
            frame_idx=0,
            model_config={
                "depth_min_m": 0.2,
                "depth_max_m": 1.5,
                "object_mask_enabled": True,
                "mask_erode_pixels": [1, 2, 3, 4, 5, 6, 7, 8],
                "phystwin_like_postprocess_enabled": True,
                "phystwin_radius_m": 0.01,
                "phystwin_nb_points": 40,
                "use_float_ffs_depth_when_available": True,
                "row_label_width": DEFAULT_ROW_LABEL_WIDTH,
            },
            column_headers=["Cam0", "Cam1", "Cam2"],
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )

        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 400)
        self.assertGreaterEqual(board.shape[1], DEFAULT_ROW_LABEL_WIDTH + 3 * 60)
        self.assertGreater(int(np.count_nonzero(board)), 0)

    def test_workflow_writes_one_result_folder_for_erode_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            static_root = aligned_root / "static"
            native_case_dir = static_root / "native_case_round1"
            ffs_case_dir = static_root / "ffs_case_round1"
            make_visualization_case(native_case_dir, frame_num=1)
            make_visualization_case(
                ffs_case_dir,
                include_depth_ffs_float_m=True,
                frame_num=1,
                depth_backend_used="ffs",
                depth_source_for_depth_dir="ffs",
            )
            mask_root = make_sam31_masks(ffs_case_dir, prompt_labels_by_object={1: "stuffed animal"})
            native_depth = np.full((8, 10), 900, dtype=np.uint16)
            ffs_depth = np.full((8, 10), 0.8, dtype=np.float32)
            for camera_idx in range(3):
                np.save(native_case_dir / "depth" / str(camera_idx) / "0.npy", native_depth)
                np.save(ffs_case_dir / "depth_ffs_float_m" / str(camera_idx) / "0.npy", ffs_depth)

            render_collector = _RenderCollector()
            summary = run_ffs_mask_erode_sweep_pcd_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                frame_idx=0,
                erode_pixels=[1, 2],
                tile_width=80,
                tile_height=60,
                row_label_width=240,
                max_points_per_camera=None,
                phystwin_like_postprocess=False,
                round_specs=[
                    {
                        "round_id": "round1",
                        "round_label": "Round 1",
                        "native_case_ref": "static/native_case_round1",
                        "ffs_case_ref": "static/ffs_case_round1",
                        "mask_root": mask_root,
                    }
                ],
                render_frame_fn=render_collector,
            )

            self.assertTrue((tmp_root / "output" / "summary.json").is_file())
            round_summary = summary["rounds"][0]
            self.assertTrue(Path(round_summary["board_path"]).is_file())
            self.assertEqual(round_summary["erode_pixels"], [1, 2])
            self.assertEqual(
                round_summary["row_headers"],
                [
                    "RealSense native depth\nobject mask unchanged",
                    "Fast-FoundationStereo depth\nobject mask unchanged",
                    "Fast-FoundationStereo depth\nobject mask eroded inward 1px",
                    "Fast-FoundationStereo depth\nobject mask eroded inward 2px",
                ],
            )
            self.assertEqual(len(render_collector.calls), 12)
            self.assertEqual(round_summary["render_contract"]["rows"], "native_depth_ffs_original_ffs_mask_erode_sweep")
            self.assertEqual(round_summary["render_contract"]["display_postprocess"], "none")
            self.assertEqual(round_summary["model_config"]["row_label_width"], 240)
            self.assertLess(
                round_summary["mask_debug"]["0"]["eroded"]["1"]["mask_pixel_count_after_erode"],
                round_summary["mask_debug"]["0"]["raw"]["mask_pixel_count"],
            )
            self.assertIn("ffs_erode_1px", round_summary["fused_point_counts"])
            self.assertIn("ffs_erode_2px", round_summary["fused_point_counts"])
            for call in render_collector.calls:
                self.assertEqual(int(call["width"]), 80)
                self.assertEqual(int(call["height"]), 60)
                self.assertEqual(np.asarray(call["intrinsic_matrix"]).shape, (3, 3))
                self.assertEqual(np.asarray(call["extrinsic_matrix"]).shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
