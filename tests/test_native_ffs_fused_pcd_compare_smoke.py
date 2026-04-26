from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.workflows.native_ffs_fused_pcd_compare import (
    build_native_ffs_fused_pcd_board,
    build_static_native_ffs_fused_pcd_round_specs,
    fuse_native_ffs_depth,
    run_native_ffs_fused_pcd_workflow,
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
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 85, dtype=np.uint8)


class NativeFfsFusedPcdCompareSmokeTest(unittest.TestCase):
    def test_static_round_specs_match_expected_case_pairs_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            specs = build_static_native_ffs_fused_pcd_round_specs(aligned_root=Path(tmp_dir) / "data")

        self.assertEqual([item["round_id"] for item in specs], ["round1", "round2", "round3"])
        self.assertTrue(str(specs[0]["native_case_ref"]).startswith("static/native_30_static_round1"))
        self.assertTrue(str(specs[0]["ffs_case_ref"]).startswith("static/ffs_30_static_round1"))
        self.assertTrue(str(specs[0]["mask_root"]).endswith("masked_pointcloud_compare_round1_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks"))

    def test_fuse_native_ffs_depth_uses_ffs_for_missing_and_below_threshold(self) -> None:
        native = np.array([[0.9, 0.0, 0.5], [0.6, 0.59, np.nan]], dtype=np.float32)
        ffs = np.array([[0.8, 0.7, 0.75], [0.65, 0.72, 0.0]], dtype=np.float32)

        fused, stats = fuse_native_ffs_depth(native, ffs, native_min_m=0.6)

        np.testing.assert_allclose(
            fused,
            np.array([[0.9, 0.7, 0.75], [0.6, 0.72, 0.0]], dtype=np.float32),
        )
        self.assertEqual(stats["native_kept_pixel_count"], 2)
        self.assertEqual(stats["native_missing_pixel_count"], 2)
        self.assertEqual(stats["native_below_threshold_pixel_count"], 2)
        self.assertEqual(stats["ffs_filled_pixel_count"], 3)
        self.assertEqual(stats["unfilled_pixel_count"], 1)

    def test_build_board_returns_3x3_matrix(self) -> None:
        rendered_rows = [[np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(3)] for _ in range(3)]
        variant_rows = [
            {"key": "native", "row_header": "Native depth", "summary_label": "native_depth"},
            {"key": "ffs_original", "row_header": "Original FFS", "summary_label": "original_ffs"},
            {"key": "fused", "row_header": "Fused depth", "summary_label": "native_ffs_fused"},
        ]

        board = build_native_ffs_fused_pcd_board(
            round_label="Round 1",
            frame_idx=0,
            native_min_m=0.6,
            model_config={
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
        self.assertGreater(board.shape[0], 120)
        self.assertGreater(board.shape[1], 180)
        self.assertGreater(int(np.count_nonzero(board)), 0)

    def test_workflow_writes_masked_object_pcd_board_and_summary(self) -> None:
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
            native_depth[1, 1] = 0
            native_depth[2, 2] = 500
            ffs_depth = np.full((8, 10), 0.8, dtype=np.float32)
            for camera_idx in range(3):
                np.save(native_case_dir / "depth" / str(camera_idx) / "0.npy", native_depth)
                np.save(ffs_case_dir / "depth_ffs_float_m" / str(camera_idx) / "0.npy", ffs_depth)

            render_collector = _RenderCollector()
            summary = run_native_ffs_fused_pcd_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                frame_idx=0,
                tile_width=80,
                tile_height=60,
                max_points_per_camera=None,
                mask_erode_pixels=0,
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

            round_summary = summary["rounds"][0]
            self.assertTrue(Path(round_summary["board_path"]).is_file())
            self.assertTrue((tmp_root / "output" / "summary.json").is_file())
            self.assertEqual(round_summary["row_headers"], ["Native depth", "Original FFS", "Fused depth"])
            self.assertTrue(round_summary["render_contract"]["object_masked"])
            self.assertFalse(round_summary["render_contract"]["formal_depth_written"])
            self.assertEqual(round_summary["render_contract"]["display_postprocess"], "none")
            self.assertEqual(len(render_collector.calls), 9)
            for call in render_collector.calls:
                self.assertEqual(int(call["width"]), 80)
                self.assertEqual(int(call["height"]), 60)
                self.assertEqual(np.asarray(call["intrinsic_matrix"]).shape, (3, 3))
                self.assertEqual(np.asarray(call["extrinsic_matrix"]).shape, (4, 4))
            fused_cam0 = round_summary["per_variant_camera"]["fused"][0]
            self.assertEqual(fused_cam0["fusion"]["ffs_filled_pixel_count"], 2)
            self.assertEqual(fused_cam0["fusion"]["native_kept_pixel_count"], 78)
            self.assertGreater(round_summary["fused_point_counts"]["fused"], round_summary["fused_point_counts"]["native"])
            self.assertEqual(float(summary["model_config"]["native_min_m"]), 0.6)


if __name__ == "__main__":
    unittest.main()
