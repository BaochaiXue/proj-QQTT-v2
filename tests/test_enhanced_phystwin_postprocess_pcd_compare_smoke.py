from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments.enhanced_phystwin_postprocess_pcd_compare import (
    build_enhanced_phystwin_postprocess_pcd_board,
    run_enhanced_phystwin_postprocess_pcd_workflow,
)
from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
    _apply_enhanced_phystwin_like_postprocess,
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
            }
        )
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 104, dtype=np.uint8)


class EnhancedPhystwinPostprocessPcdCompareSmokeTest(unittest.TestCase):
    def test_enhanced_postprocess_removes_disconnected_component(self) -> None:
        main = np.asarray(
            [[x, y, z] for x in (0.0, 0.004, 0.008) for y in (0.0, 0.004, 0.008) for z in (0.0, 0.004, 0.008)],
            dtype=np.float32,
        )
        floating = np.asarray([[0.12, 0.0, 0.0], [0.124, 0.0, 0.0], [0.128, 0.0, 0.0], [0.132, 0.0, 0.0]], dtype=np.float32)
        points = np.concatenate([main, floating], axis=0)
        colors = np.full((len(points), 3), 255, dtype=np.uint8)

        filtered_points, filtered_colors, stats = _apply_enhanced_phystwin_like_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=0.02,
            nb_points=2,
            component_voxel_size_m=0.01,
            keep_near_main_gap_m=0.0,
        )

        self.assertEqual(len(filtered_points), len(main))
        self.assertEqual(len(filtered_colors), len(main))
        self.assertEqual(stats["component_count"], 2)
        self.assertEqual(stats["removed_component_count"], 1)
        self.assertEqual(stats["removed_point_count"], len(floating))
        self.assertGreater(stats["removed_components"][0]["bbox_gap_to_main_m"], 0.05)

    def test_build_board_returns_6x3_matrix(self) -> None:
        rendered_rows = [[np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(3)] for _ in range(6)]
        variant_rows = [
            {"key": "native_raw", "row_header": "RealSense native depth\nno postprocessing"},
            {"key": "native_pt", "row_header": "RealSense native depth\nPhysTwin-like radius-neighbor filter"},
            {"key": "native_enhanced", "row_header": "RealSense native depth\nenhanced radius + component filter"},
            {"key": "ffs_raw", "row_header": "Fast-FoundationStereo depth\nno postprocessing"},
            {"key": "ffs_pt", "row_header": "Fast-FoundationStereo depth\nPhysTwin-like radius-neighbor filter"},
            {"key": "ffs_enhanced", "row_header": "Fast-FoundationStereo depth\nenhanced radius + component filter"},
        ]
        board = build_enhanced_phystwin_postprocess_pcd_board(
            round_label="Round 1",
            frame_idx=0,
            model_config={
                "object_mask_enabled": True,
                "mask_erode_pixels": 0,
                "phystwin_radius_m": 0.01,
                "phystwin_nb_points": 40,
                "enhanced_component_voxel_size_m": 0.01,
                "enhanced_keep_near_main_gap_m": 0.0,
                "row_label_width": 240,
            },
            column_headers=["Cam0", "Cam1", "Cam2"],
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 240)
        self.assertGreater(board.shape[1], 240 + 3 * 60)

    def test_workflow_writes_6x3_board_and_component_stats(self) -> None:
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
            summary = run_enhanced_phystwin_postprocess_pcd_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                frame_idx=0,
                tile_width=80,
                tile_height=60,
                row_label_width=240,
                max_points_per_camera=None,
                phystwin_radius_m=1.0,
                phystwin_nb_points=1,
                enhanced_component_voxel_size_m=0.5,
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
            self.assertEqual(len(render_collector.calls), 18)
            self.assertEqual(round_summary["render_contract"]["rows"], "native_none_pt_enhanced_ffs_none_pt_enhanced")
            self.assertEqual(
                round_summary["row_headers"],
                [
                    "RealSense native depth\nno postprocessing",
                    "RealSense native depth\nPhysTwin-like radius-neighbor filter",
                    "RealSense native depth\nenhanced radius + component filter",
                    "Fast-FoundationStereo depth\nno postprocessing",
                    "Fast-FoundationStereo depth\nPhysTwin-like radius-neighbor filter",
                    "Fast-FoundationStereo depth\nenhanced radius + component filter",
                ],
            )
            self.assertIn("native_enhanced", round_summary["postprocess_stats_by_variant"])
            self.assertEqual(
                round_summary["postprocess_stats_by_variant"]["ffs_enhanced"]["mode"],
                "enhanced_phystwin_like_radius_then_component_filter",
            )
            self.assertEqual(round_summary["model_config"]["enhanced_component_voxel_size_m"], 0.5)


if __name__ == "__main__":
    unittest.main()
