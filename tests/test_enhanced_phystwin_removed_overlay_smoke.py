from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments.enhanced_phystwin_removed_overlay import (
    build_enhanced_phystwin_removed_overlay_board,
    run_enhanced_phystwin_removed_overlay_workflow,
)
from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
    _apply_enhanced_phystwin_like_postprocess,
    _apply_enhanced_phystwin_like_postprocess_with_trace,
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
                "camera_idx": int(kwargs["camera_idx"]),
            }
        )
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 116, dtype=np.uint8)


class EnhancedPhystwinRemovedOverlaySmokeTest(unittest.TestCase):
    def test_trace_masks_match_existing_enhanced_filter(self) -> None:
        main = np.asarray(
            [[x, y, z] for x in (0.0, 0.004, 0.008) for y in (0.0, 0.004, 0.008) for z in (0.0, 0.004, 0.008)],
            dtype=np.float32,
        )
        floating = np.asarray(
            [[0.12, 0.0, 0.0], [0.124, 0.0, 0.0], [0.128, 0.0, 0.0], [0.132, 0.0, 0.0]],
            dtype=np.float32,
        )
        radius_outlier = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32)
        points = np.concatenate([main, floating, radius_outlier], axis=0)
        colors = np.full((len(points), 3), 255, dtype=np.uint8)

        traced_points, traced_colors, traced_stats, trace = _apply_enhanced_phystwin_like_postprocess_with_trace(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=0.02,
            nb_points=2,
            component_voxel_size_m=0.01,
            keep_near_main_gap_m=0.0,
        )
        legacy_points, legacy_colors, legacy_stats = _apply_enhanced_phystwin_like_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=0.02,
            nb_points=2,
            component_voxel_size_m=0.01,
            keep_near_main_gap_m=0.0,
        )

        np.testing.assert_allclose(traced_points, legacy_points)
        np.testing.assert_array_equal(traced_colors, legacy_colors)
        self.assertEqual(traced_stats["output_point_count"], legacy_stats["output_point_count"])
        self.assertEqual(int(np.count_nonzero(trace["kept_mask"])), len(main))
        self.assertEqual(int(np.count_nonzero(trace["radius_removed_mask"])), len(radius_outlier))
        self.assertEqual(int(np.count_nonzero(trace["component_removed_mask"])), len(floating))
        self.assertEqual(int(np.count_nonzero(trace["removed_mask"])), len(floating) + len(radius_outlier))

    def test_build_board_returns_4x3_matrix(self) -> None:
        image_rows = [[np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(3)] for _ in range(4)]
        board = build_enhanced_phystwin_removed_overlay_board(
            round_label="Round 1",
            frame_idx=0,
            model_config={
                "highlight_scope": "all",
                "phystwin_radius_m": 0.01,
                "phystwin_nb_points": 40,
                "enhanced_component_voxel_size_m": 0.01,
                "highlight_alpha": 0.65,
                "row_label_width": 220,
            },
            column_headers=["Cam0", "Cam1", "Cam2"],
            image_rows=image_rows,
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 120)
        self.assertGreater(board.shape[1], 220 + 3 * 60)

    def test_workflow_writes_4x3_overlay_and_summary(self) -> None:
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

            render_collector = _RenderCollector()
            summary = run_enhanced_phystwin_removed_overlay_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                frame_idx=0,
                tile_width=80,
                tile_height=60,
                row_label_width=220,
                max_points_per_camera=None,
                phystwin_radius_m=0.001,
                phystwin_nb_points=2,
                enhanced_component_voxel_size_m=0.5,
                render_frame_fn=render_collector,
                round_specs=[
                    {
                        "round_id": "round1",
                        "round_label": "Round 1",
                        "native_case_ref": "static/native_case_round1",
                        "ffs_case_ref": "static/ffs_case_round1",
                        "mask_root": mask_root,
                    }
                ],
            )

            round_summary = summary["rounds"][0]
            self.assertTrue(Path(round_summary["board_path"]).is_file())
            self.assertEqual(len(render_collector.calls), 3)
            self.assertEqual(round_summary["render_contract"]["rows"], "rgb_mask_pcd_removed_depth_removed_rgb_removed")
            self.assertGreater(round_summary["total_removed_point_count"], 0)
            self.assertTrue(any(item["removed_overlay_pixel_count"] > 0 for item in round_summary["per_camera"]))
            self.assertEqual(round_summary["model_config"]["highlight_scope"], "all")


if __name__ == "__main__":
    unittest.main()
