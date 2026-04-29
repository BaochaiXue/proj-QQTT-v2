from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments.ffs_mask_erode_multipage_sweep_pcd_compare import (
    run_ffs_mask_erode_multipage_sweep_pcd_workflow,
)
from data_process.visualization.experiments.ffs_mask_erode_sweep_pcd_compare import (
    build_default_mask_erode_multipage_specs,
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
        return np.full((int(kwargs["height"]), int(kwargs["width"]), 3), 96, dtype=np.uint8)


class FfsMaskErodeMultipageSweepPcdCompareSmokeTest(unittest.TestCase):
    def test_default_page_specs_match_requested_10x3_contract(self) -> None:
        specs = build_default_mask_erode_multipage_specs()
        self.assertEqual([spec["page_id"] for spec in specs], ["page1_erode_01_08", "page2_erode_09_18", "page3_erode_19_28"])
        self.assertEqual([spec["erode_pixels"] for spec in specs], [list(range(1, 9)), list(range(9, 19)), list(range(19, 29))])
        self.assertEqual([spec["include_baselines"] for spec in specs], [True, False, False])
        self.assertEqual([len(spec["erode_pixels"]) + (2 if spec["include_baselines"] else 0) for spec in specs], [10, 10, 10])

    def test_workflow_writes_multipage_result_folder(self) -> None:
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

            page_specs = [
                {
                    "page_id": "page1_erode_01_02",
                    "page_label": "Page 1 | native/original + erode 1..2px",
                    "include_baselines": True,
                    "erode_pixels": [1, 2],
                    "row_contract": "native/original + erode 1..2px",
                },
                {
                    "page_id": "page2_erode_03_04",
                    "page_label": "Page 2 | erode 3..4px",
                    "include_baselines": False,
                    "erode_pixels": [3, 4],
                    "row_contract": "erode 3..4px only",
                },
            ]
            render_collector = _RenderCollector()
            summary = run_ffs_mask_erode_multipage_sweep_pcd_workflow(
                aligned_root=aligned_root,
                output_root=tmp_root / "output",
                frame_idx=0,
                tile_width=80,
                tile_height=60,
                row_label_width=240,
                max_points_per_camera=None,
                phystwin_like_postprocess=False,
                board_page_specs=page_specs,
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
            self.assertEqual(summary["erode_pixels"], [1, 2, 3, 4])
            round_summary = summary["rounds"][0]
            self.assertEqual(len(round_summary["pages"]), 2)
            self.assertEqual(len(round_summary["board_paths"]), 2)
            self.assertEqual(round_summary["render_contract"]["page_layout"], "multipage_10x3")
            self.assertEqual(round_summary["render_contract"]["rows"], "native_depth_ffs_original_ffs_mask_erode_multipage_sweep")
            for board_path in round_summary["board_paths"]:
                self.assertTrue(Path(board_path).is_file())
            self.assertEqual(
                round_summary["pages"][0]["row_headers"][0:2],
                [
                    "RealSense native depth\nobject mask unchanged",
                    "Fast-FoundationStereo depth\nobject mask unchanged",
                ],
            )
            self.assertEqual(
                round_summary["pages"][0]["row_headers"][2:],
                [
                    "Fast-FoundationStereo depth\nobject mask eroded inward 1px",
                    "Fast-FoundationStereo depth\nobject mask eroded inward 2px",
                ],
            )
            self.assertEqual(
                round_summary["pages"][1]["row_headers"],
                [
                    "Fast-FoundationStereo depth\nobject mask eroded inward 3px",
                    "Fast-FoundationStereo depth\nobject mask eroded inward 4px",
                ],
            )
            self.assertFalse(round_summary["pages"][1]["include_baselines"])
            self.assertEqual(len(render_collector.calls), 18)
            self.assertIn("ffs_erode_4px", round_summary["fused_point_counts"])
            for call in render_collector.calls:
                self.assertEqual(int(call["width"]), 80)
                self.assertEqual(int(call["height"]), 60)


if __name__ == "__main__":
    unittest.main()
