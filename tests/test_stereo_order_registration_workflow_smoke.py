from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from data_process.visualization.stereo_audit import run_stereo_order_registration_workflow


def _make_camera_cloud(camera_idx: int, points: np.ndarray, colors: np.ndarray, *, color_path: Path) -> dict:
    return {
        "camera_idx": int(camera_idx),
        "serial": f"serial-{camera_idx}",
        "points": np.asarray(points, dtype=np.float32),
        "colors": np.asarray(colors, dtype=np.uint8),
        "source_camera_idx": np.full((len(points),), int(camera_idx), dtype=np.int16),
        "source_serial": np.full((len(points),), f"serial-{camera_idx}", dtype=object),
        "K_color": np.array([[120.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "c2w": np.eye(4, dtype=np.float32),
        "color_path": str(color_path),
    }


def _synthetic_turntable_state(tmp_root: Path) -> tuple[dict, list[dict]]:
    table = np.array(
        [
            [-0.15, -0.15, 0.0],
            [0.15, -0.15, 0.0],
            [-0.15, 0.15, 0.0],
            [0.15, 0.15, 0.0],
        ],
        dtype=np.float32,
    )
    object_points = np.array(
        [
            [-0.04, -0.02, 0.06],
            [0.03, -0.02, 0.06],
            [-0.03, 0.03, 0.12],
            [0.04, 0.02, 0.10],
            [0.00, 0.00, 0.18],
        ],
        dtype=np.float32,
    )
    table_colors = np.tile(np.array([[170, 150, 120]], dtype=np.uint8), (len(table), 1))
    object_colors = np.tile(np.array([[110, 145, 195]], dtype=np.uint8), (len(object_points), 1))
    native_camera_clouds = []
    ffs_camera_clouds = []
    swapped_camera_clouds = []
    for camera_idx, x_jitter in enumerate((0.0, 0.004, -0.003)):
        color_path = tmp_root / f"cam{camera_idx}.png"
        cv2.imwrite(str(color_path), np.full((48, 64, 3), 160, dtype=np.uint8))
        jitter = np.array([x_jitter, 0.002 * camera_idx, 0.0], dtype=np.float32)
        native_points = np.concatenate([table, object_points + jitter], axis=0)
        ffs_points = np.concatenate([table, object_points + jitter * 1.4], axis=0)
        swapped_points = np.concatenate([table, object_points + jitter * 2.4 + np.array([0.0, 0.0, 0.012], dtype=np.float32)], axis=0)
        colors = np.concatenate([table_colors, object_colors], axis=0)
        native_camera_clouds.append(_make_camera_cloud(camera_idx, native_points, colors, color_path=color_path))
        ffs_camera_clouds.append(_make_camera_cloud(camera_idx, ffs_points, colors, color_path=color_path))
        swapped_camera_clouds.append(_make_camera_cloud(camera_idx, swapped_points, colors, color_path=color_path))

    native_object_camera_clouds = [
        {**cloud, "points": np.asarray(cloud["points"], dtype=np.float32)[len(table):], "colors": np.asarray(cloud["colors"], dtype=np.uint8)[len(table):]}
        for cloud in native_camera_clouds
    ]
    ffs_object_camera_clouds = [
        {**cloud, "points": np.asarray(cloud["points"], dtype=np.float32)[len(table):], "colors": np.asarray(cloud["colors"], dtype=np.uint8)[len(table):]}
        for cloud in ffs_camera_clouds
    ]
    scene = {
        "native_points": np.concatenate([item["points"] for item in native_camera_clouds], axis=0),
        "native_colors": np.concatenate([item["colors"] for item in native_camera_clouds], axis=0),
        "ffs_points": np.concatenate([item["points"] for item in ffs_camera_clouds], axis=0),
        "ffs_colors": np.concatenate([item["colors"] for item in ffs_camera_clouds], axis=0),
        "native_object_points": np.concatenate([item["points"] for item in native_object_camera_clouds], axis=0),
        "ffs_object_points": np.concatenate([item["points"] for item in ffs_object_camera_clouds], axis=0),
        "native_object_camera_clouds": native_object_camera_clouds,
        "ffs_object_camera_clouds": ffs_object_camera_clouds,
        "plane_point": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "plane_normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "crop_bounds": {
            "min": np.array([-0.18, -0.18, -0.02], dtype=np.float32),
            "max": np.array([0.18, 0.18, 0.25], dtype=np.float32),
        },
        "object_roi_bounds": {
            "min": np.array([-0.05, -0.04, 0.05], dtype=np.float32),
            "max": np.array([0.05, 0.04, 0.20], dtype=np.float32),
        },
        "focus_point": np.array([0.0, 0.0, 0.10], dtype=np.float32),
    }
    selection = {
        "same_case_mode": False,
        "native_case_dir": Path("native_case"),
        "ffs_case_dir": Path("ffs_case"),
        "native_frame_idx": 0,
        "ffs_frame_idx": 0,
        "camera_ids": [0, 1, 2],
    }
    refinement = {
        "final_ffs_masks": None,
        "pass2_refinement_valid": True,
    }
    return {
        "selection": selection,
        "manual_image_roi_by_camera": {0: (4, 4, 60, 44), 1: (4, 4, 60, 44), 2: (4, 4, 60, 44)},
        "scene": scene,
        "refinement": refinement,
    }, swapped_camera_clouds


class StereoOrderRegistrationWorkflowSmokeTest(unittest.TestCase):
    def test_workflow_writes_only_main_board_and_summary_by_default_and_reuses_shared_view_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            turntable_state, swapped_camera_clouds = _synthetic_turntable_state(Path(tmp_dir))
            output_dir = Path(tmp_dir) / "registration_board"
            with mock.patch("data_process.visualization.stereo_audit._build_turntable_scene", return_value=turntable_state), \
                    mock.patch("data_process.visualization.stereo_audit._build_swapped_ffs_camera_clouds", return_value=swapped_camera_clouds):
                run_stereo_order_registration_workflow(
                    aligned_root=Path(tmp_dir),
                    output_dir=output_dir,
                    ffs_repo=Path("repo"),
                    model_path=Path("model"),
                    panel_width=200,
                    panel_height=160,
                )

            top_level = sorted(path.name for path in output_dir.iterdir())
            self.assertEqual(top_level, ["01_stereo_order_registration_board.png", "match_board_summary.json"])
            summary = json.loads((output_dir / "match_board_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["top_level_output"], str((output_dir / "01_stereo_order_registration_board.png").resolve()))
            self.assertIsNone(summary["closeup_output"])
            self.assertEqual([item["label"] for item in summary["board_views"]], ["Oblique", "Top", "Front", "Side"])
            top_scale = summary["board_views"][1]["ortho_scale"]
            front_scale = summary["board_views"][2]["ortho_scale"]
            side_scale = summary["board_views"][3]["ortho_scale"]
            for row_name in ("native", "ffs_current", "ffs_swapped"):
                row_metrics = summary["board_row_metrics"][row_name]
                self.assertEqual(row_metrics[1]["ortho_scale"], top_scale)
                self.assertEqual(row_metrics[2]["ortho_scale"], front_scale)
                self.assertEqual(row_metrics[3]["ortho_scale"], side_scale)
            self.assertEqual(summary["source_color_map_bgr"], {"0": [0, 0, 255], "1": [0, 255, 0], "2": [255, 0, 0]})

    def test_workflow_gates_closeup_and_debug_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            turntable_state, swapped_camera_clouds = _synthetic_turntable_state(Path(tmp_dir))
            output_dir = Path(tmp_dir) / "registration_board"
            with mock.patch("data_process.visualization.stereo_audit._build_turntable_scene", return_value=turntable_state), \
                    mock.patch("data_process.visualization.stereo_audit._build_swapped_ffs_camera_clouds", return_value=swapped_camera_clouds):
                run_stereo_order_registration_workflow(
                    aligned_root=Path(tmp_dir),
                    output_dir=output_dir,
                    ffs_repo=Path("repo"),
                    model_path=Path("model"),
                    panel_width=200,
                    panel_height=160,
                    write_debug=True,
                    write_closeup=True,
                )

            self.assertTrue((output_dir / "02_stereo_order_closeup_board.png").is_file())
            self.assertTrue((output_dir / "debug" / "registration_board_debug.json").is_file())


if __name__ == "__main__":
    unittest.main()
