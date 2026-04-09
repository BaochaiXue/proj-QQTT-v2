from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from data_process.visualization.types import CompareCaseSelection, RenderOutputSpec, ViewConfig


class VisualTypesContractSmokeTest(unittest.TestCase):
    def test_render_output_spec_to_dict(self) -> None:
        spec = RenderOutputSpec(
            name="geom",
            render_mode="neutral_gray_shaded",
            video_name="orbit_compare_geom.mp4",
            gif_name="orbit_compare_geom.gif",
            sheet_name="turntable_keyframes_geom.png",
            frames_dir_name="frames_geom",
        )
        payload = spec.to_dict()
        self.assertEqual(payload["name"], "geom")
        self.assertEqual(payload["render_mode"], "neutral_gray_shaded")
        self.assertTrue(payload["write_video"])

    def test_compare_case_selection_to_dict(self) -> None:
        selection = CompareCaseSelection(
            aligned_root=Path("data"),
            native_case_dir=Path("data/native"),
            ffs_case_dir=Path("data/ffs"),
            same_case_mode=False,
            native_frame_idx=0,
            ffs_frame_idx=0,
            camera_ids=[0, 1, 2],
            serial_numbers=["a", "b", "c"],
            native_c2w=[np.eye(4, dtype=np.float32) for _ in range(3)],
        )
        payload = selection.to_dict()
        self.assertEqual(payload["camera_ids"], [0, 1, 2])
        self.assertEqual(payload["serial_numbers"], ["a", "b", "c"])

    def test_view_config_to_dict(self) -> None:
        view = ViewConfig(
            view_name="oblique",
            label="Oblique",
            center=np.zeros((3,), dtype=np.float32),
            camera_position=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            radius=1.5,
            camera_idx=0,
        )
        payload = view.to_dict()
        self.assertEqual(payload["view_name"], "oblique")
        self.assertEqual(payload["camera_idx"], 0)
