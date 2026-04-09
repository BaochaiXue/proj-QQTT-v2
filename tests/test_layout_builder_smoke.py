from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.layouts import compose_grid_2x3, compose_hero_compare, compose_keyframe_sheet, compose_side_by_side_large


class LayoutBuilderSmokeTest(unittest.TestCase):
    def test_layout_helpers_build_expected_canvases(self) -> None:
        panel = np.full((120, 160, 3), 90, dtype=np.uint8)
        grid = compose_grid_2x3(
            title="Example",
            column_headers=["Cam0", "Cam1", "Cam2"],
            row_headers=["Native", "FFS"],
            native_images=[panel, panel, panel],
            ffs_images=[panel, panel, panel],
        )
        side_by_side = compose_side_by_side_large(
            title_lines=["Example", "frame_idx=0"],
            native_image=np.full((180, 220, 3), 60, dtype=np.uint8),
            ffs_image=np.full((180, 220, 3), 120, dtype=np.uint8),
            overview_inset=np.full((120, 160, 3), 30, dtype=np.uint8),
        )
        hero = compose_hero_compare(
            title_lines=["Example | Native vs FFS", "frame=0 | orbit=+0.0 deg"],
            native_image=np.full((180, 220, 3), 60, dtype=np.uint8),
            ffs_image=np.full((180, 220, 3), 120, dtype=np.uint8),
            overview_inset=np.full((120, 160, 3), 30, dtype=np.uint8),
        )
        sheet = compose_keyframe_sheet([side_by_side, side_by_side], max_width=1600, max_height=1200)
        self.assertEqual(grid.ndim, 3)
        self.assertEqual(side_by_side.ndim, 3)
        self.assertEqual(hero.ndim, 3)
        self.assertEqual(sheet.ndim, 3)
        self.assertGreater(sheet.shape[0], 0)
        self.assertGreater(sheet.shape[1], 0)
