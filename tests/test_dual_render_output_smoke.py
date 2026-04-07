from __future__ import annotations

import unittest

from data_process.visualization.turntable_compare import build_render_output_specs


class DualRenderOutputSmokeTest(unittest.TestCase):
    def test_dual_render_specs_include_geom_and_rgb_outputs(self) -> None:
        specs = build_render_output_specs(
            geom_render_mode="neutral_gray_shaded",
            render_both_modes=True,
        )
        self.assertEqual([spec["name"] for spec in specs], ["geom", "rgb"])
        self.assertEqual(specs[0]["video_name"], "orbit_compare_geom.mp4")
        self.assertEqual(specs[1]["sheet_name"], "turntable_keyframes_rgb.png")


if __name__ == "__main__":
    unittest.main()
