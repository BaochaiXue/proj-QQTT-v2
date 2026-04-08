from __future__ import annotations

import unittest

from data_process.visualization.turntable_compare import build_render_output_specs


class DualRenderOutputSmokeTest(unittest.TestCase):
    def test_render_specs_include_geom_rgb_and_support_outputs(self) -> None:
        specs = build_render_output_specs(
            geom_render_mode="neutral_gray_shaded",
            render_both_modes=True,
        )
        self.assertEqual([spec["name"] for spec in specs], ["geom", "rgb", "support"])
        self.assertEqual(specs[0]["video_name"], "orbit_compare_geom.mp4")
        self.assertEqual(specs[1]["sheet_name"], "turntable_keyframes_rgb.png")
        self.assertEqual(specs[2]["video_name"], "orbit_compare_support.mp4")


if __name__ == "__main__":
    unittest.main()
