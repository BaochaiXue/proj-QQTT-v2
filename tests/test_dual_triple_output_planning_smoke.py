from __future__ import annotations

import unittest

from data_process.visualization.turntable_compare import build_render_output_specs


class DualTripleOutputPlanningSmokeTest(unittest.TestCase):
    def test_geom_and_support_remain_when_rgb_reference_is_disabled(self) -> None:
        specs = build_render_output_specs(
            geom_render_mode="neutral_gray_shaded",
            render_both_modes=False,
        )
        self.assertEqual([spec["name"] for spec in specs], ["geom", "support"])


if __name__ == "__main__":
    unittest.main()
