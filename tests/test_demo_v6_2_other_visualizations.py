"""Tests for Demo v6.2 visualization artifact destinations."""

from __future__ import annotations

import unittest
from pathlib import Path

from demo_v6_2.others import lbs_shape_prior_from_chunks
from demo_v6_2.others import render_online_depth_realsense
from demo_v6_2.others import view_shape_prior_outputs


EXPECTED_ARTIFACT_DIR = Path("demo_v6_2/others/obj_shape_asap_outputs")


class VisualizationDefaultsTests(unittest.TestCase):
    def test_generated_artifacts_default_to_demo_v6_2(self) -> None:
        self.assertEqual(
            render_online_depth_realsense.DEFAULT_OUTPUT_DIR,
            EXPECTED_ARTIFACT_DIR,
        )
        self.assertEqual(
            lbs_shape_prior_from_chunks.DEFAULT_ARTIFACT_DIR,
            EXPECTED_ARTIFACT_DIR,
        )
        self.assertEqual(
            lbs_shape_prior_from_chunks.DEFAULT_PREVIEW_VIDEO_PATH.parent,
            EXPECTED_ARTIFACT_DIR,
        )
        self.assertEqual(
            lbs_shape_prior_from_chunks.DEFAULT_CONTACT_SHEET_PATH.parent,
            EXPECTED_ARTIFACT_DIR,
        )
        self.assertEqual(
            render_online_depth_realsense.DEFAULT_GRID_FRAME_COUNT,
            35,
        )
        self.assertEqual(
            render_online_depth_realsense.DEFAULT_GRID_COLUMNS,
            7,
        )

    def test_user_facing_descriptions_identify_demo_v6_2(self) -> None:
        parsers = [
            render_online_depth_realsense.build_parser(),
            lbs_shape_prior_from_chunks.build_parser(),
            view_shape_prior_outputs.build_parser(),
        ]

        for parser in parsers:
            with self.subTest(description=parser.description):
                self.assertIn("Demo v6.2", parser.description)


if __name__ == "__main__":
    unittest.main()
