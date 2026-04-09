from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.depth_diagnostics import annotate_rois, normalize_roi_entries, parse_named_roi_spec


class DepthPanelRoiOverlaySmokeTest(unittest.TestCase):
    def test_named_roi_parse_and_overlay_changes_pixels(self) -> None:
        roi_entry = parse_named_roi_spec("Head:10,12,28,30")
        self.assertEqual(roi_entry["name"], "Head")
        normalized = normalize_roi_entries([roi_entry], image_shape=(64, 64))
        self.assertEqual(normalized[0]["name"], "Head")

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        annotated = annotate_rois(image, normalized)
        self.assertGreater(int(np.abs(annotated.astype(np.int32) - image.astype(np.int32)).sum()), 0)
        self.assertTrue(np.any(annotated[12:30, 10] != 0))
