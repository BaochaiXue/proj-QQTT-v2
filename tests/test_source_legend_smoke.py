from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2

from data_process.visualization.source_compare import build_source_legend_image, write_source_legend_image


class SourceLegendSmokeTest(unittest.TestCase):
    def test_source_legend_generation_and_write(self) -> None:
        image = build_source_legend_image()
        self.assertEqual(image.ndim, 3)
        self.assertGreater(int((image > 0).sum()), 0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "legend.png"
            written_path = write_source_legend_image(output_path)
            loaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            self.assertEqual(str(output_path), written_path)
            self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
