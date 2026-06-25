import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqtt.utils.config import Config


class ConfigMetadataTest(unittest.TestCase):
    def test_apply_camera_metadata_updates_fps_and_num_substeps(self):
        cfg = Config()
        cfg.dt = 5e-5
        cfg.FPS = 30
        cfg.num_substeps = 667

        cfg.apply_camera_metadata(
            {
                "fps": 5,
                "intrinsics": [[[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]],
                "WH": [640, 480],
            }
        )

        self.assertEqual(cfg.FPS, 5)
        self.assertEqual(cfg.num_substeps, round(1.0 / 5.0 / 5e-5))
        np.testing.assert_allclose(
            cfg.intrinsics,
            np.array([[[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]]),
        )
        self.assertEqual(cfg.WH, [640, 480])


if __name__ == "__main__":
    unittest.main()
