from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.single_view_shape_align import ShapeAlignmentConfig
from qqtt.demo.single_view_shape_align import _validate_alignment


class SingleViewShapeAlignTest(unittest.TestCase):
    def test_observation_p95_is_diagnostic_not_acceptance_gate(self) -> None:
        aligned = np.asarray(
            [
                [-1.0, 0.0, -0.5],
                [1.0, 0.0, -0.5],
                [0.0, -1.0, -0.1],
                [0.0, 1.0, -0.1],
            ],
            dtype=np.float32,
        )
        observation = np.asarray(
            [
                [-100.0, 0.0, -0.5],
                [100.0, 0.0, -0.5],
                [0.0, -1.0, -0.1],
                [0.0, 1.0, -0.1],
            ],
            dtype=np.float32,
        )

        valid, payload = _validate_alignment(
            aligned,
            observation,
            config=ShapeAlignmentConfig(),
        )

        self.assertTrue(valid)
        self.assertGreater(payload["observation_to_aligned_p95_m"], 0.06)
        self.assertNotIn("coverage_valid", payload)
        self.assertNotIn("max_observation_to_aligned_p95_m", payload)


if __name__ == "__main__":
    unittest.main()
