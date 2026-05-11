from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qqtt.tracking.base import BackendUnavailableError
from qqtt.tracking.backends.nvofa import NvofaBackend


class Demo3TrackingNvofaStubSmokeTest(unittest.TestCase):
    def test_missing_nvofa_reports_unavailable_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = NvofaBackend(external_root=Path(tmp_dir))
            availability = backend.availability()
            if availability.available:
                self.skipTest(f"NVOFA binding/helper is available on this machine: {availability.reason}")
            self.assertFalse(availability.available)
            self.assertIn("not", availability.reason.lower())
            with self.assertRaises(BackendUnavailableError):
                backend.track_sequence(
                    frames_rgb=[np.zeros((4, 4, 3), dtype=np.uint8)],
                    query_points_yx=np.array([[1.0, 1.0]], dtype=np.float32),
                    camera_idx=0,
                )


if __name__ == "__main__":
    unittest.main()
