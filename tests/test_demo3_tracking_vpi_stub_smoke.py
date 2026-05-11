from __future__ import annotations

import unittest

import numpy as np

from qqtt.tracking.base import BackendUnavailableError
from qqtt.tracking.backends.vpi_lk import VpiLkBackend


class Demo3TrackingVpiStubSmokeTest(unittest.TestCase):
    def test_vpi_lk_missing_dependency_is_clean(self) -> None:
        backend = VpiLkBackend()
        availability = backend.availability()
        if availability.available:
            frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
            result = backend.track_sequence(
                frames_rgb=frames,
                query_points_yx=np.array([[3.0, 3.0]], dtype=np.float32),
                camera_idx=0,
            )
            self.assertEqual(result.tracks_yx.shape, (2, 1, 2))
        else:
            self.assertFalse(backend.is_available())
            self.assertIn("vpi", availability.reason.lower())
            with self.assertRaises(BackendUnavailableError):
                backend.track_sequence(
                    frames_rgb=[np.zeros((8, 8, 3), dtype=np.uint8)],
                    query_points_yx=np.array([[3.0, 3.0]], dtype=np.float32),
                    camera_idx=0,
                )


if __name__ == "__main__":
    unittest.main()
