from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qqtt.tracking.base import TrackingResult
from qqtt.tracking.io import (
    load_cotracker_like_npz,
    load_phystwin_tracking_npz,
    save_cotracker_like_npz,
    save_phystwin_tracking_npz,
)


class Demo3TrackingIoSmokeTest(unittest.TestCase):
    def test_phystwin_npz_save_load_preserves_yx_contract(self) -> None:
        tracks = np.array([[[10.0, 20.0]]], dtype=np.float32)
        visibility = np.array([[1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_phystwin_tracking_npz(
                Path(tmp_dir) / "0.npz",
                tracks_yx=tracks,
                visibility=visibility,
                metadata={"backend": "unit", "camera_idx": 0, "coordinate_order": "yx"},
            )
            loaded = load_phystwin_tracking_npz(path)

        np.testing.assert_allclose(loaded.tracks_yx, tracks)
        np.testing.assert_allclose(loaded.visibility, visibility)
        self.assertEqual(loaded.coordinate_order, "yx")
        self.assertEqual(loaded.stats["metadata"]["coordinate_order"], "yx")

    def test_cotracker_like_wrapper_keeps_metadata_and_query_points(self) -> None:
        result = TrackingResult(
            tracks_yx=np.array([[[2.0, 5.0], [3.0, 6.0]]], dtype=np.float32),
            visibility=np.array([[1.0, 0.0]], dtype=np.float32),
            confidence=np.array([[0.9, 0.1]], dtype=np.float32),
            backend="unit_backend",
            camera_idx=1,
            query_points_yx=np.array([[2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_cotracker_like_npz(result, Path(tmp_dir) / "1.npz", metadata={"image_size": [8, 10]})
            loaded, metadata = load_cotracker_like_npz(path)

        np.testing.assert_allclose(loaded.tracks_yx, result.tracks_yx)
        np.testing.assert_allclose(loaded.query_points_yx, result.query_points_yx)
        self.assertEqual(metadata["coordinate_order"], "yx")
        self.assertEqual(metadata["camera_idx"], 1)
        self.assertEqual(metadata["image_size"], [8, 10])


if __name__ == "__main__":
    unittest.main()
