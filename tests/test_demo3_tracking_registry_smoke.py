from __future__ import annotations

import unittest

import numpy as np

from qqtt.tracking.registry import BACKEND_NAMES, available_backend_names, check_backend_availability, create_backend


class Demo3TrackingRegistrySmokeTest(unittest.TestCase):
    def test_registry_reports_planned_backends_without_crashing(self) -> None:
        self.assertIn("cotracker3_online", BACKEND_NAMES)
        self.assertIn("nvofa", available_backend_names())
        availability = check_backend_availability(["nvofa"])
        self.assertFalse(availability["nvofa"].available)
        self.assertIn("not", availability["nvofa"].reason.lower())
        self.assertFalse(create_backend("nvofa").is_available())

    def test_fake_backend_returns_deterministic_yx_tracks(self) -> None:
        backend = create_backend("fake")
        frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
        result = backend.track_sequence(
            frames_rgb=frames,
            query_points_yx=np.array([[2.0, 5.0]], dtype=np.float32),
            camera_idx=0,
        )
        self.assertEqual(result.tracks_yx.shape, (2, 1, 2))
        np.testing.assert_allclose(result.tracks_yx[:, 0, :], np.array([[2.0, 5.0], [2.0, 5.0]], dtype=np.float32))

    def test_cotracker_backend_converts_model_xy_output_to_yx(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch is not installed: {exc}")

        from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend

        class _FakeModel:
            def __call__(self, video, *, queries, is_online=False):
                _ = is_online
                batch, frames = video.shape[0], video.shape[1]
                num_points = queries.shape[1]
                xy = queries[:, :, 1:].float()
                tracks = xy[:, None, :, :].repeat(1, frames, 1, 1)
                visibility = torch.ones((batch, frames, num_points), dtype=torch.float32, device=video.device)
                return tracks, visibility

        backend = CoTracker3OnlineBackend(device="cpu", model=_FakeModel())
        frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
        result = backend.track_sequence(
            frames_rgb=frames,
            query_points_yx=np.array([[2.0, 5.0]], dtype=np.float32),
            camera_idx=0,
        )

        self.assertEqual(result.tracks_yx.shape, (2, 1, 2))
        np.testing.assert_allclose(result.tracks_yx[:, 0, :], np.array([[2.0, 5.0], [2.0, 5.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
