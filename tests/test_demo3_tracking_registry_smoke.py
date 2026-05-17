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

    def test_cotracker_online_update_uses_16_frame_window_and_8_frame_step(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch is not installed: {exc}")

        from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend

        class _FakeOnlineModel:
            step = 8

            def __init__(self) -> None:
                self.calls: list[tuple[bool, int]] = []
                self.num_points = 0

            def __call__(
                self,
                *,
                video_chunk,
                is_first_step=False,
                queries=None,
                grid_size=0,
                add_support_grid=False,
            ):
                _ = grid_size, add_support_grid
                self.calls.append((bool(is_first_step), int(video_chunk.shape[1])))
                if is_first_step:
                    self.num_points = int(queries.shape[1])
                    return None, None
                batch, frames = int(video_chunk.shape[0]), int(video_chunk.shape[1])
                tracks = torch.zeros((batch, frames, self.num_points, 2), dtype=torch.float32, device=video_chunk.device)
                visibility = torch.ones((batch, frames, self.num_points), dtype=torch.float32, device=video_chunk.device)
                return tracks, visibility

        model = _FakeOnlineModel()
        backend = CoTracker3OnlineBackend(device="cpu", model=model)
        backend.initialize([], np.array([[2.0, 5.0], [3.0, 6.0]], dtype=np.float32))
        frame = np.zeros((4, 6, 3), dtype=np.uint8)

        for _ in range(15):
            result = backend.update(frame)
            self.assertEqual(result.stats["stream_status"], "buffering")
            self.assertEqual(result.tracks_yx.shape, (0, 2, 2))
        first = backend.update(frame)
        self.assertEqual(first.stats["stream_status"], "published")
        self.assertEqual(first.stats["online_window_len"], 16)
        self.assertEqual(first.stats["online_step"], 8)
        self.assertEqual(first.stats["chunk_start_idx"], 0)
        self.assertEqual(first.stats["chunk_end_idx"], 15)
        self.assertEqual(first.tracks_yx.shape, (16, 2, 2))
        self.assertEqual(model.calls, [(True, 16), (False, 16)])

        for _ in range(7):
            result = backend.update(frame)
            self.assertEqual(result.stats["stream_status"], "waiting_for_step")
        second = backend.update(frame)
        self.assertEqual(second.stats["stream_status"], "published")
        self.assertEqual(second.stats["chunk_start_idx"], 8)
        self.assertEqual(second.stats["chunk_end_idx"], 23)
        self.assertEqual(second.tracks_yx.shape, (16, 2, 2))
        self.assertEqual(model.calls, [(True, 16), (False, 16), (False, 16)])

    def test_cotracker_track_sequence_replays_online_update_frame_by_frame(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch is not installed: {exc}")

        from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend

        class _FakeOnlineModel:
            step = 8

            def __init__(self) -> None:
                self.calls: list[tuple[bool, int]] = []
                self.query_xy = None

            def __call__(
                self,
                *,
                video_chunk,
                is_first_step=False,
                queries=None,
                grid_size=0,
                add_support_grid=False,
            ):
                _ = grid_size, add_support_grid
                self.calls.append((bool(is_first_step), int(video_chunk.shape[1])))
                if is_first_step:
                    self.query_xy = queries[:, :, 1:].float()
                    return None, None
                if self.query_xy is None:
                    raise RuntimeError("queries were not initialized")
                batch, frames = int(video_chunk.shape[0]), int(video_chunk.shape[1])
                tracks = self.query_xy[:, None, :, :].repeat(1, frames, 1, 1)
                visibility = torch.ones((batch, frames, self.query_xy.shape[1]), dtype=torch.float32, device=video_chunk.device)
                return tracks, visibility

        model = _FakeOnlineModel()
        backend = CoTracker3OnlineBackend(device="cpu", model=model)
        frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(24)]
        result = backend.track_sequence(
            frames_rgb=frames,
            query_points_yx=np.array([[2.0, 5.0], [3.0, 4.0]], dtype=np.float32),
            camera_idx=1,
        )

        self.assertEqual(result.stats["mode"], "cotracker3_online_streaming_replay")
        self.assertEqual(result.stats["published_chunks"], 2)
        self.assertEqual(result.stats["stream_tail_unpublished_frames"], 0)
        self.assertEqual(result.tracks_yx.shape, (24, 2, 2))
        self.assertEqual(model.calls, [(True, 16), (False, 16), (False, 16)])
        np.testing.assert_allclose(result.tracks_yx[:, 0, :], np.array([[2.0, 5.0]], dtype=np.float32).repeat(24, axis=0))
        np.testing.assert_allclose(result.visibility, np.ones((24, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
