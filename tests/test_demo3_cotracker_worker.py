from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.cotracker3_overlay_worker import (
    CoTracker3OverlayWorker,
    LatestTrackingOverlaySlot,
    TrackingOverlayInputPacket,
    TrackingOverlayPacket,
)
from qqtt.tracking.base import TrackingResult


class _FakeOnlineBackend:
    name = "fake_cotracker3_online"

    def __init__(self, *, window_len: int = 16, step: int = 8) -> None:
        self.window_len = int(window_len)
        self.step = int(step)
        self.frames_seen = 0
        self.last_published_frame_count = 0
        self.update_calls = 0
        self.query_points_yx = np.empty((0, 2), dtype=np.float32)

    def initialize(self, frames, query_points_yx, masks=None) -> None:
        _ = frames, masks
        self.query_points_yx = np.asarray(query_points_yx, dtype=np.float32)

    def update(self, frame) -> TrackingResult:
        _ = np.asarray(frame)
        self.frames_seen += 1
        self.update_calls += 1
        n = int(len(self.query_points_yx))
        if self.frames_seen < self.window_len:
            return TrackingResult(
                tracks_yx=np.empty((0, n, 2), dtype=np.float32),
                visibility=np.empty((0, n), dtype=np.float32),
                backend=self.name,
                query_points_yx=self.query_points_yx,
                stats={"stream_status": "buffering"},
            )
        if self.last_published_frame_count and self.frames_seen - self.last_published_frame_count < self.step:
            return TrackingResult(
                tracks_yx=np.empty((0, n, 2), dtype=np.float32),
                visibility=np.empty((0, n), dtype=np.float32),
                backend=self.name,
                query_points_yx=self.query_points_yx,
                stats={"stream_status": "waiting_for_step"},
            )
        self.last_published_frame_count = self.frames_seen
        tracks = np.repeat(self.query_points_yx[None, :, :], self.window_len, axis=0).astype(np.float32)
        tracks[-1, :, 0] += float(self.frames_seen)
        visibility = np.ones((self.window_len, n), dtype=np.float32)
        return TrackingResult(
            tracks_yx=tracks,
            visibility=visibility,
            backend=self.name,
            query_points_yx=self.query_points_yx,
            stats={
                "stream_status": "published",
                "chunk_start_idx": self.frames_seen - self.window_len,
                "chunk_end_idx": self.frames_seen - 1,
                "model_run_ms": 1.25,
            },
        )


class Demo3CoTrackerWorkerTest(unittest.TestCase):
    def _packet(self, frame_idx: int) -> TrackingOverlayInputPacket:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.uint8)
        return TrackingOverlayInputPacket(
            group_id=frame_idx,
            frame_idx=frame_idx,
            timestamp_s=float(frame_idx) / 30.0,
            rgb_by_camera={0: rgb},
            mask_by_camera={0: mask},
        )

    def test_fake_backend_receives_frames_one_by_one_and_publishes_on_window(self) -> None:
        backend = _FakeOnlineBackend()
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            query_count=10,
            overlay_max_points_per_camera=5,
        )

        for frame_idx in range(15):
            self.assertIsNone(worker.process_group(self._packet(frame_idx)))
        self.assertEqual(backend.update_calls, 15)
        self.assertIsNone(worker.latest_overlay())

        overlay = worker.process_group(self._packet(15))
        self.assertIsNotNone(overlay)
        self.assertEqual(backend.update_calls, 16)
        self.assertEqual(overlay.publish_range, (0, 15))  # type: ignore[union-attr]
        self.assertEqual(overlay.camera_tracks_yx[0].shape, (5, 2))  # type: ignore[union-attr]
        self.assertEqual(worker.latest_overlay().seq, 15)  # type: ignore[union-attr]

    def test_later_publish_occurs_every_step_frames(self) -> None:
        backend = _FakeOnlineBackend()
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            query_count=10,
            overlay_max_points_per_camera=5,
        )
        for frame_idx in range(16):
            worker.process_group(self._packet(frame_idx))

        for frame_idx in range(16, 23):
            self.assertIsNone(worker.process_group(self._packet(frame_idx)))
        second = worker.process_group(self._packet(23))

        self.assertIsNotNone(second)
        self.assertEqual(second.publish_range, (8, 23))  # type: ignore[union-attr]
        self.assertEqual(worker.published_packets, 2)

    def test_latest_slot_drops_old_overlay_packets(self) -> None:
        slot = LatestTrackingOverlaySlot()
        first = TrackingOverlayPacket(
            group_id=1,
            frame_idx=1,
            timestamp_s=0.0,
            camera_tracks_yx={},
            camera_visibility={},
        )
        second = TrackingOverlayPacket(
            group_id=2,
            frame_idx=2,
            timestamp_s=0.0,
            camera_tracks_yx={},
            camera_visibility={},
        )

        slot.publish(first)
        slot.publish(second)

        self.assertEqual(slot.snapshot()["dropped"], 1)
        self.assertEqual(slot.get_optional().seq, 2)  # type: ignore[union-attr]
        self.assertEqual(slot.take_latest().seq, 2)  # type: ignore[union-attr]
        self.assertIsNone(slot.take_latest())

    def test_renderer_can_proceed_when_overlay_slot_empty(self) -> None:
        slot = LatestTrackingOverlaySlot()

        def render_tick() -> bool:
            _overlay = slot.get_optional()
            return True

        self.assertTrue(render_tick())
        self.assertIsNone(slot.get_optional())


if __name__ == "__main__":
    unittest.main()
