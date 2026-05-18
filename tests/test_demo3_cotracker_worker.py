from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from qqtt.demo.cotracker3_overlay_worker import (
    CoTracker3OverlayThread,
    CoTracker3OverlayWorker,
    LatestTrackingInputSlot,
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
        return self._packet_with_mask(frame_idx, mask)

    def _packet_with_mask(self, frame_idx: int, mask: np.ndarray) -> TrackingOverlayInputPacket:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        return TrackingOverlayInputPacket(
            group_id=frame_idx,
            frame_idx=frame_idx,
            timestamp_s=float(frame_idx) / 30.0,
            rgb_by_camera={0: rgb},
            mask_by_camera={0: mask},
            object_mask_by_camera={0: mask},
            controller_mask_by_camera={0: mask},
        )

    def _packet_with_component_masks(
        self,
        frame_idx: int,
        *,
        object_mask: np.ndarray,
        controller_mask: np.ndarray,
    ) -> TrackingOverlayInputPacket:
        rgb = np.zeros((*object_mask.shape, 3), dtype=np.uint8)
        union = np.asarray(object_mask, dtype=bool) | np.asarray(controller_mask, dtype=bool)
        return TrackingOverlayInputPacket(
            group_id=frame_idx,
            frame_idx=frame_idx,
            timestamp_s=float(frame_idx) / 30.0,
            rgb_by_camera={0: rgb},
            mask_by_camera={0: union},
            object_mask_by_camera={0: object_mask},
            controller_mask_by_camera={0: controller_mask},
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

        before_publish_s = time.perf_counter()
        overlay = worker.process_group(self._packet(15))
        after_publish_s = time.perf_counter()
        self.assertIsNotNone(overlay)
        self.assertEqual(backend.update_calls, 16)
        self.assertEqual(overlay.publish_range, (0, 15))  # type: ignore[union-attr]
        self.assertEqual(overlay.camera_tracks_yx[0].shape, (5, 2))  # type: ignore[union-attr]
        self.assertGreaterEqual(overlay.timestamp_s, before_publish_s)  # type: ignore[union-attr]
        self.assertLessEqual(overlay.timestamp_s, after_publish_s)  # type: ignore[union-attr]
        self.assertEqual(overlay.source_timestamp_s, 15.0 / 30.0)  # type: ignore[union-attr]
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

    def test_empty_first_mask_is_not_cached_forever(self) -> None:
        backend = _FakeOnlineBackend(window_len=2, step=1)
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            query_count=4,
            overlay_max_points_per_camera=4,
        )
        empty = np.zeros((8, 8), dtype=np.uint8)
        nonempty = np.ones((8, 8), dtype=np.uint8)

        self.assertIsNone(worker.process_group(self._packet_with_mask(0, empty)))
        self.assertEqual(backend.update_calls, 0)
        self.assertIsNone(worker.process_group(self._packet_with_mask(1, nonempty)))
        overlay = worker.process_group(self._packet_with_mask(2, nonempty))

        self.assertIsNotNone(overlay)
        self.assertEqual(backend.update_calls, 2)
        self.assertEqual(overlay.camera_tracks_yx[0].shape, (4, 2))  # type: ignore[union-attr]

    def test_object_only_first_frame_does_not_initialize(self) -> None:
        backend = _FakeOnlineBackend(window_len=1, step=1)
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            overlay_max_points_per_camera=4,
        )
        object_mask = np.zeros((8, 8), dtype=bool)
        object_mask[:4, :] = True
        controller_mask = np.zeros((8, 8), dtype=bool)

        self.assertIsNone(
            worker.process_group(
                self._packet_with_component_masks(
                    0,
                    object_mask=object_mask,
                    controller_mask=controller_mask,
                )
            )
        )

        self.assertEqual(backend.update_calls, 0)
        self.assertTrue(worker.snapshot()["cotracker_waiting_for_object_controller_by_camera"][0])

    def test_controller_only_first_frame_does_not_initialize(self) -> None:
        backend = _FakeOnlineBackend(window_len=1, step=1)
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            overlay_max_points_per_camera=4,
        )
        object_mask = np.zeros((8, 8), dtype=bool)
        controller_mask = np.zeros((8, 8), dtype=bool)
        controller_mask[4:, :] = True

        self.assertIsNone(
            worker.process_group(
                self._packet_with_component_masks(
                    0,
                    object_mask=object_mask,
                    controller_mask=controller_mask,
                )
            )
        )

        self.assertEqual(backend.update_calls, 0)
        self.assertTrue(worker.snapshot()["cotracker_waiting_for_object_controller_by_camera"][0])

    def test_object_controller_union_initializes_dense_queries(self) -> None:
        backend = _FakeOnlineBackend(window_len=1, step=1)
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            overlay_max_points_per_camera=30,
        )
        object_mask = np.zeros((100, 100), dtype=bool)
        object_mask[:60, :] = True
        controller_mask = np.zeros((100, 100), dtype=bool)
        controller_mask[40:, :] = True

        overlay = worker.process_group(
            self._packet_with_component_masks(
                0,
                object_mask=object_mask,
                controller_mask=controller_mask,
            )
        )

        self.assertIsNotNone(overlay)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot["query_mode"], "phystwin_dense")
        self.assertEqual(snapshot["query_count_request"], "auto")
        self.assertEqual(snapshot["tracking_query_count_actual_by_camera"][0], 5000)
        self.assertEqual(snapshot["tracking_union_pixels_by_camera"][0], 10000)
        self.assertEqual(snapshot["overlay_display_count_by_camera"][0], 30)
        self.assertEqual(overlay.camera_tracks_yx[0].shape, (30, 2))  # type: ignore[union-attr]

    def test_sampling_calls_phystwin_dense_with_seed_and_camera_idx(self) -> None:
        backend = _FakeOnlineBackend(window_len=1, step=1)
        object_mask = np.ones((8, 8), dtype=bool)
        controller_mask = np.ones((8, 8), dtype=bool)
        sampled = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            overlay_max_points_per_camera=3,
            seed=42,
            sampling_device="cpu",
        )

        with mock.patch("qqtt.demo.cotracker3_overlay_worker.sample_phystwin_dense", return_value=sampled) as sampler:
            worker.process_group(
                self._packet_with_component_masks(
                    0,
                    object_mask=object_mask,
                    controller_mask=controller_mask,
                )
            )

        sampler.assert_called_once()
        _args, kwargs = sampler.call_args
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["camera_idx"], 0)
        self.assertEqual(kwargs["torch_device"], "cpu")

    def test_overlay_slot_marks_stale_packets(self) -> None:
        slot = LatestTrackingOverlaySlot()
        slot.publish(
            TrackingOverlayPacket(
                group_id=1,
                frame_idx=1,
                timestamp_s=10.0,
                camera_tracks_yx={},
                camera_visibility={},
            )
        )

        self.assertFalse(slot.get_optional(now_s=10.1, stale_timeout_s=0.5).stale)  # type: ignore[union-attr]
        self.assertTrue(slot.get_optional(now_s=11.0, stale_timeout_s=0.5).stale)  # type: ignore[union-attr]
        self.assertIsNone(slot.get_fresh(now_s=11.0, stale_timeout_s=0.5))

    def test_overlay_thread_processes_latest_input_without_blocking_producer(self) -> None:
        backend = _FakeOnlineBackend(window_len=2, step=1)
        input_slot = LatestTrackingInputSlot()
        worker = CoTracker3OverlayWorker(
            camera_ids=(0,),
            backend_factory=lambda _camera_idx: backend,
            query_count=4,
            overlay_max_points_per_camera=4,
        )
        thread = CoTracker3OverlayThread(worker=worker, input_slot=input_slot, poll_interval_s=0.0001)
        thread.start()
        try:
            input_slot.publish(self._packet(0))
            for _ in range(200):
                if thread.processed_packets >= 1:
                    break
                time.sleep(0.001)
            input_slot.publish(self._packet(1))
            for _ in range(200):
                if worker.latest_overlay() is not None:
                    break
                time.sleep(0.001)
            self.assertIsNotNone(worker.latest_overlay())
        finally:
            thread.stop(timeout_s=1.0)
        self.assertGreaterEqual(thread.snapshot()["processed_packets"], 2)


if __name__ == "__main__":
    unittest.main()
