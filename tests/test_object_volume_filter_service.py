from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from qqtt.demo.services.object_volume_filter_service import (
    ObjectVolumeFilterConfig,
    ObjectVolumeFilterInput,
    ObjectVolumeFilterOutput,
    ObjectVolumeFilterService,
    ObjectVolumeFilterWorker,
)


class ObjectVolumeFilterServiceTests(unittest.TestCase):
    def test_filter_sync_keeps_one_point_per_occupied_voxel(self) -> None:
        service = ObjectVolumeFilterService(
            ObjectVolumeFilterConfig(base_voxel_m=0.005, adaptive=False, origin_policy="world")
        )
        points = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.001, 0.001, 0.001],
                [0.006, 0.000, 0.000],
            ],
            dtype=np.float32,
        )
        colors = np.arange(9, dtype=np.uint8).reshape(3, 3)

        output = service.filter_sync(
            ObjectVolumeFilterInput(seq=7, timestamp_s=1.0, object_xyz_world=points, object_rgb=colors)
        )

        self.assertEqual(output.seq, 7)
        self.assertEqual(output.occupied_voxels, 2)
        self.assertEqual(output.output_points, 2)
        self.assertEqual(output.stats["object_volume_output_points"], 2)
        self.assertEqual(output.stats["object_volume_occupied_voxels"], 2)
        self.assertTrue(output.stats["object_volume_exact"])
        self.assertEqual(output.stats["object_volume_sampler_impl"], "numpy-unique")
        self.assertIn("object_volume_key_ms", output.stats)

    def test_larger_voxel_reduces_or_preserves_output_count(self) -> None:
        points = np.array([[float(i) * 0.001, 0.0, 0.0] for i in range(20)], dtype=np.float32)
        colors = np.zeros((20, 3), dtype=np.uint8)
        small = ObjectVolumeFilterService(ObjectVolumeFilterConfig(base_voxel_m=0.002, adaptive=False))
        large = ObjectVolumeFilterService(ObjectVolumeFilterConfig(base_voxel_m=0.010, adaptive=False))

        small_output = small.filter_sync(
            ObjectVolumeFilterInput(seq=1, timestamp_s=1.0, object_xyz_world=points, object_rgb=colors)
        )
        large_output = large.filter_sync(
            ObjectVolumeFilterInput(seq=1, timestamp_s=1.0, object_xyz_world=points, object_rgb=colors)
        )

        self.assertGreaterEqual(small_output.output_points, large_output.output_points)

    def test_emergency_cap_applies_after_voxel_sampling(self) -> None:
        service = ObjectVolumeFilterService(
            ObjectVolumeFilterConfig(base_voxel_m=0.005, adaptive=False, emergency_max_points=4)
        )
        points = np.array([[float(i) * 0.01, 0.0, 0.0] for i in range(10)], dtype=np.float32)
        colors = np.zeros((10, 3), dtype=np.uint8)

        output = service.filter_sync(
            ObjectVolumeFilterInput(seq=1, timestamp_s=1.0, object_xyz_world=points, object_rgb=colors)
        )

        self.assertEqual(output.occupied_voxels, 10)
        self.assertEqual(output.output_points, 4)
        self.assertTrue(output.safety_cap_triggered)
        self.assertFalse(output.stats["object_volume_exact"])

    def test_snapshot_reports_latest_output(self) -> None:
        service = ObjectVolumeFilterService(ObjectVolumeFilterConfig(adaptive=False))
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.zeros((1, 3), dtype=np.uint8)

        service.submit_latest(
            ObjectVolumeFilterInput(seq=5, timestamp_s=1.0, object_xyz_world=points, object_rgb=colors)
        )

        snapshot = service.snapshot()
        self.assertEqual(snapshot["processed_count"], 1)
        self.assertEqual(snapshot["published_count"], 1)
        self.assertEqual(snapshot["latest_seq"], 5)

    def test_worker_drops_stale_pending_input_and_publishes_latest(self) -> None:
        class BlockingService:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()
                self.processed: list[int] = []

            def filter_sync(self, packet: ObjectVolumeFilterInput) -> ObjectVolumeFilterOutput:
                if not self.processed:
                    self.started.set()
                    self.release.wait(timeout=2.0)
                self.processed.append(int(packet.seq))
                return ObjectVolumeFilterOutput(
                    seq=int(packet.seq),
                    timestamp_s=float(packet.timestamp_s),
                    object_xyz_world=np.empty((0, 3), dtype=np.float32),
                    object_rgb=np.empty((0, 3), dtype=np.uint8),
                    controller_xyz_world=None,
                    controller_rgb=None,
                    voxel_size_m=0.005,
                    occupied_voxels=0,
                    input_points=0,
                    output_points=0,
                    filter_ms=0.0,
                    safety_cap_triggered=False,
                    stats={"object_volume_exact": True},
                )

        service = BlockingService()
        worker = ObjectVolumeFilterWorker(service=service)  # type: ignore[arg-type]
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        worker.start()
        try:
            worker.submit_latest(
                ObjectVolumeFilterInput(seq=1, timestamp_s=time.time(), object_xyz_world=empty_points, object_rgb=empty_colors)
            )
            self.assertTrue(service.started.wait(timeout=2.0))
            worker.submit_latest(
                ObjectVolumeFilterInput(seq=2, timestamp_s=time.time(), object_xyz_world=empty_points, object_rgb=empty_colors)
            )
            worker.submit_latest(
                ObjectVolumeFilterInput(seq=3, timestamp_s=time.time(), object_xyz_world=empty_points, object_rgb=empty_colors)
            )
            service.release.set()
            deadline_s = time.time() + 2.0
            while time.time() < deadline_s:
                latest = worker.get_latest()
                if latest is not None and latest.seq == 3:
                    break
                time.sleep(0.01)
            latest = worker.get_latest()
            self.assertIsNotNone(latest)
            self.assertEqual(latest.seq, 3)
            self.assertEqual(service.processed, [1, 3])
            self.assertGreaterEqual(worker.snapshot()["object_volume_input_queue_replaced_count"], 1)
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
