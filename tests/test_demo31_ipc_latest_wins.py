from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.demo31_dual_gpu_ipc import (
    LatestMaskCache,
    LatestWinsQueue,
    TrackingInputLitePacket,
    should_publish_tracking_input,
)


class Demo31IpcLatestWinsTest(unittest.TestCase):
    def test_latest_wins_queue_replaces_old_input_without_blocking(self) -> None:
        endpoint = LatestWinsQueue()
        endpoint.publish_latest("old")
        replaced = endpoint.publish_latest("new")

        self.assertEqual(replaced, 1)
        self.assertEqual(endpoint.take_latest(), "new")
        self.assertIsNone(endpoint.take_latest())
        self.assertEqual(endpoint.snapshot()["published"], 2)
        self.assertEqual(endpoint.snapshot()["replaced"], 1)

    def test_tracking_lite_packet_contains_only_rgb_and_mask(self) -> None:
        packet = TrackingInputLitePacket(
            group_id=1,
            frame_idx=2,
            timestamp_s=3.0,
            rgb_by_camera={0: np.zeros((4, 4, 3), dtype=np.uint8)},
            mask_by_camera={0: np.ones((4, 4), dtype=bool)},
        )

        self.assertEqual(packet.seq, 1)
        self.assertFalse(hasattr(packet, "depth_by_camera"))
        self.assertFalse(hasattr(packet, "intrinsics_by_camera"))
        self.assertFalse(hasattr(packet, "c2w_by_camera"))

        overlay_input = packet.to_overlay_input_packet()
        self.assertIsNone(overlay_input.depth_by_camera)
        self.assertIsNone(overlay_input.intrinsics_by_camera)
        self.assertIsNone(overlay_input.c2w_by_camera)

    def test_tracking_input_fps_gate(self) -> None:
        self.assertTrue(should_publish_tracking_input(now_s=10.0, last_publish_s=None, target_fps=10.0))
        self.assertFalse(should_publish_tracking_input(now_s=10.05, last_publish_s=10.0, target_fps=10.0))
        self.assertTrue(should_publish_tracking_input(now_s=10.11, last_publish_s=10.0, target_fps=10.0))
        self.assertFalse(should_publish_tracking_input(now_s=10.11, last_publish_s=10.0, target_fps=0.0))

    def test_strict_mask_policy_requires_matching_group(self) -> None:
        cache = LatestMaskCache()
        cache.publish(group_id=5, timestamp_s=100.0, mask_by_camera={0: "mask5"})

        self.assertIsNone(cache.select(group_id=6, now_s=100.1, policy="strict", stale_timeout_ms=250.0))
        selected = cache.select(group_id=5, now_s=100.1, policy="strict", stale_timeout_ms=250.0)

        self.assertIsNotNone(selected)
        self.assertFalse(selected.reused)  # type: ignore[union-attr]
        self.assertEqual(selected.source_group_id, 5)  # type: ignore[union-attr]

    def test_latest_reuse_mask_policy_records_age_and_reuse(self) -> None:
        cache = LatestMaskCache()
        cache.publish(group_id=5, timestamp_s=100.0, mask_by_camera={0: "mask5"})

        selected = cache.select(group_id=6, now_s=100.2, policy="latest-reuse", stale_timeout_ms=250.0)

        self.assertIsNotNone(selected)
        self.assertTrue(selected.reused)  # type: ignore[union-attr]
        self.assertAlmostEqual(selected.age_ms, 200.0, places=3)  # type: ignore[union-attr]
        snapshot = cache.snapshot()
        self.assertEqual(snapshot["reuse_count"], 1)
        self.assertEqual(snapshot["selection_count"], 1)
        self.assertGreater(snapshot["mask_reuse_ratio"], 0.0)

    def test_latest_reuse_rejects_stale_mask(self) -> None:
        cache = LatestMaskCache()
        cache.publish(group_id=5, timestamp_s=100.0, mask_by_camera={0: "mask5"})

        self.assertIsNone(cache.select(group_id=6, now_s=100.5, policy="latest-reuse", stale_timeout_ms=250.0))
        self.assertEqual(cache.snapshot()["stale_reject_count"], 1)


if __name__ == "__main__":
    unittest.main()

