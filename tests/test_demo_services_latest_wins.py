from __future__ import annotations

import unittest

from qqtt.demo.services.latest_wins import LatestValueSlot, LatestWinsQueue


class DemoServicesLatestWinsTests(unittest.TestCase):
    def test_latest_value_slot_replaces_pending_item(self) -> None:
        slot: LatestValueSlot[int] = LatestValueSlot()

        self.assertEqual(slot.publish_latest(1), 0)
        self.assertEqual(slot.publish_latest(2), 1)

        self.assertEqual(slot.take_latest(), 2)
        self.assertIsNone(slot.take_latest())
        self.assertEqual(slot.snapshot()["replaced"], 1)

    def test_latest_wins_queue_drops_stale_items(self) -> None:
        endpoint = LatestWinsQueue()

        endpoint.publish_latest("old")
        endpoint.publish_latest("new")

        self.assertEqual(endpoint.take_latest(), "new")
        self.assertEqual(endpoint.snapshot()["replaced"], 1)


if __name__ == "__main__":
    unittest.main()
