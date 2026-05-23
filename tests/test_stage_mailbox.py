from __future__ import annotations

import unittest

from qqtt.demo.services.stage_mailbox import LatestOnlyStageMailbox


class StageMailboxTests(unittest.TestCase):
    def test_latest_pending_replaces_without_touching_active(self) -> None:
        mailbox: LatestOnlyStageMailbox[int] = LatestOnlyStageMailbox()

        mailbox.publish_latest(1)
        active = mailbox.take_next()
        self.assertEqual(active, 1)
        self.assertEqual(mailbox.active(), 1)

        self.assertEqual(mailbox.publish_latest(2), 0)
        self.assertEqual(mailbox.publish_latest(3), 1)

        self.assertEqual(mailbox.active(), 1)
        self.assertEqual(mailbox.pending(), 3)
        self.assertIsNone(mailbox.take_next())
        self.assertEqual(mailbox.complete_active(), 1)
        self.assertEqual(mailbox.take_next(), 3)

    def test_snapshot_counts_drop_pending_not_active(self) -> None:
        mailbox: LatestOnlyStageMailbox[str] = LatestOnlyStageMailbox()
        mailbox.publish_latest("old")
        mailbox.publish_latest("new")

        snapshot = mailbox.snapshot()

        self.assertEqual(snapshot["published"], 2)
        self.assertEqual(snapshot["pending_replaced"], 1)
        self.assertFalse(snapshot["active_present"])
        self.assertTrue(snapshot["pending_present"])


if __name__ == "__main__":
    unittest.main()
