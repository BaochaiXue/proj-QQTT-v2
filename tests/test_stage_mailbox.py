from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from qqtt.demo.services.stage_mailbox import LatestOnlyStageMailbox


@dataclass(frozen=True)
class _Bundle:
    group_id: int
    value: str


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

    def test_complete_active_accepts_replaced_bundle_with_same_group(self) -> None:
        mailbox: LatestOnlyStageMailbox[_Bundle] = LatestOnlyStageMailbox()
        mailbox.publish_latest(_Bundle(group_id=7, value="input"))
        active = mailbox.take_next()
        self.assertIsNotNone(active)
        out = replace(active, value="output")  # type: ignore[arg-type]

        mailbox.publish_latest(_Bundle(group_id=8, value="next"))
        completed = mailbox.complete_active(out)

        self.assertEqual(completed, active)
        self.assertIsNone(mailbox.active())
        self.assertEqual(mailbox.take_next(), _Bundle(group_id=8, value="next"))

    def test_complete_active_can_match_group_id(self) -> None:
        mailbox: LatestOnlyStageMailbox[_Bundle] = LatestOnlyStageMailbox()
        mailbox.publish_latest(_Bundle(group_id=11, value="input"))
        active = mailbox.take_next()
        self.assertIsNotNone(active)

        self.assertIsNone(mailbox.complete_active(group_id=12))
        self.assertEqual(mailbox.active(), active)
        self.assertEqual(mailbox.complete_active(group_id=11), active)
        self.assertIsNone(mailbox.active())


if __name__ == "__main__":
    unittest.main()
