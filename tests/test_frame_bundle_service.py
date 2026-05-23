from __future__ import annotations

from types import SimpleNamespace
import unittest

from qqtt.demo.services.frame_bundle_service import BundleStore


class FrameBundleServiceTests(unittest.TestCase):
    def test_protected_bundle_cannot_be_evicted(self) -> None:
        store = BundleStore(max_groups=2)
        store.upsert(1)
        store.protect(1)
        store.upsert(2)
        store.upsert(3)

        self.assertIn(1, store.group_ids())
        self.assertEqual(store.snapshot()["bundle_store_evicted_count"], 1)
        self.assertEqual(store.snapshot()["bundle_store_protected_groups"], 1)

    def test_tracker_result_uses_exact_precomputed_packet(self) -> None:
        store = BundleStore(max_groups=4)
        packet = SimpleNamespace(group_id=7)
        store.attach_precomputed_render_packet(packet)

        match = store.take_for_tracker_result(7)

        self.assertEqual(match.match_mode, "exact")
        self.assertIs(match.bundle.precomputed_render_packet, packet)  # type: ignore[union-attr]
        self.assertFalse(match.used_nearest)

    def test_tracker_result_never_matches_other_bundle_by_default(self) -> None:
        store = BundleStore(max_groups=4)
        store.attach_precomputed_render_packet(SimpleNamespace(group_id=8))

        match = store.take_for_tracker_result(7)

        self.assertIsNone(match.bundle)
        self.assertEqual(match.match_mode, "missing-exact")

    def test_debug_policy_can_match_nearest_bundle(self) -> None:
        store = BundleStore(max_groups=4)
        store.attach_precomputed_render_packet(SimpleNamespace(group_id=8))

        match = store.take_for_tracker_result(7, allow_nearest=True)

        self.assertEqual(match.match_mode, "nearest")
        self.assertEqual(match.bundle.group_id, 8)  # type: ignore[union-attr]
        self.assertTrue(match.used_nearest)

    def test_drop_through_keeps_protected_bundle(self) -> None:
        store = BundleStore(max_groups=4)
        store.attach_precomputed_render_packet(SimpleNamespace(group_id=1))
        store.attach_precomputed_render_packet(SimpleNamespace(group_id=2))
        store.protect(1)

        store.drop_through(2)

        self.assertEqual(store.group_ids(), {1})


if __name__ == "__main__":
    unittest.main()
