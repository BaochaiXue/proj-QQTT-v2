from __future__ import annotations

import unittest

from scripts.harness.object_case_registry import (
    OBJECT_RAW_CASES,
    STATIC_OBJECT_RAW_CASES,
    STILL_OBJECT_RAW_CASES,
    get_raw_object_capture_spec,
)


class HarnessObjectCaseRegistryTest(unittest.TestCase):
    def test_still_object_round1_uses_distinct_case_name(self) -> None:
        spec = get_raw_object_capture_spec(object_set="still_object", round_id="round1")

        self.assertEqual(spec.raw_case_name, "both_30_still_object_round1_20260428")
        self.assertEqual(spec.capture_mode, "both_eval")
        self.assertEqual(spec.streams, ("color", "depth", "ir_left", "ir_right"))

    def test_still_object_round2_uses_distinct_case_name(self) -> None:
        spec = get_raw_object_capture_spec(object_set="still_object", round_id="round2")

        self.assertEqual(spec.raw_case_name, "both_30_still_object_round2_20260428")
        self.assertEqual(spec.capture_mode, "both_eval")
        self.assertEqual(spec.streams, ("color", "depth", "ir_left", "ir_right"))

    def test_still_object_round3_uses_distinct_case_name(self) -> None:
        spec = get_raw_object_capture_spec(object_set="still_object", round_id="round3")

        self.assertEqual(spec.raw_case_name, "both_30_still_object_round3_20260428")
        self.assertEqual(spec.capture_mode, "both_eval")
        self.assertEqual(spec.streams, ("color", "depth", "ir_left", "ir_right"))

    def test_still_object_round4_uses_distinct_case_name(self) -> None:
        spec = get_raw_object_capture_spec(object_set="still_object", round_id="round4")

        self.assertEqual(spec.raw_case_name, "both_30_still_object_round4_20260428")
        self.assertEqual(spec.capture_mode, "both_eval")
        self.assertEqual(spec.streams, ("color", "depth", "ir_left", "ir_right"))

    def test_still_object_round7_uses_distinct_case_name(self) -> None:
        spec = get_raw_object_capture_spec(object_set="still_object", round_id="round7")

        self.assertEqual(spec.raw_case_name, "both_30_still_object_round7_20260428")
        self.assertEqual(spec.capture_mode, "both_eval")
        self.assertEqual(spec.streams, ("color", "depth", "ir_left", "ir_right"))

    def test_still_object_round8_uses_distinct_case_name(self) -> None:
        spec = get_raw_object_capture_spec(object_set="still_object", round_id="round8")

        self.assertEqual(spec.raw_case_name, "both_30_still_object_round8_20260428")
        self.assertEqual(spec.capture_mode, "both_eval")
        self.assertEqual(spec.streams, ("color", "depth", "ir_left", "ir_right"))

    def test_static_object_cases_do_not_include_still_object_round(self) -> None:
        static_names = {spec.raw_case_name for spec in STATIC_OBJECT_RAW_CASES}

        self.assertNotIn("both_30_still_object_round1_20260428", static_names)
        self.assertNotIn("both_30_still_object_round2_20260428", static_names)
        self.assertNotIn("both_30_still_object_round3_20260428", static_names)
        self.assertNotIn("both_30_still_object_round4_20260428", static_names)
        self.assertNotIn("both_30_still_object_round7_20260428", static_names)
        self.assertNotIn("both_30_still_object_round8_20260428", static_names)
        self.assertTrue(all("still_object" not in name for name in static_names))

    def test_object_set_round_pairs_are_unique(self) -> None:
        keys = [(spec.object_set, spec.round_id) for spec in OBJECT_RAW_CASES]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual([spec.round_id for spec in STILL_OBJECT_RAW_CASES], ["round1", "round2", "round3", "round4", "round7", "round8"])
