from __future__ import annotations

import unittest

from scripts.harness.probe_d455_stream_capability import (
    STREAM_SET_DEFS,
    build_followup_probe_cases,
    build_initial_probe_cases,
    case_key,
    stable_order_serials,
)


class D455ProbeMatrixBuilderTest(unittest.TestCase):
    def test_initial_cases_cover_single_and_three_camera_topologies(self) -> None:
        cases = build_initial_probe_cases(
            ["239222303506", "239222300433", "239222300781"],
            warmup_s=2.0,
            duration_s=10.0,
        )
        single_cases = [case for case in cases if case["topology_type"] == "single"]
        triple_cases = [case for case in cases if case["topology_type"] == "three_camera"]

        self.assertEqual(len(single_cases), 3 * len(STREAM_SET_DEFS))
        self.assertEqual(len(triple_cases), len(STREAM_SET_DEFS))
        self.assertEqual(triple_cases[0]["serials"], ["239222300433", "239222300781", "239222303506"])

    def test_stable_serial_order_is_preserved(self) -> None:
        ordered = stable_order_serials(["239222303506", "239222300433", "239222300781"])
        self.assertEqual(ordered, ["239222300433", "239222300781", "239222303506"])

    def test_followup_logic_adds_emitter_off_and_640_fallback(self) -> None:
        results = [
            {
                "topology_type": "single",
                "serials": ["239222300433"],
                "stream_set": "ir_pair",
                "width": 848,
                "height": 480,
                "fps": 30,
                "emitter_request": "on",
                "warmup_s": 2.0,
                "duration_s": 10.0,
                "success": True,
            },
            {
                "topology_type": "three_camera",
                "serials": ["239222300433", "239222300781", "239222303506"],
                "stream_set": "rgb_ir_pair",
                "width": 848,
                "height": 480,
                "fps": 30,
                "emitter_request": "on",
                "warmup_s": 2.0,
                "duration_s": 10.0,
                "success": False,
            },
        ]

        followups = build_followup_probe_cases(results)
        followup_keys = {case_key(case) for case in followups}

        self.assertIn(
            (
                "single",
                ("239222300433",),
                "ir_pair",
                848,
                480,
                30,
                "off",
            ),
            followup_keys,
        )
        self.assertIn(
            (
                "three_camera",
                ("239222300433", "239222300781", "239222303506"),
                "rgb_ir_pair",
                640,
                480,
                30,
                "on",
            ),
            followup_keys,
        )


if __name__ == "__main__":
    unittest.main()
