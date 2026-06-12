from __future__ import annotations

import unittest

from qqtt.demo import demo22_runtime as demo22


class Demo3TrackingContractSmokeTest(unittest.TestCase):
    def test_demo22_runtime_contains_disabled_tracking_overlay_contract(self) -> None:
        argv = [
            "--dry-run",
            "--show-tracking-overlay",
            "--tracking-backend",
            "cotracker3_online",
            "--tracking-source",
            "offline_npz",
        ]
        parser = demo22.build_arg_parser()
        args = parser.parse_args(argv)
        args = demo22.apply_preset_defaults(
            args,
            explicit_options=demo22.explicit_cli_options(argv),
        )
        contract = demo22.build_contract(args)
        self.assertTrue(contract["tracking_overlay"]["enabled"])
        self.assertEqual(contract["tracking_overlay"]["backend"], "cotracker3_online")
        self.assertEqual(contract["tracking_overlay"]["max_points"], 30)
        self.assertFalse(contract["tracking_overlay"]["blocking_render"])


if __name__ == "__main__":
    unittest.main()
