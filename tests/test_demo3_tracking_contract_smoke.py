from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Demo3TrackingContractSmokeTest(unittest.TestCase):
    def test_contract_doc_exists(self) -> None:
        path = ROOT / "docs" / "demo3_tracking_backend_overlay_contract.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn("multi-backend tracking benchmark", text)
        self.assertIn("coordinate order y,x", text)

    def test_demo22_dry_run_contains_disabled_tracking_overlay_contract(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py",
                "--dry-run",
                "--show-tracking-overlay",
                "--tracking-backend",
                "cotracker3_online",
                "--tracking-source",
                "offline_npz",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        contract = json.loads(result.stdout)
        self.assertTrue(contract["tracking_overlay"]["enabled"])
        self.assertEqual(contract["tracking_overlay"]["backend"], "cotracker3_online")
        self.assertEqual(contract["tracking_overlay"]["max_points"], 30)
        self.assertFalse(contract["tracking_overlay"]["blocking_render"])


if __name__ == "__main__":
    unittest.main()
