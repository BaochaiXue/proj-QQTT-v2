from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AgentsScopeContractSmokeTest(unittest.TestCase):
    def test_agents_mentions_compare_scope_and_current_entrypoints(self) -> None:
        content = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("native-vs-FFS comparison visualization", content)
        self.assertIn("aligned native-vs-FFS comparison visualization remains an in-scope diagnostic utility", content)
        self.assertIn("scripts/harness/visual_compare_turntable.py", content)
        self.assertIn("scripts/harness/visual_compare_depth_panels.py", content)
        self.assertIn("scripts/harness/visual_compare_reprojection.py", content)
