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
        self.assertIn("scripts/harness/visual_compare_depth_triplet_ply.py", content)
        self.assertIn("scripts/harness/visual_compare_depth_triplet_video.py", content)
        self.assertIn("scripts/harness/visual_compare_rerun.py", content)

    def test_agents_mentions_sanctioned_demo_proxy_scope(self) -> None:
        content = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("sanctioned single-camera", content)
        self.assertIn("realtime demo/proxy diagnostics", content)
        self.assertIn("scripts/harness/realtime_single_camera_pointcloud.py", content)
        self.assertIn("qqtt/demo/realtime_single_camera_pointcloud.py", content)
        self.assertIn("services/ffs_remote/", content)
        self.assertIn("formal recording/alignment code must not depend on those layers", content)

    def test_agents_mentions_single_camera_branch_policy(self) -> None:
        content = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("Single-Camera Branch Policy", content)
        self.assertIn("single-camera-specific modifications must be made, committed, and pushed on the `single-camera` branch", content)
        self.assertIn("Do not commit or push single-camera changes directly to `main`", content)
        self.assertIn("git push origin single-camera", content)
