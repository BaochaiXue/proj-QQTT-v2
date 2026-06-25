from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


SCRIPT = Path("scripts/run_single_proj_qqtt_only_fake_live.sh")


class QqttOnlyScriptTest(unittest.TestCase):
    def test_script_is_bash_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_script_uses_demo_v4_and_not_realtime_phystwin(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("demo_v4/realtime_futurephystwin_chunks.py", text)
        self.assertNotIn("demo_v5/realtime_futurephystwin_chunks.py", text)
        self.assertNotIn("train_online_zero_then_first.py", text)
        self.assertNotIn("optimize_online_cma.py", text)


if __name__ == "__main__":
    unittest.main()
