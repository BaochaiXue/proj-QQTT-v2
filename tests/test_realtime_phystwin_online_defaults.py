from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _arg_default(script_relpath: str, option: str) -> int:
    tree = ast.parse((REPO_ROOT / script_relpath).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return int(keyword.value.value)
    raise AssertionError(f"{option} default not found in {script_relpath}")


class RealtimePhysTwinOnlineDefaultsTest(unittest.TestCase):
    def test_online_entrypoints_default_to_35_frame_segments(self) -> None:
        for script_relpath in (
            "realtime_phystwin/optimize_online_cma.py",
            "realtime_phystwin/train_online_warp.py",
            "realtime_phystwin/train_online_zero_then_first.py",
        ):
            with self.subTest(script=script_relpath):
                self.assertEqual(_arg_default(script_relpath, "--segment_len"), 35)


if __name__ == "__main__":
    unittest.main()
