#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_v2_1 import realtime_three_view_masked_fused_pcd as demo21  # noqa: E402


DEFAULT_PRESET = demo21.PRESET_DEMO22_STAGED_PARALLEL_5FPS


def _with_default_preset(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--preset" not in args and not any(str(arg).startswith("--preset=") for arg in args):
        return ["--preset", DEFAULT_PRESET, *args]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return demo21.main(_with_default_preset(argv))


if __name__ == "__main__":
    raise SystemExit(main())
