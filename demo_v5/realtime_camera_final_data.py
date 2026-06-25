#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from qqtt.demo import realtime_masked_edgetam_pcd as masked_pcd  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return masked_pcd.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
