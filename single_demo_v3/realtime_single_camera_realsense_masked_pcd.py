#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from qqtt.demo import single_demo_v3_runtime as runtime  # noqa: E402


def build_arg_parser():
    return runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3)


def main(argv: Sequence[str] | None = None) -> int:
    return runtime.main(argv, demo_version=runtime.DEMO_VERSION_3)


if __name__ == "__main__":
    raise SystemExit(main())
