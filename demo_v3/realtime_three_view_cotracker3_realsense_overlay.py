#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqtt.demo import demo3_runtime as runtime  # noqa: E402


def build_arg_parser():
    return runtime.build_arg_parser()


def main(argv: Sequence[str] | None = None) -> int:
    return runtime.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
