#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _preconfigure_main_cuda(argv: Sequence[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mask-gpu", default="0")
    known, _unknown = pre.parse_known_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(known.mask_gpu)


def build_arg_parser():
    _preconfigure_main_cuda(None)
    from qqtt.demo import demo31_runtime as runtime  # noqa: E402

    return runtime.build_arg_parser()


def main(argv: Sequence[str] | None = None) -> int:
    _preconfigure_main_cuda(argv)
    from qqtt.demo import demo31_runtime as runtime  # noqa: E402

    return runtime.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

