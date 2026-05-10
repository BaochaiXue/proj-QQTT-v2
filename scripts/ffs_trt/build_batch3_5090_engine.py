from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.depth_backends.ffs_defaults import (
    DEFAULT_FFS_MODEL_PATH,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR,
)
from scripts.ffs_trt import build_batch3_4090_engine as batch3_builder


DEFAULT_TIMING_CACHE = DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR.parents[1] / "timing_cache.bin"


def _configure_defaults() -> None:
    batch3_builder.DEFAULT_FFS_REPO = DEFAULT_FFS_REPO
    batch3_builder.DEFAULT_WEIGHT = DEFAULT_FFS_MODEL_PATH
    batch3_builder.DEFAULT_OUT_DIR = DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR
    batch3_builder.DEFAULT_TIMING_CACHE = DEFAULT_TIMING_CACHE
    batch3_builder.BATCH1_PATH_TOKEN = (
        "ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/"
        "engines/model_20-30-48_iters_4_res_480x864"
    )


def build_parser() -> argparse.ArgumentParser:
    _configure_defaults()
    parser = batch3_builder.build_parser()
    parser.description = "Build isolated RTX 5090 Laptop Fast-FoundationStereo TensorRT batch=3 engines."
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_defaults()
    return batch3_builder.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
