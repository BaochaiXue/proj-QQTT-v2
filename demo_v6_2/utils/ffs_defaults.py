"""Default Fast-FoundationStereo model/TensorRT paths (vendored for demo_v6_1).

Vendored verbatim from ``data_process/depth_backends/ffs_defaults.py`` so
demo_v6_1 does not import the repo-level ``data_process`` package. These are
argparse defaults (overridable) pointing at the vendored FFS model assets under
the repo root; ``parents[2]`` still resolves to the repo root from
``demo_v6_1/utils/``.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_FFS_REPO = DEFAULT_RUNTIME_ASSET_ROOT / "Fast-FoundationStereo"

DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5"
    / "engines"
    / "model_20-30-48_iters_4_res_480x864"
)
