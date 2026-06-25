from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FFS_ENV_NAME = "FFS-SAM-RS"
DEFAULT_FFS_ENV_PYTHON = Path("python")

DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_FFS_REPO = DEFAULT_RUNTIME_ASSET_ROOT / "Fast-FoundationStereo"
DEFAULT_FFS_MODEL_NAME = "20-30-48"
DEFAULT_FFS_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / DEFAULT_FFS_MODEL_NAME / "model_best_bp2_serialize.pth"
DEFAULT_FFS_SCALE = 1.0
DEFAULT_FFS_VALID_ITERS = 4
DEFAULT_FFS_MAX_DISP = 192

DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL = 5
DEFAULT_FFS_TRT_INPUT_SIZE = (480, 848)
DEFAULT_FFS_TRT_ENGINE_SIZE = (480, 864)
DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5"
    / "engines"
    / "model_20-30-48_iters_4_res_480x864"
)
