#!/usr/bin/env python3
"""Demo v6.2 path bootstrap and module constants."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _resolve_repo_root() -> Path:
    """Resolve repo root."""
    candidates: list[Path] = []
    # Vendored under demo_v7/runtime/: repo root is three levels up.
    candidates.extend([Path(__file__).resolve().parents[3], Path.cwd()])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (
            (root / "data_process").is_dir()
            and (root / "demo_v7").is_dir()
            and (root / "qqtt").is_dir()
        ):
            return root
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _resolve_repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


from qqtt.env.camera.table_calibration import TABLE_WORLD_FRAME_KIND  # noqa: E402
from qqtt.tracking.backends.point_tracker_adapter import TRACKER_BACKEND_TAPNEXTPP  # noqa: E402
from qqtt.tracking.sampling import PHYSTWIN_DENSE_QUERY_POINTS  # noqa: E402


# ---------------------------------------------------------------------------
# Module constants: modes, defaults, geometry layer names, object/track ids
# ---------------------------------------------------------------------------
DEFAULT_EDGETAM_MODEL_ID = str(Path("vendor") / "demo_runtime" / "EdgeTAM-hf")
DEFAULT_PROFILE = "848x480"
DEFAULT_EDGETAM_COMPILE_MODE = "vision-reduce-overhead"
# Pre-pay the deferred torch.compile + CUDA-graph capture with a throwaway
# prompted forward during warm-up (byte-A/B gated: frame-0 masks must be
# identical to the lazy-compile path before this may be True).
DEFAULT_EDGETAM_PRECOMPILE_FIRST_FORWARD = True
TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_ONLY = "controller-only"
TRACK_MODE_NONE = "none"
TRACK_MODES = (
    TRACK_MODE_CONTROLLER_OBJECT,
    TRACK_MODE_OBJECT_ONLY,
    TRACK_MODE_CONTROLLER_ONLY,
    TRACK_MODE_NONE,
)
DEFAULT_TRACK_MODE = "controller-object"
DEPTH_SOURCES = ("ffs", "realsense", "none")
# Orchestrator config vocabulary (camera.depth_backend) -> --depth-source.
DEPTH_BACKEND_TO_DEPTH_SOURCE = {
    "ir-ffs": "ffs",
    "native-realsense": "realsense",
}
INPUT_SOURCE_LIVE = "live"
INPUT_SOURCE_FAKE_LIVE = "fake-live"
INPUT_SOURCES = (INPUT_SOURCE_LIVE, INPUT_SOURCE_FAKE_LIVE)
PCD_MODES = ("masked", "none")
DEFAULT_PCD_MODE = "masked"
TRACKER_QUERY_SOURCE_UNION_MASK = "object_controller_union_mask"
FAKE_LIVE_FRAME_SELECTION_POLICY = "drop_source_frames_preserve_recording_time"
DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES = 64
DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES = 8
HAND_A_ID = 1
OBJECT_ID = 2
HAND_B_ID = 3
EDGE_TAM_OBJECT_LABELS = {
    HAND_A_ID: "hand_a",
    OBJECT_ID: "object",
    HAND_B_ID: "hand_b",
}
CONTROLLER_COLOR_RGB = (255, 96, 32)
OBJECT_COLOR_RGB = (64, 180, 255)
CAMERA_COLOR_FRAME = "camera_color_frame"
DEFAULT_TRACKER_DISPLAY_SCOPE = "union"
DEFAULT_TRACKER_BACKEND = TRACKER_BACKEND_TAPNEXTPP
DEFAULT_TRACKER_QUERY_COUNT = PHYSTWIN_DENSE_QUERY_POINTS
DEFAULT_TRACKER_SEED = 42
QUERY_CONTROLLER_INSTANCE_HAND_A = 1
QUERY_CONTROLLER_INSTANCE_HAND_B = 2
HEADLESS_CAPTURE_SAVED_PCD_SOURCE = "origin_processed_mask_dense_world"


def table_world_enabled(table_c2w: np.ndarray | None) -> bool:
    """Return whether the run operates in the calibrated table-world frame."""
    return table_c2w is not None


def pcd_coordinate_frame(table_c2w: np.ndarray | None) -> str:
    """Return the coordinate frame PCD products are expressed in."""
    return (
        TABLE_WORLD_FRAME_KIND if table_world_enabled(table_c2w) else CAMERA_COLOR_FRAME
    )


DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_TAPNET_REPO_DIR = DEFAULT_RUNTIME_ASSET_ROOT / "tapnet"
DEFAULT_TAPNEXTPP_CHECKPOINT = (
    DEFAULT_RUNTIME_ASSET_ROOT / "checkpoints" / "tapnextpp" / "tapnextpp_ckpt.pt"
)
DEFAULT_FFS_REPO = DEFAULT_RUNTIME_ASSET_ROOT / "Fast-FoundationStereo"
DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5"
    / "engines"
    / "model_20-30-48_iters_4_res_480x864"
)


def _resolve_path(value: str | Path) -> Path:
    """Resolve a filesystem path to an absolute expanded path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
