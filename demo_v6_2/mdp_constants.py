#!/usr/bin/env python3
"""Demo v6.1 shared imports, path bootstrap, and module constants."""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Generic, TypeVar

import numpy as np


def _resolve_repo_root() -> Path:
    """Resolve repo root."""
    candidates: list[Path] = []
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (
            (root / "data_process").is_dir()
            and (root / "demo_v6_2").is_dir()
            and (root / "qqtt").is_dir()
        ):
            return root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


def _repo_relative_path_text(path: str | Path | None) -> str | None:
    """Return the repo relative path text."""
    if path is None:
        return None
    original = Path(path)
    try:
        resolved = original.expanduser().resolve()
    except OSError:
        return str(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


from demo_v6_2.utils.camera import (  # noqa: E402
    CameraIntrinsics,
    SUPPORTED_CAPTURE_FPS,
    apply_emitter,
    camera_intrinsics_from_rs,
    load_realsense_module,
    parse_profile,
    resolve_serial,
    rs_extrinsics_to_matrix,
    rs_intrinsics_to_matrix,
    rs_translation_norm,
)
from demo_v6_2.utils.concurrency import (  # noqa: E402
    LatestSlot,
    elapsed_ms as _elapsed_ms,
    packet_seq as _packet_seq,
)
from demo_v6_2.utils.ffs_align import FfsIrToColorAligner, validate_ffs_paths  # noqa: E402
from demo_v6_2 import main_warmup  # noqa: E402
from demo_v6_2 import shape_prior_warmup  # noqa: E402
from demo_v6_2.tracking import CONTROLLER_FINAL_COUNT  # noqa: E402
from demo_v6_2.main_warmup import InitialMaskBundle  # noqa: E402
from demo_v6_2.utils.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
)
from qqtt.env.camera.table_calibration import (  # noqa: E402
    TABLE_WORLD_FRAME_KIND,
    TableCalibrationLoadError,
    load_table_calibration_transforms,
)
from demo_v6_2.utils.projection import lift_tracks_yx_to_world  # noqa: E402
from demo_v6_2.utils.query_rainbow import query_rainbow_colors_from_points_yx_rgb_u8  # noqa: E402
from demo_v6_2.phystwin_strict_product import (  # noqa: E402
    COMPATIBILITY_TARGET_PHYSTWIN,
    DEFAULT_TRACKING_PRODUCT_BACKEND,
    PHYSTWIN_STRICT_EXECUTION_MODE,
    TRACKING_PRODUCT_BACKENDS,
    finalize_headless_capture,
    normalize_tracking_product_backend,
    prepare_phystwin_frame,
    write_prepared_phystwin_frame,
)
from qqtt.tracking.backends.point_tracker_adapter import (  # noqa: E402
    TRACKER_BACKEND_NONE,
    TRACKER_BACKEND_TAPNEXTPP,
    TRACKER_BACKENDS,
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
    normalize_tracker_backend,
)
from qqtt.tracking.sampling import PHYSTWIN_DENSE_QUERY_POINTS, sample_phystwin_dense  # noqa: E402


# ---------------------------------------------------------------------------
# Module constants: modes, defaults, geometry layer names, object/track ids
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = str(Path("vendor") / "demo_runtime" / "EdgeTAM-hf")
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_COMPILE_MODE = "vision-reduce-overhead"
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
DEFAULT_DEPTH_SOURCE = "ffs"
INPUT_SOURCE_LIVE = "live"
INPUT_SOURCE_FAKE_LIVE = "fake-live"
INPUT_SOURCES = (INPUT_SOURCE_LIVE, INPUT_SOURCE_FAKE_LIVE)
DEFAULT_FAKE_LIVE_CASE = Path("data_collect/stuffed_animal_hand_both_eval_5fps_normal")
PCD_MODES = ("masked", "none")
DEFAULT_PCD_MODE = "masked"
DEMO_VISUAL_MODE_PCD = "pcd"
DEMO_VISUAL_MODE_TRACKING = "tracking"
DEMO_VISUAL_MODES = (DEMO_VISUAL_MODE_PCD, DEMO_VISUAL_MODE_TRACKING)
DEFAULT_DEMO_VISUAL_MODE = DEMO_VISUAL_MODE_TRACKING
TRACKER_QUERY_SOURCE_UNION_MASK = "object_controller_union_mask"
TRACKER_MARKER_GATE_TARGET_MASK_DEPTH = "target_mask_depth"
FAKE_LIVE_FRAME_SELECTION_POLICY = "drop_source_frames_preserve_recording_time"
DEFAULT_PCD_MASK_ERODE_PIXELS = 0
DEFAULT_OBJECT_PCD_MASK_ERODE_PIXELS: int | None = None
DEFAULT_CONTROLLER_PCD_MASK_ERODE_PIXELS: int | None = None
DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES = 64
DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD = 0.0
DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES = 8
HAND_A_ID = 1
OBJECT_ID = 2
HAND_B_ID = 3
CONTROLLER_ID = HAND_A_ID
EDGE_TAM_OBJECT_LABELS = {
    HAND_A_ID: "hand_a",
    OBJECT_ID: "object",
    HAND_B_ID: "hand_b",
}
CONTROLLER_COLOR_RGB = (255, 96, 32)
OBJECT_COLOR_RGB = (64, 180, 255)
GEOMETRY_CONTROLLER = "masked_edgetam_controller"
GEOMETRY_OBJECT = "masked_edgetam_object"
GEOMETRY_TRACKER_OBJECT = "tapnextpp_tracker_markers_object"
GEOMETRY_TRACKER_CONTROLLER = "tapnextpp_tracker_markers_controller"
COORDINATE_FRAME = "camera_color_frame"
TABLE_Z_M = 0.0
DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M = (0.005, 0.010, 0.020, 0.030)
DEFAULT_TABLE_Z_FILTER_THRESHOLD_M = 0.0
# Origin/data_process table frame: z < 0 is above the table, z > 0 is invalid.
TABLE_Z_ABOVE_DIRECTION = "negative"
TABLE_Z_FILTER_CLASS_OBJECT = "object"
TABLE_Z_FILTER_CLASS_CONTROLLER = "controller"
TABLE_Z_FILTER_CLASS_BOTH = "both"
TABLE_Z_FILTER_CLASSES = (
    TABLE_Z_FILTER_CLASS_OBJECT,
    TABLE_Z_FILTER_CLASS_CONTROLLER,
    TABLE_Z_FILTER_CLASS_BOTH,
)
TRACKER_DISPLAY_SCOPE_CONTROLLER = "controller"
TRACKER_DISPLAY_SCOPE_OBJECT = "object"
TRACKER_DISPLAY_SCOPE_UNION = "union"
DEFAULT_TRACKER_DISPLAY_SCOPE = TRACKER_DISPLAY_SCOPE_UNION
DEFAULT_TRACKER_BACKEND = TRACKER_BACKEND_TAPNEXTPP
DEFAULT_TRACKER_QUERY_COUNT = PHYSTWIN_DENSE_QUERY_POINTS
DEFAULT_TRACKER_SEED = 42
DEFAULT_TRACKER_MARKER_POINT_SIZE = 8.0
QUERY_CONTROLLER_INSTANCE_NONE = 0
QUERY_CONTROLLER_INSTANCE_HAND_A = 1
QUERY_CONTROLLER_INSTANCE_HAND_B = 2
HEADLESS_CAPTURE_SAVED_PCD_SOURCE = "full_masked_depth"
DEFAULT_LOSSLESS_INPUT_FPS = 5.0


def table_world_enabled(table_c2w: np.ndarray | None) -> bool:
    """Return whether the run operates in the calibrated table-world frame."""
    return table_c2w is not None


def pcd_coordinate_frame(table_c2w: np.ndarray | None) -> str:
    """Return the coordinate frame PCD products are expressed in."""
    return (
        TABLE_WORLD_FRAME_KIND if table_world_enabled(table_c2w) else COORDINATE_FRAME
    )


# ---------------------------------------------------------------------------
# Shared dataclasses & packet types flowing between pipeline stages
# ---------------------------------------------------------------------------
DEFAULT_LOSSLESS_MAX_BACKLOG_SECONDS = 3.0


DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_TAPNET_REPO_DIR = DEFAULT_RUNTIME_ASSET_ROOT / "tapnet"
DEFAULT_TAPNEXTPP_CHECKPOINT = (
    DEFAULT_RUNTIME_ASSET_ROOT / "checkpoints" / "tapnextpp" / "tapnextpp_ckpt.pt"
)


def _resolve_path(value: str | Path) -> Path:
    """Resolve a filesystem path to an absolute expanded path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


__all__ = [
    _n for _n in list(globals()) if not _n.startswith("__") and _n != "annotations"
]
