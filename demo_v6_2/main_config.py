"""Demo v6.1 realtime orchestration defaults and module-level constants.

This module owns the config/default.yaml loader and every module-level default
so the other Demo v6.1 orchestration modules can reach the ``DEFAULT_*`` values
without an import cycle.
"""

from __future__ import annotations

from pathlib import Path
import sys

import yaml


# Keep this repo at the front of the import path when the script is launched
# from another working directory. Removing the existing entry first avoids a
# duplicate path while preserving the "current checkout wins" import order.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"


def load_default_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load Demo v6.1 defaults from YAML."""
    text = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"default config must be a mapping: {path}")
    return dict(loaded)


_DEFAULT_CONFIG = load_default_config()


def _cfg(section: str, key: str) -> object:
    """Read one default; config/default.yaml is the single source of defaults."""
    return _DEFAULT_CONFIG[section][key]


def _cfg_optional_path(section: str, key: str) -> Path | None:
    """Read an optional path default; empty/None YAML values mean "unset"."""
    value = _cfg(section, key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


# Defaults below describe the current Demo v6.1 realtime path.
DEFAULT_DATA_PROCESS_BASE_PATH = Path(str(_cfg("paths", "data_process_base_path")))
DEFAULT_INPUT_SOURCE = str(_cfg("input", "input_source"))
DEFAULT_FAKE_LIVE_CASE = _cfg_optional_path("input", "fake_live_case")
DEFAULT_REPLAY_FPS = float(_cfg("input", "replay_fps"))
DEFAULT_CHUNK_SECONDS = float(_cfg("chunking", "chunk_seconds"))
DEFAULT_CHUNK_POLL_INTERVAL_S = float(_cfg("chunking", "chunk_poll_interval_s"))
DEFAULT_CAMERA_SOURCE_REPLAY_FPS = float(_cfg("input", "camera_source_replay_fps"))
DEFAULT_CAMERA_FPS = int(_cfg("camera", "camera_fps"))
CAMERA_FPS_CHOICES = tuple(int(item) for item in _cfg("camera", "camera_fps_choices"))
# List schema so a future multi-camera runtime can extend it; the current
# single-camera runtime enforces exactly one entry (resolve_camera_serials).
DEFAULT_CAMERA_SERIALS = tuple(str(item) for item in _cfg("camera", "camera_serials"))
DEFAULT_CAMERA_COLOR_EXPOSURE = float(_cfg("camera", "camera_color_exposure"))
DEFAULT_CAMERA_COLOR_GAIN = float(_cfg("camera", "camera_color_gain"))
DEFAULT_CASE_PREFIX = str(_cfg("camera", "case_prefix"))
DEFAULT_DEPTH_BACKEND = str(_cfg("camera", "depth_backend"))
DEFAULT_MAX_CHUNKS: int | None = (
    None
    if _cfg("chunking", "max_chunks") is None
    else int(_cfg("chunking", "max_chunks"))
)
DEFAULT_SHAPE_PRIOR_TIMEOUT_MS = int(_cfg("shape_prior", "shape_prior_timeout_ms"))
DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S = float(
    _cfg("shape_prior", "shape_prior_chunk_wait_timeout_s")
)
CONFIG_SHAPE_PRIOR_CONTROLLER_NAME = str(
    _cfg("shape_prior", "shape_prior_controller_name")
)
DEFAULT_SHAPE_PRIOR_SAM3D_ROOT = _cfg_optional_path(
    "shape_prior", "shape_prior_sam3d_root"
)
DEFAULT_SHAPE_PRIOR_CONFIG = _cfg_optional_path("shape_prior", "shape_prior_config")
DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "main_data_processing_cuda_visible_devices")
)
DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "shape_prior_warmup_cuda_visible_devices")
)
DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "visualizer_cuda_visible_devices")
)
DEFAULT_PERCEPTION_DEVICE = str(_cfg("camera", "perception_device"))
DEFAULT_TRACKER_DEVICE = str(_cfg("camera", "tracker_device"))
DEFAULT_INFERENCE_DTYPE = str(_cfg("camera", "inference_dtype"))
DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD = float(
    _cfg("camera", "edgetam_mask_logit_threshold")
)
# Downstream consumers are mutually exclusive per session: the Demo v6.1
# viewer window or the Phystwin_shen online trainer + HTML viewer.
DOWNSTREAM_MODE_DISABLED = "disabled"
DOWNSTREAM_MODE_DEMO_VISUALIZER = "demo_visualizer"
DOWNSTREAM_MODE_PHYSTWIN_SHEN = "phystwin_shen"
DOWNSTREAM_MODES = (
    DOWNSTREAM_MODE_DISABLED,
    DOWNSTREAM_MODE_DEMO_VISUALIZER,
    DOWNSTREAM_MODE_PHYSTWIN_SHEN,
)
DEFAULT_DOWNSTREAM_MODE = str(_cfg("downstream", "mode"))
DEFAULT_PHYSTWIN_SHEN_REPO_PATH = Path(str(_cfg("phystwin_shen", "repo_path")))
DEFAULT_PHYSTWIN_SHEN_CONDA_ENV = str(_cfg("phystwin_shen", "conda_env"))
DEFAULT_PHYSTWIN_SHEN_CASE_NAME = str(_cfg("phystwin_shen", "case_name"))
DEFAULT_PHYSTWIN_SHEN_VIEWER_HOST = str(_cfg("phystwin_shen", "host"))
DEFAULT_PHYSTWIN_SHEN_VIEWER_PORT = int(_cfg("phystwin_shen", "port"))
PHYSTWIN_SHEN_VIEWER_CAM_IDX = int(_cfg("phystwin_shen", "cam_idx"))
PHYSTWIN_SHEN_VIEWER_POINT_MODE = str(_cfg("phystwin_shen", "point_mode"))
PHYSTWIN_SHEN_VIEWER_POINT_STRIDE = int(_cfg("phystwin_shen", "point_stride"))
PHYSTWIN_SHEN_TRAIN_DEVICE = str(_cfg("phystwin_shen", "device"))
PHYSTWIN_SHEN_TRAIN_BATCH_SIZE = int(_cfg("phystwin_shen", "batch_size"))
PHYSTWIN_SHEN_TRAIN_SEGMENT_LEN = int(_cfg("phystwin_shen", "segment_len"))
PHYSTWIN_SHEN_TRAIN_SEGMENT_STRIDE = int(_cfg("phystwin_shen", "segment_stride"))
PHYSTWIN_SHEN_TRAIN_POLL_SEC = float(_cfg("phystwin_shen", "poll_sec"))
PHYSTWIN_SHEN_TRAIN_RECENT_WINDOW_COUNT = int(
    _cfg("phystwin_shen", "recent_window_count")
)
PHYSTWIN_SHEN_TRAIN_REALTIME_VIS_EVERY = int(
    _cfg("phystwin_shen", "realtime_vis_every")
)
PHYSTWIN_SHEN_TRAIN_STOP_WHEN_FINISHED = bool(
    _cfg("phystwin_shen", "stop_when_finished")
)
DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "phystwin_shen_cuda_visible_devices")
)
DEFAULT_VISUALIZER_CONDA_ENV = str(_cfg("visualizer", "visualizer_conda_env"))
DEFAULT_VISUALIZER_CAM_IDX = int(_cfg("visualizer", "visualizer_cam_idx"))
DEFAULT_VISUALIZER_POLL_SEC = float(_cfg("visualizer", "visualizer_poll_sec"))
DEFAULT_VISUALIZER_PLAYBACK_FPS = float(_cfg("visualizer", "visualizer_playback_fps"))
DEFAULT_VISUALIZER_OBJECT_STRIDE = int(_cfg("visualizer", "visualizer_object_stride"))
DEFAULT_VISUALIZER_OBJECT_RADIUS = int(_cfg("visualizer", "visualizer_object_radius"))
DEFAULT_VISUALIZER_CONTROLLER_RADIUS = int(
    _cfg("visualizer", "visualizer_controller_radius")
)
DEFAULT_VISUALIZER_OBJECT_COLOR_MODE = str(
    _cfg("visualizer", "visualizer_object_color_mode")
)
VISUALIZER_LAYOUT_SIDE_BY_SIDE = str(
    _cfg("visualizer", "visualizer_layout_side_by_side")
)
VISUALIZER_LAYOUT_OUTPUT_ONLY = str(_cfg("visualizer", "visualizer_layout_output_only"))
VISUALIZER_LAYOUTS = tuple(
    str(item) for item in _cfg("visualizer", "visualizer_layouts")
)
DEFAULT_VISUALIZER_LAYOUT = str(_cfg("visualizer", "visualizer_layout"))
DEFAULT_VISUALIZER_RENDER_MODE = str(_cfg("visualizer", "visualizer_render_mode"))
DEFAULT_TABLE_CALIBRATE_PATH = Path(str(_cfg("paths", "table_calibrate_path")))
DEFAULT_SAM31_CHECKPOINT_PATH = Path(str(_cfg("paths", "sam31_checkpoint_path")))
SAM31_CHECKPOINT_ENV = str(_cfg("paths", "sam31_checkpoint_env"))
EDGE_TAM_TRACKING_IDENTITIES = ("hand_a", "object", "hand_b")

CAPTURE_DIR_NAME = "capture"
DATA_DIR_NAME = "data"
ONLINE_DATA_DIR_NAME = "online_data"
SHAPE_PRIOR_CASE_DIR_NAME = "shape_prior_case"
SHAPE_PRIOR_DIR_NAME = "shape_prior"
RUN_SUMMARY_NAME = "run_summary.json"
