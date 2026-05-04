#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "data_process").is_dir() and (root / "demo_v2").is_dir():
            return root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_v2.realtime_single_camera_pointcloud import (  # noqa: E402
    CameraIntrinsics,
    CoalescedPostGate,
    ColorFloat32Buffer,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    FfsIrToColorAligner,
    LatestSlot,
    RenderStats,
    SUPPORTED_CAPTURE_FPS,
    SUPPORTED_PROFILES,
    _apply_emitter,
    _elapsed_ms,
    _load_open3d_modules,
    _load_realsense_module,
    apply_wslg_open3d_env_defaults,
    build_projection_grid,
    camera_intrinsics_from_rs,
    ensure_float32_c_contiguous,
    parse_profile,
    pointcloud_update_requires_readd,
    resolve_serial,
    rs_extrinsics_to_matrix,
    rs_intrinsics_to_matrix,
    rs_translation_norm,
    validate_ffs_paths,
    warm_up_numba_ffs_align,
)


DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_COMPILE_MODE = "vision-reduce-overhead"
COMPILE_MODES = ("vision-reduce-overhead",)
INIT_MODES = ("sam31-first-frame", "saved-masks")
DEFAULT_INIT_MODE = "sam31-first-frame"
TRACK_MODES = ("controller-object", "object-only")
DEFAULT_TRACK_MODE = "controller-object"
DEPTH_SOURCES = ("ffs", "realsense")
DEFAULT_DEPTH_SOURCE = "ffs"
CONTROLLER_ID = 1
OBJECT_ID = 2
OBJECT_LABELS = {CONTROLLER_ID: "controller", OBJECT_ID: "object"}
CONTROLLER_COLOR_RGB = (255, 96, 32)
OBJECT_COLOR_RGB = (64, 180, 255)
GEOMETRY_CONTROLLER = "masked_edgetam_controller"
GEOMETRY_OBJECT = "masked_edgetam_object"
COORDINATE_FRAME = "camera_color_frame"
DEBUG_LOG_INTERVAL_S = 1.0
WARMUP_HUD_TEXT = (
    "System warming up. Keep one steady pose.\n"
    "SAM3.1 first-frame initialization and compiled EdgeTAM startup are running..."
)


@dataclass(frozen=True)
class PipelineTiming:
    wait_ms: float = 0.0
    align_ms: float = 0.0
    frame_copy_ms: float = 0.0
    ffs_ms: float = 0.0
    ffs_align_ms: float = 0.0
    preprocess_ms: float = 0.0
    prompt_ms: float = 0.0
    model_ms: float = 0.0
    postprocess_ms: float = 0.0
    mask_ms: float = 0.0
    pcd_ms: float = 0.0
    open3d_convert_ms: float = 0.0
    open3d_update_ms: float = 0.0
    receive_to_render_ms: float = 0.0


@dataclass(frozen=True)
class RealtimeCameraRuntime:
    pipeline: object
    align: object | None
    serial: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    k_color: np.ndarray
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class FramePacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float
    timing: PipelineTiming
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class MaskPacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    timing: PipelineTiming
    controller_mask: np.ndarray
    object_mask: np.ndarray
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class MaskedPcdPacket:
    seq: int
    controller_xyz_m: np.ndarray
    controller_colors_rgb_u8: np.ndarray
    object_xyz_m: np.ndarray
    object_colors_rgb_u8: np.ndarray
    intrinsics: CameraIntrinsics
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    dropped_seg_frames: int
    timing: PipelineTiming

    @property
    def controller_point_count(self) -> int:
        return int(self.controller_xyz_m.shape[0])

    @property
    def object_point_count(self) -> int:
        return int(self.object_xyz_m.shape[0])

    @property
    def point_count(self) -> int:
        return self.controller_point_count + self.object_point_count


class StageStats:
    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def record(self, now_s: float | None = None) -> None:
        now = time.perf_counter() if now_s is None else float(now_s)
        with self._lock:
            self._times.append(now)
            cutoff = now - self.window_s
            while len(self._times) > 1 and self._times[0] < cutoff:
                self._times.popleft()

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1] - self._times[0]
            if elapsed <= 0:
                return 0.0
            return float((len(self._times) - 1) / elapsed)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    try:
        rgb = tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer R,G,B") from exc
    if any(item < 0 or item > 255 for item in rgb):
        raise argparse.ArgumentTypeError("R,G,B values must be in [0, 255]")
    return rgb  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-D455 realtime HF EdgeTAM masked point-cloud demo. Captures live "
            "RealSense color plus FFS stereo depth by default, tracks controller/object "
            "or object-only with one HF EdgeTAM streaming session, and renders only the masked PCD."
        )
    )
    parser.add_argument("--serial", default=None, help="Optional RealSense D400 serial. Defaults to first detected D400.")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES, default=DEFAULT_PROFILE, help="Capture profile.")
    parser.add_argument("--fps", choices=SUPPORTED_CAPTURE_FPS, type=int, default=DEFAULT_FPS, help="Capture FPS.")
    parser.add_argument(
        "--depth-source",
        choices=DEPTH_SOURCES,
        default=DEFAULT_DEPTH_SOURCE,
        help="Depth source. ffs streams color+IR stereo and runs the repo default TensorRT FFS depth path.",
    )
    parser.add_argument(
        "--ffs-repo",
        type=Path,
        default=DEFAULT_FFS_REPO,
        help="Fast-FoundationStereo repo path. Used when --depth-source ffs.",
    )
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help=(
            "Two-stage TensorRT FFS engine directory. Default is the 20-30-48 / "
            "valid_iters=4 / 848x480->864x480 / builderOptimizationLevel=5 artifact."
        ),
    )
    parser.add_argument(
        "--ffs-trt-root",
        type=Path,
        default=None,
        help="Optional TensorRT Python package/root override forwarded to the FFS runner.",
    )
    parser.add_argument(
        "--emitter",
        choices=("auto", "on", "off"),
        default="auto",
        help="RealSense emitter policy. Defaults to leaving the current device setting unchanged.",
    )
    parser.add_argument(
        "--init-mode",
        choices=INIT_MODES,
        default=DEFAULT_INIT_MODE,
        help="Frame-0 initialization mode. Default runs SAM3.1 once on the live first frame.",
    )
    parser.add_argument(
        "--track-mode",
        choices=TRACK_MODES,
        default=DEFAULT_TRACK_MODE,
        help="Objects tracked by EdgeTAM. Default tracks controller+object; object-only skips controller masks.",
    )
    parser.add_argument(
        "--controller-init-mask",
        default=None,
        help="Binary frame-0 controller mask PNG for explicit saved-masks debugging mode.",
    )
    parser.add_argument(
        "--object-init-mask",
        default=None,
        help="Binary frame-0 object mask PNG for explicit saved-masks debugging mode.",
    )
    parser.add_argument(
        "--controller-prompt",
        default="hand",
        help="SAM3.1 prompt label to union as controller in sam31-first-frame mode.",
    )
    parser.add_argument(
        "--object-prompt",
        default="stuffed animal",
        help="SAM3.1 prompt label to use as object in sam31-first-frame mode.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF EdgeTAM model id.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, usually cuda.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=DEFAULT_DTYPE, help="Inference dtype.")
    parser.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default=DEFAULT_COMPILE_MODE,
        help="Required EdgeTAM compile mode. Compiles only vision_encoder with reduce-overhead.",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.2, help="Minimum valid depth in meters.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Maximum valid depth in meters. Use <=0 to disable.")
    parser.add_argument(
        "--pcd-max-points",
        type=int,
        default=60000,
        help="Max masked points per object. Use 0 to keep every masked valid depth pixel.",
    )
    parser.add_argument(
        "--pcd-color-mode",
        choices=("rgb", "class"),
        default="rgb",
        help="Point-cloud colors. rgb uses the live color frame; class uses fixed controller/object colors.",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument("--render-every-n", type=int, default=1, help="Render every Nth PCD packet.")
    parser.add_argument("--latency-target-ms", type=float, default=80.0, help="HUD latency target.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional auto-stop duration. Use 0 to run until closed.")
    parser.add_argument("--controller-color", type=_parse_rgb_triplet, default=CONTROLLER_COLOR_RGB, help="Controller RGB color.")
    parser.add_argument("--object-color", type=_parse_rgb_triplet, default=OBJECT_COLOR_RGB, help="Object RGB color.")
    parser.add_argument("--debug", action="store_true", help="Print once-per-second timing/debug stats.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    parse_profile(args.profile)
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if args.depth_max_m > 0 and args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be <=0 or greater than --depth-min-m")
    if args.pcd_max_points < 0:
        raise ValueError("--pcd-max-points must be >= 0")
    if args.render_every_n < 1:
        raise ValueError("--render-every-n must be >= 1")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.compile_mode != DEFAULT_COMPILE_MODE:
        raise ValueError("Demo 2.0 requires compiled EdgeTAM: --compile-mode vision-reduce-overhead")
    if args.depth_source == "ffs":
        validate_ffs_paths(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.ffs_trt_model_dir))
    if args.track_mode not in TRACK_MODES:
        raise ValueError(f"--track-mode must be one of {', '.join(TRACK_MODES)}")
    if args.init_mode == "saved-masks":
        if not args.object_init_mask:
            raise ValueError("saved-masks mode requires --object-init-mask")
        if controller_tracking_enabled(args) and not args.controller_init_mask:
            raise ValueError("saved-masks controller-object mode requires --controller-init-mask")
        required_masks = [("--object-init-mask", args.object_init_mask)]
        if controller_tracking_enabled(args):
            required_masks.append(("--controller-init-mask", args.controller_init_mask))
        for flag, value in required_masks:
            path = _resolve_path(value)
            if not path.is_file():
                raise ValueError(f"{flag} does not exist: {path}")


def _start_realsense_pipeline(args: argparse.Namespace) -> RealtimeCameraRuntime:
    rs = _load_realsense_module()
    width, height = parse_profile(args.profile)
    serial = resolve_serial(rs, args.serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(args.fps))
    if args.depth_source == "ffs":
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, int(args.fps))
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, int(args.fps))
    else:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(args.fps))
    profile = pipeline.start(config)
    try:
        _apply_emitter(profile, args.emitter, rs)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = camera_intrinsics_from_rs(color_stream.get_intrinsics())
        k_color = rs_intrinsics_to_matrix(color_stream.get_intrinsics())
        if args.depth_source == "ffs":
            ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
            ir_left_to_color = ir_left_profile.get_extrinsics_to(color_stream)
            return RealtimeCameraRuntime(
                pipeline=pipeline,
                align=None,
                serial=serial,
                intrinsics=intrinsics,
                depth_scale_m_per_unit=depth_scale,
                k_color=k_color,
                k_ir_left=rs_intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
                t_ir_left_to_color=rs_extrinsics_to_matrix(ir_left_to_color),
                ir_baseline_m=rs_translation_norm(ir_left_to_right),
            )
        align = rs.align(rs.stream.color)
    except Exception:
        pipeline.stop()
        raise
    return RealtimeCameraRuntime(
        pipeline=pipeline,
        align=align,
        serial=serial,
        intrinsics=intrinsics,
        depth_scale_m_per_unit=depth_scale,
        k_color=k_color,
    )


def _load_gray_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))
    except Exception as exc:
        raise ValueError(f"failed to load mask image {path}: {exc}") from exc


def load_binary_mask(path: str | Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_path = _resolve_path(path)
    image = _load_gray_image(mask_path)
    if image.ndim != 2:
        raise ValueError(f"mask must be a 2D image: {mask_path}")
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(f"mask shape {tuple(image.shape)} does not match frame shape {tuple(expected_shape)}: {mask_path}")
    return np.ascontiguousarray(image > 0)


def _masked_sample_indices(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_m.ndim != 2 or mask.ndim != 2:
        raise ValueError("depth_m and mask must be 2D arrays")
    if depth_m.shape != mask.shape:
        raise ValueError("depth and mask shapes must match")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    valid = np.isfinite(depth_m) & (depth_m > np.float32(depth_min_m))
    if depth_max_m > 0:
        valid &= depth_m < np.float32(depth_max_m)
    selected = valid & np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rows, cols = np.nonzero(selected)
    if max_points > 0 and rows.shape[0] > max_points:
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.choice(rows.shape[0], int(max_points), replace=False)
        rows = rows[indices]
        cols = cols[indices]
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False)


def backproject_masked_rgbd(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    color_mode: str,
    class_rgb: tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if depth_m.shape != color_bgr.shape[:2]:
        raise ValueError("color and depth shapes must match")
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    if color_mode not in {"rgb", "class"}:
        raise ValueError("color_mode must be 'rgb' or 'class'")

    rows, cols = _masked_sample_indices(
        depth_m=depth_m,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        max_points=max_points,
        rng=rng,
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    points = np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)
    if color_mode == "rgb":
        colors = np.ascontiguousarray(color_bgr[rows, cols, ::-1], dtype=np.uint8)
    else:
        colors = make_solid_colors(points.shape[0], class_rgb)
    return points, colors


def backproject_masked(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    rows, cols = _masked_sample_indices(
        depth_m=depth_m,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        max_points=max_points,
        rng=rng,
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    return np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)


def make_solid_colors(point_count: int, rgb: tuple[int, int, int]) -> np.ndarray:
    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    color = np.asarray(rgb, dtype=np.uint8).reshape(1, 3)
    return np.repeat(color, int(point_count), axis=0)


def controller_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) == "controller-object"


def object_id_labels(track_mode: str = DEFAULT_TRACK_MODE) -> dict[int, str]:
    if track_mode == "object-only":
        return {OBJECT_ID: OBJECT_LABELS[OBJECT_ID]}
    if track_mode == "controller-object":
        return dict(OBJECT_LABELS)
    raise ValueError(f"unsupported track mode: {track_mode}")


def active_object_ids(args: argparse.Namespace) -> list[int]:
    return list(object_id_labels(str(args.track_mode)).keys())


def _coerce_object_ids(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(item) for item in list(value)]


def _extract_binary_mask(mask_tensor: Any) -> np.ndarray:
    value = mask_tensor
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"expected 2D mask after squeeze, got {array.shape}")
    return np.ascontiguousarray(array > 0)


def extract_object_masks_from_hf_output(output: Any, post_masks: Any) -> dict[int, np.ndarray]:
    object_ids = _coerce_object_ids(getattr(output, "object_ids"))
    if len(object_ids) != len(post_masks):
        raise RuntimeError(f"HF output object_ids length {len(object_ids)} != mask length {len(post_masks)}")
    return {int(obj_id): _extract_binary_mask(post_masks[idx]) for idx, obj_id in enumerate(object_ids)}


def _load_hf_streaming_runtime() -> Any:
    from scripts.harness.experiments import run_hf_edgetam_streaming_realcase as hf_stream

    hf_stream._load_runtime_dependencies()
    return hf_stream


def _sync_if_needed(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _time_runtime_ms(torch_module: Any, device: str, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync_if_needed(torch_module, device)
    started = time.perf_counter()
    value = fn()
    _sync_if_needed(torch_module, device)
    return value, _elapsed_ms(started, time.perf_counter())


def _bgr_to_pil_rgb(color_bgr: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(color_bgr[:, :, ::-1]))


def _write_first_frame_case(color_bgr: np.ndarray, root: Path) -> Path:
    case_dir = root / "sam31_frame0_case"
    color_dir = case_dir / "color" / "0"
    color_dir.mkdir(parents=True, exist_ok=True)
    image = _bgr_to_pil_rgb(color_bgr)
    image.save(color_dir / "0.png")
    return case_dir


def _load_label_masks_from_sam31_root(
    *,
    mask_root: Path,
    label: str,
    frame_token: str = "0",
    camera_idx: int = 0,
) -> list[np.ndarray]:
    info_path = mask_root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"SAM3.1 mask info not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    label_norm = str(label).strip().lower()
    masks: list[np.ndarray] = []
    for obj_id, obj_label in info.items():
        if str(obj_label).strip().lower() != label_norm:
            continue
        mask_path = mask_root / "mask" / str(int(camera_idx)) / str(obj_id) / f"{frame_token}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"SAM3.1 mask image not found: {mask_path}")
        masks.append(np.ascontiguousarray(_load_gray_image(mask_path) > 0))
    return masks


def _union_masks(masks: list[np.ndarray], *, label: str) -> np.ndarray:
    if not masks:
        raise RuntimeError(f"SAM3.1 did not produce a mask for label {label!r}")
    output = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != output.shape:
            raise RuntimeError("SAM3.1 masks for one label have inconsistent shapes")
        output |= mask.astype(bool)
    return np.ascontiguousarray(output)


def release_sam31_runtime_resources(device: str = DEFAULT_DEVICE) -> None:
    helper = sys.modules.get("scripts.harness.sam31_mask_helper")
    autocast_context = getattr(helper, "_CUDA_AUTOCAST_CONTEXT", None) if helper is not None else None
    if autocast_context is not None:
        try:
            autocast_context.__exit__(None, None, None)
        except Exception as exc:
            print(f"[WARN] SAM3.1 autocast cleanup failed: {type(exc).__name__}: {exc}", flush=True)
        if helper is not None:
            setattr(helper, "_CUDA_AUTOCAST_CONTEXT", None)

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] SAM3.1 CUDA cleanup failed: {type(exc).__name__}: {exc}", flush=True)


def run_sam31_first_frame_masks(color_bgr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="qqtt_sam31_first_frame_") as tmp:
        root = Path(tmp)
        case_dir = _write_first_frame_case(color_bgr, root)
        output_dir = root / "sam31_masks"
        from scripts.harness.sam31_mask_helper import run_case_segmentation

        prompt_labels = [str(args.object_prompt)]
        if controller_tracking_enabled(args):
            prompt_labels.append(str(args.controller_prompt))
        try:
            run_case_segmentation(
                case_root=case_dir,
                text_prompt=",".join(prompt_labels),
                camera_ids=(0,),
                output_dir=output_dir,
                source_mode="frames",
                checkpoint_path=None,
                ann_frame_index=0,
                keep_session_frames=False,
                session_root=None,
                overwrite=True,
                async_loading_frames=False,
                compile_model=False,
                max_num_objects=16,
            )
        finally:
            release_sam31_runtime_resources(str(args.device))
        object_masks = _load_label_masks_from_sam31_root(mask_root=output_dir, label=args.object_prompt)
        object_mask = _union_masks(
            object_masks,
            label=args.object_prompt,
        )
        if not controller_tracking_enabled(args):
            return np.zeros_like(object_mask, dtype=bool), object_mask
        controller_masks = _load_label_masks_from_sam31_root(mask_root=output_dir, label=args.controller_prompt)
        return _union_masks(controller_masks, label=args.controller_prompt), object_mask


def resolve_initial_masks(frame: FramePacket, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = tuple(frame.color_bgr.shape[:2])
    if args.init_mode == "saved-masks":
        object_mask = load_binary_mask(args.object_init_mask, expected_shape=expected_shape)
        if not controller_tracking_enabled(args):
            return np.zeros_like(object_mask, dtype=bool), object_mask
        controller_mask = load_binary_mask(args.controller_init_mask, expected_shape=expected_shape)
        return controller_mask, object_mask
    if args.init_mode == "sam31-first-frame":
        controller_mask, object_mask = run_sam31_first_frame_masks(frame.color_bgr, args)
        if controller_mask.shape != expected_shape or object_mask.shape != expected_shape:
            raise RuntimeError("SAM3.1 frame-0 masks do not match captured frame shape")
        return controller_mask, object_mask
    raise ValueError(f"unsupported init mode: {args.init_mode}")


class RealtimeMaskedEdgeTamPcdDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.width, self.height = parse_profile(args.profile)
        self.runtime: RealtimeCameraRuntime | None = None
        self.ray_x: np.ndarray | None = None
        self.ray_y: np.ndarray | None = None
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.mask_slot: LatestSlot[MaskPacket] = LatestSlot()
        self.render_slot: LatestSlot[MaskedPcdPacket] = LatestSlot()
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._request_render_update: Callable[[], None] = lambda: None
        self.capture_stats = StageStats()
        self.seg_stats = StageStats()
        self.pcd_stats = StageStats()
        self.render_stats = RenderStats()
        self._last_debug_log_s = 0.0
        self.ffs_runner: object | None = None
        self.ir_to_color_aligner: FfsIrToColorAligner | None = None
        self._ir_to_color_aligner_key: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[float, ...],
            tuple[float, ...],
            tuple[float, ...],
        ] | None = None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        return self.runtime.intrinsics

    @property
    def serial(self) -> str:
        if self.runtime is None:
            return "<not-started>"
        return self.runtime.serial

    def run(self) -> int:
        apply_wslg_open3d_env_defaults()
        if self.args.depth_source == "ffs":
            self.ffs_runner = self._create_ffs_runner()
            warm_up_numba_ffs_align()
        self.runtime = _start_realsense_pipeline(self.args)
        try:
            self.ray_x, self.ray_y = build_projection_grid(
                width=self.width,
                height=self.height,
                stride=1,
                intrinsics=self.runtime.intrinsics,
            )
            self._run_open3d_viewer()
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self.stop_event.set()
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._threads.clear()
        if self.runtime is not None:
            try:
                self.runtime.pipeline.stop()
            except Exception:
                pass
            self.runtime = None

    def _create_ffs_runner(self) -> object:
        try:
            from data_process.depth_backends import FastFoundationStereoTensorRTRunner

            return FastFoundationStereoTensorRTRunner(
                ffs_repo=Path(self.args.ffs_repo),
                model_dir=Path(self.args.ffs_trt_model_dir),
                trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to start FFS TensorRT runner: {type(exc).__name__}: {exc}") from exc

    def _get_ir_to_color_aligner(
        self,
        *,
        depth_shape: tuple[int, int],
        color_shape: tuple[int, int],
        k_ir_left: np.ndarray,
        t_ir_left_to_color: np.ndarray,
        k_color: np.ndarray,
    ) -> FfsIrToColorAligner:
        k_ir = np.asarray(k_ir_left, dtype=np.float32).reshape(3, 3)
        transform = np.asarray(t_ir_left_to_color, dtype=np.float32).reshape(4, 4)
        k_col = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
        key = (
            (int(depth_shape[0]), int(depth_shape[1])),
            (int(color_shape[0]), int(color_shape[1])),
            tuple(float(v) for v in k_ir.ravel()),
            tuple(float(v) for v in transform.ravel()),
            tuple(float(v) for v in k_col.ravel()),
        )
        if self._ir_to_color_aligner_key != key or self.ir_to_color_aligner is None:
            self.ir_to_color_aligner = FfsIrToColorAligner(
                k_ir_left=k_ir,
                t_ir_left_to_color=transform,
                k_color=k_col,
                ir_shape=depth_shape,
                color_shape=color_shape,
            )
            self._ir_to_color_aligner_key = key
        return self.ir_to_color_aligner

    def _start_threads(self) -> None:
        workers = (
            ("capture", self._capture_worker),
            ("seg", self._seg_worker),
            ("pcd", self._pcd_worker),
        )
        for name, target in workers:
            thread = threading.Thread(target=target, name=f"masked-edgetam-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _capture_worker(self) -> None:
        assert self.runtime is not None
        seq = 0
        pipeline = self.runtime.pipeline
        align = self.runtime.align
        while not self.stop_event.is_set():
            wait_start_s = time.perf_counter()
            try:
                frames = pipeline.wait_for_frames()
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[ERROR] RealSense capture failed: {type(exc).__name__}: {exc}", flush=True)
                self.stop_event.set()
                break
            receive_perf_s = time.perf_counter()
            align_start_s = receive_perf_s
            if self.args.depth_source == "ffs":
                align_done_s = receive_perf_s
                color_frame = frames.get_color_frame()
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2)
                if not color_frame or not ir_left_frame or not ir_right_frame:
                    continue
                depth_frame = None
            else:
                assert align is not None
                aligned = align.process(frames)
                align_done_s = time.perf_counter()
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                ir_left_frame = None
                ir_right_frame = None
            copy_start_s = time.perf_counter()
            color_bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()).copy())
            if self.args.depth_source == "ffs":
                assert ir_left_frame is not None and ir_right_frame is not None
                depth_u16 = None
                ir_left_u8 = np.ascontiguousarray(np.asanyarray(ir_left_frame.get_data()).copy())
                ir_right_u8 = np.ascontiguousarray(np.asanyarray(ir_right_frame.get_data()).copy())
            else:
                assert depth_frame is not None
                depth_u16 = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())
                ir_left_u8 = None
                ir_right_u8 = None
            copy_done_s = time.perf_counter()
            packet = FramePacket(
                seq=seq,
                color_bgr=color_bgr,
                depth_source=str(self.args.depth_source),
                intrinsics=self.runtime.intrinsics,
                depth_scale_m_per_unit=self.runtime.depth_scale_m_per_unit,
                receive_perf_s=receive_perf_s,
                timing=PipelineTiming(
                    wait_ms=_elapsed_ms(wait_start_s, receive_perf_s),
                    align_ms=_elapsed_ms(align_start_s, align_done_s),
                    frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
                ),
                depth_u16=depth_u16,
                ir_left_u8=ir_left_u8,
                ir_right_u8=ir_right_u8,
                k_ir_left=self.runtime.k_ir_left,
                t_ir_left_to_color=self.runtime.t_ir_left_to_color,
                k_color=self.runtime.k_color,
                ir_baseline_m=self.runtime.ir_baseline_m,
            )
            self.capture_slot.put(packet)
            self.capture_stats.record(copy_done_s)
            seq += 1

    def _init_hf_model(self) -> tuple[Any, Any, Any, Any, Any]:
        hf_stream = _load_hf_streaming_runtime()
        torch_module = hf_stream.torch
        if str(self.args.device).startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype = hf_stream._dtype_from_name(self.args.dtype)
        model = hf_stream.EdgeTamVideoModel.from_pretrained(self.args.model_id).to(
            self.args.device,
            dtype=dtype,
        )
        model.eval()
        model, compile_metadata = hf_stream._apply_compile_mode(model, self.args.compile_mode)
        processor = hf_stream.Sam2VideoProcessor.from_pretrained(self.args.model_id)
        print(
            "[edgetam] "
            f"model={self.args.model_id} device={self.args.device} dtype={self.args.dtype} "
            f"track_mode={self.args.track_mode} compile_mode={self.args.compile_mode} "
            f"applied={compile_metadata.get('applied_targets', [])}",
            flush=True,
        )
        return hf_stream, torch_module, dtype, model, processor

    def _seg_worker(self) -> None:
        try:
            hf_stream, torch_module, dtype, model, processor = self._init_hf_model()
            first_frame = self._wait_for_first_frame()
            if first_frame is None:
                return
            controller_mask, object_mask = resolve_initial_masks(first_frame, self.args)
            session = hf_stream.EdgeTamVideoInferenceSession(
                video=None,
                video_height=int(first_frame.color_bgr.shape[0]),
                video_width=int(first_frame.color_bgr.shape[1]),
                inference_device=self.args.device,
                inference_state_device=self.args.device,
                video_storage_device=self.args.device,
                dtype=dtype,
            )
            with torch_module.inference_mode():
                first_packet = self._run_segmentation_frame(
                    hf_stream=hf_stream,
                    torch_module=torch_module,
                    dtype=dtype,
                    model=model,
                    processor=processor,
                    session=session,
                    frame=first_frame,
                    initial_controller_mask=controller_mask,
                    initial_object_mask=object_mask,
                    add_prompt=True,
                )
                self.mask_slot.put(first_packet)
                self.seg_stats.record(first_packet.process_done_perf_s)
                last_seq = first_frame.seq
                while not self.stop_event.is_set():
                    frame = self.capture_slot.get_latest_after(last_seq)
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    last_seq = frame.seq
                    try:
                        packet = self._run_segmentation_frame(
                            hf_stream=hf_stream,
                            torch_module=torch_module,
                            dtype=dtype,
                            model=model,
                            processor=processor,
                            session=session,
                            frame=frame,
                            initial_controller_mask=controller_mask,
                            initial_object_mask=object_mask,
                            add_prompt=False,
                        )
                    except Exception as exc:
                        print(f"[ERROR] EdgeTAM segmentation failed: {type(exc).__name__}: {exc}", flush=True)
                        self.stop_event.set()
                        break
                    self.mask_slot.put(packet)
                    self.seg_stats.record(packet.process_done_perf_s)
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] segmentation worker failed: {type(exc).__name__}: {exc}", flush=True)
            self.stop_event.set()

    def _wait_for_first_frame(self) -> FramePacket | None:
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(-1)
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def _autocast_context(self, torch_module: Any) -> Any:
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = torch_module.bfloat16 if self.args.dtype == "bfloat16" else torch_module.float16
        return torch_module.autocast("cuda", dtype=dtype)

    def _run_segmentation_frame(
        self,
        *,
        hf_stream: Any,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        session: Any,
        frame: FramePacket,
        initial_controller_mask: np.ndarray,
        initial_object_mask: np.ndarray,
        add_prompt: bool,
    ) -> MaskPacket:
        image = _bgr_to_pil_rgb(frame.color_bgr)
        inputs, preprocess_ms = _time_runtime_ms(
            torch_module,
            self.args.device,
            lambda: processor(images=image, device=self.args.device, return_tensors="pt"),
        )
        pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
        prompt_ms = 0.0
        with self._autocast_context(torch_module):
            if add_prompt:
                prompt_obj_ids: list[int] = []
                prompt_masks: list[np.ndarray] = []
                if controller_tracking_enabled(self.args):
                    prompt_obj_ids.append(CONTROLLER_ID)
                    prompt_masks.append(np.asarray(initial_controller_mask, dtype=bool))
                prompt_obj_ids.append(OBJECT_ID)
                prompt_masks.append(np.asarray(initial_object_mask, dtype=bool))
                _unused, prompt_ms = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=prompt_obj_ids,
                        input_masks=prompt_masks,
                    ),
                )
            output, model_ms = _time_runtime_ms(
                torch_module,
                self.args.device,
                lambda: model(inference_session=session, frame=pixel_values),
            )
            post_masks, postprocess_ms = _time_runtime_ms(
                torch_module,
                self.args.device,
                lambda: processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0],
            )
        masks_by_id = extract_object_masks_from_hf_output(output, post_masks)
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids: {missing}")
        object_mask = masks_by_id[OBJECT_ID]
        controller_mask = masks_by_id.get(CONTROLLER_ID)
        if controller_mask is None:
            controller_mask = np.zeros_like(object_mask, dtype=bool)
        process_done_s = time.perf_counter()
        timing = replace(
            frame.timing,
            preprocess_ms=preprocess_ms,
            prompt_ms=prompt_ms,
            model_ms=model_ms,
            postprocess_ms=postprocess_ms,
            mask_ms=float(preprocess_ms + prompt_ms + model_ms + postprocess_ms),
        )
        return MaskPacket(
            seq=frame.seq,
            color_bgr=frame.color_bgr,
            depth_source=frame.depth_source,
            intrinsics=frame.intrinsics,
            depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
            receive_perf_s=frame.receive_perf_s,
            process_done_perf_s=process_done_s,
            dropped_capture_frames=self.capture_slot.dropped_count,
            timing=timing,
            controller_mask=controller_mask,
            object_mask=object_mask,
            depth_u16=frame.depth_u16,
            ir_left_u8=frame.ir_left_u8,
            ir_right_u8=frame.ir_right_u8,
            k_ir_left=frame.k_ir_left,
            t_ir_left_to_color=frame.t_ir_left_to_color,
            k_color=frame.k_color,
            ir_baseline_m=frame.ir_baseline_m,
        )

    def _pcd_worker(self) -> None:
        last_seq = -1
        rng = np.random.default_rng()
        assert self.ray_x is not None and self.ray_y is not None
        ray_x = self.ray_x
        ray_y = self.ray_y
        while not self.stop_event.is_set():
            mask_packet = self.mask_slot.get_latest_after(last_seq)
            if mask_packet is None:
                time.sleep(0.001)
                continue
            last_seq = mask_packet.seq
            start_s = time.perf_counter()
            ffs_ms = 0.0
            ffs_align_ms = 0.0
            if mask_packet.depth_source == "ffs":
                try:
                    depth_m, ffs_ms, ffs_align_ms = self._compute_ffs_depth_color_m(mask_packet)
                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"[WARN] FFS depth frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
            else:
                if mask_packet.depth_u16 is None:
                    continue
                depth_m = np.ascontiguousarray(
                    mask_packet.depth_u16.astype(np.float32) * np.float32(mask_packet.depth_scale_m_per_unit)
                )
            controller_xyz, controller_colors = backproject_masked_rgbd(
                color_bgr=mask_packet.color_bgr,
                depth_m=depth_m,
                mask=mask_packet.controller_mask,
                ray_x=ray_x,
                ray_y=ray_y,
                depth_min_m=float(self.args.depth_min_m),
                depth_max_m=float(self.args.depth_max_m),
                max_points=int(self.args.pcd_max_points),
                color_mode=str(self.args.pcd_color_mode),
                class_rgb=tuple(self.args.controller_color),
                rng=rng,
            )
            object_xyz, object_colors = backproject_masked_rgbd(
                color_bgr=mask_packet.color_bgr,
                depth_m=depth_m,
                mask=mask_packet.object_mask,
                ray_x=ray_x,
                ray_y=ray_y,
                depth_min_m=float(self.args.depth_min_m),
                depth_max_m=float(self.args.depth_max_m),
                max_points=int(self.args.pcd_max_points),
                color_mode=str(self.args.pcd_color_mode),
                class_rgb=tuple(self.args.object_color),
                rng=rng,
            )
            done_s = time.perf_counter()
            timing = replace(
                mask_packet.timing,
                ffs_ms=ffs_ms,
                ffs_align_ms=ffs_align_ms,
                pcd_ms=_elapsed_ms(start_s, done_s),
            )
            packet = MaskedPcdPacket(
                seq=mask_packet.seq,
                controller_xyz_m=controller_xyz,
                controller_colors_rgb_u8=controller_colors,
                object_xyz_m=object_xyz,
                object_colors_rgb_u8=object_colors,
                intrinsics=mask_packet.intrinsics,
                receive_perf_s=mask_packet.receive_perf_s,
                process_done_perf_s=done_s,
                dropped_capture_frames=mask_packet.dropped_capture_frames,
                dropped_seg_frames=self.mask_slot.dropped_count,
                timing=timing,
            )
            self.render_slot.put(packet)
            self.pcd_stats.record(done_s)
            if packet.seq % int(self.args.render_every_n) == 0:
                self._request_render_update()

    def _compute_ffs_depth_color_m(self, packet: MaskPacket) -> tuple[np.ndarray, float, float]:
        runner = self.ffs_runner
        if runner is None:
            raise RuntimeError("FFS runner is not initialized")
        if (
            packet.ir_left_u8 is None
            or packet.ir_right_u8 is None
            or packet.k_ir_left is None
            or packet.t_ir_left_to_color is None
            or packet.k_color is None
            or packet.ir_baseline_m <= 0
        ):
            raise RuntimeError("FFS packet is missing IR stereo calibration/data")

        ffs_start_s = time.perf_counter()
        output = runner.run_pair(
            packet.ir_left_u8,
            packet.ir_right_u8,
            K_ir_left=packet.k_ir_left,
            baseline_m=float(packet.ir_baseline_m),
        )
        ffs_done_s = time.perf_counter()
        depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
        k_ir_left_used = np.asarray(output.get("K_ir_left_used", packet.k_ir_left), dtype=np.float32)
        align_start_s = time.perf_counter()
        aligner = self._get_ir_to_color_aligner(
            depth_shape=depth_ir_left_m.shape,
            color_shape=packet.color_bgr.shape[:2],
            k_ir_left=k_ir_left_used,
            t_ir_left_to_color=packet.t_ir_left_to_color,
            k_color=packet.k_color,
        )
        depth_color_m = np.ascontiguousarray(aligner.align(depth_ir_left_m), dtype=np.float32)
        align_done_s = time.perf_counter()
        return (
            depth_color_m,
            _elapsed_ms(ffs_start_s, ffs_done_s),
            _elapsed_ms(align_start_s, align_done_s),
        )

    def _run_open3d_viewer(self) -> None:
        o3d, gui, rendering = _load_open3d_modules()
        o3c = o3d.core
        device = o3c.Device("CPU:0")
        app = gui.Application.instance
        app.initialize()
        window = app.create_window("Demo 2.0 Realtime EdgeTAM Masked PCD", 1280, 800)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])
        hud_label = gui.Label(WARMUP_HUD_TEXT)
        hud_label.text_color = gui.Color(1.0, 1.0, 1.0)
        hud_panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        hud_panel.add_child(hud_label)
        window.add_child(scene_widget)
        window.add_child(hud_panel)

        def on_layout(layout_context: object) -> None:
            rect = window.content_rect
            scene_widget.frame = rect
            em = window.theme.font_size
            preferred = hud_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
            hud_panel.frame = gui.Rect(
                rect.x + 0.5 * em,
                rect.y + 0.5 * em,
                max(preferred.width, 660),
                max(preferred.height, (13.0 if self.args.debug else 9.0) * em),
            )

        window.set_on_layout(on_layout)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = float(self.args.point_size)

        class GeometryState:
            def __init__(self, name: str) -> None:
                self.name = name
                self.pcd = o3d.t.geometry.PointCloud(device)
                self.color_buffer = ColorFloat32Buffer()
                self.refs: dict[str, np.ndarray | None] = {"points": None, "colors": None}
                self.added = False
                self.capacity = 0
                self.warned = False

            def update(self, points_xyz_m: np.ndarray, colors_rgb_u8: np.ndarray) -> tuple[float, float]:
                convert_start_s = time.perf_counter()
                points = ensure_float32_c_contiguous(points_xyz_m)
                colors = self.color_buffer.convert(colors_rgb_u8)
                self.refs["points"] = points
                self.refs["colors"] = colors
                self.pcd.point.positions = o3c.Tensor.from_numpy(points)
                self.pcd.point.colors = o3c.Tensor.from_numpy(colors)
                convert_ms = _elapsed_ms(convert_start_s, time.perf_counter())

                update_start_s = time.perf_counter()
                if points.shape[0] == 0:
                    if self.added:
                        try:
                            scene_widget.scene.remove_geometry(self.name)
                        except Exception:
                            pass
                    self.added = False
                    self.capacity = 0
                    return convert_ms, _elapsed_ms(update_start_s, time.perf_counter())

                if pointcloud_update_requires_readd(
                    geometry_added=self.added,
                    current_capacity=self.capacity,
                    point_count=int(points.shape[0]),
                ):
                    if self.added:
                        try:
                            scene_widget.scene.remove_geometry(self.name)
                        except Exception:
                            pass
                    scene_widget.scene.add_geometry(self.name, self.pcd, material)
                    self.added = True
                    self.capacity = int(points.shape[0])
                else:
                    try:
                        flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG
                        scene_widget.scene.scene.update_geometry(self.name, self.pcd, flags)
                        self.capacity = max(self.capacity, int(points.shape[0]))
                    except Exception as exc:
                        if not self.warned:
                            print(f"[WARN] update_geometry fallback for {self.name}: {type(exc).__name__}: {exc}", flush=True)
                            self.warned = True
                        try:
                            scene_widget.scene.remove_geometry(self.name)
                        except Exception:
                            pass
                        scene_widget.scene.add_geometry(self.name, self.pcd, material)
                        self.added = True
                        self.capacity = int(points.shape[0])
                return convert_ms, _elapsed_ms(update_start_s, time.perf_counter())

        controller_state = GeometryState(GEOMETRY_CONTROLLER)
        object_state = GeometryState(GEOMETRY_OBJECT)
        camera_initialized = {"value": False}
        render_post_gate = CoalescedPostGate()
        last_render_seq = {"value": -1}

        def reset_camera() -> None:
            intrinsic_matrix = np.array(
                [
                    [self.intrinsics.fx, 0.0, self.intrinsics.cx],
                    [0.0, self.intrinsics.fy, self.intrinsics.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            extrinsic = np.eye(4, dtype=np.float64)
            bounds = o3d.geometry.AxisAlignedBoundingBox([-10.0, -10.0, 0.01], [10.0, 10.0, 20.0])
            scene_widget.setup_camera(intrinsic_matrix, extrinsic, self.width, self.height, bounds)

        def render_latest() -> bool:
            packet = self.render_slot.get_latest_after(last_render_seq["value"])
            if packet is None:
                return False
            last_render_seq["value"] = packet.seq
            controller_convert_ms, controller_update_ms = controller_state.update(
                packet.controller_xyz_m,
                packet.controller_colors_rgb_u8,
            )
            object_convert_ms, object_update_ms = object_state.update(
                packet.object_xyz_m,
                packet.object_colors_rgb_u8,
            )
            if not camera_initialized["value"] and packet.point_count > 0:
                reset_camera()
                camera_initialized["value"] = True
            render_time_s = time.perf_counter()
            latency_ms = _elapsed_ms(packet.receive_perf_s, render_time_s)
            timing = replace(
                packet.timing,
                open3d_convert_ms=float(controller_convert_ms + object_convert_ms),
                open3d_update_ms=float(controller_update_ms + object_update_ms),
                receive_to_render_ms=latency_ms,
            )
            self.render_stats.record_render(render_time_s=render_time_s, latency_ms=latency_ms)
            hud_label.text = self._format_hud(packet=packet, timing=timing)
            self._maybe_log_debug(packet=packet, timing=timing, now_s=render_time_s)
            return True

        def render_latest_on_main_thread() -> None:
            try:
                if self.stop_event.is_set():
                    return
                rendered = render_latest()
                if rendered and hasattr(window, "post_redraw"):
                    try:
                        window.post_redraw()
                    except Exception:
                        pass
            finally:
                render_post_gate.mark_done()
                if not self.stop_event.is_set() and self.render_slot.latest_seq() > last_render_seq["value"]:
                    request_render_update()

        def request_render_update() -> None:
            if self.stop_event.is_set():
                return
            if not render_post_gate.try_mark_pending():
                return
            try:
                app.post_to_main_thread(window, render_latest_on_main_thread)
            except Exception:
                render_post_gate.mark_done()

        fast_exit_after_open3d = os.environ.get("QQTT_WSLG_OPEN3D_FAST_EXIT") == "1"

        def stop_and_quit_open3d() -> None:
            self.stop_event.set()
            self._request_render_update = lambda: None
            if fast_exit_after_open3d:
                self.stop()
                os._exit(0)
            try:
                app.quit()
            except Exception:
                pass

        def on_close() -> bool:
            stop_and_quit_open3d()
            return True

        window.set_on_close(on_close)
        self._request_render_update = request_render_update
        self._start_threads()

        timer: threading.Timer | None = None
        if self.args.duration_s > 0:
            timer = threading.Timer(
                float(self.args.duration_s),
                lambda: app.post_to_main_thread(window, stop_and_quit_open3d),
            )
            timer.daemon = True
            timer.start()
        try:
            app.run()
        finally:
            self._request_render_update = lambda: None
            if timer is not None:
                timer.cancel()

    def _format_hud(self, *, packet: MaskedPcdPacket, timing: PipelineTiming) -> str:
        status = "late" if timing.receive_to_render_ms > self.args.latency_target_ms else "ok"
        max_points = "uncapped" if self.args.pcd_max_points == 0 else str(self.args.pcd_max_points)
        return (
            f"capture/seg/pcd/render FPS: {self.capture_stats.fps:.1f} / {self.seg_stats.fps:.1f} / "
            f"{self.pcd_stats.fps:.1f} / {self.render_stats.render_fps:.1f}\n"
            f"latency: {timing.receive_to_render_ms:.1f} ms ({status}, target {self.args.latency_target_ms:.1f} ms)\n"
            f"points controller/object: {packet.controller_point_count} / {packet.object_point_count}  max/object: {max_points}\n"
            f"dropped capture/seg/pcd: {packet.dropped_capture_frames} / {packet.dropped_seg_frames} / "
            f"{self.render_slot.dropped_count}\n"
            f"EdgeTAM: {self.args.model_id}  mode={self.args.track_mode}  compile={self.args.compile_mode}  dtype={self.args.dtype}\n"
            f"depth: {self.args.depth_source}  color={self.args.pcd_color_mode}\n"
            f"serial/profile/fps: {self.serial}  {self.args.profile}@{self.args.fps}\n"
            f"frame: {COORDINATE_FRAME}  meters  x right / y down / z forward"
        )

    def _maybe_log_debug(self, *, packet: MaskedPcdPacket, timing: PipelineTiming, now_s: float) -> None:
        if not self.args.debug or now_s - self._last_debug_log_s < DEBUG_LOG_INTERVAL_S:
            return
        self._last_debug_log_s = now_s
        print(
            "[masked-edgetam-debug] "
            f"seq={packet.seq} "
            f"capture_fps={self.capture_stats.fps:.1f} "
            f"seg_fps={self.seg_stats.fps:.1f} "
            f"pcd_fps={self.pcd_stats.fps:.1f} "
            f"render_fps={self.render_stats.render_fps:.1f} "
            f"mask_ms={timing.mask_ms:.2f} "
            f"preprocess_ms={timing.preprocess_ms:.2f} "
            f"prompt_ms={timing.prompt_ms:.2f} "
            f"model_ms={timing.model_ms:.2f} "
            f"postprocess_ms={timing.postprocess_ms:.2f} "
            f"ffs_ms={timing.ffs_ms:.2f} "
            f"ffs_align_ms={timing.ffs_align_ms:.2f} "
            f"pcd_ms={timing.pcd_ms:.2f} "
            f"render_ms={timing.open3d_update_ms:.2f} "
            f"e2e_latency_ms={timing.receive_to_render_ms:.2f} "
            f"controller_points={packet.controller_point_count} "
            f"object_points={packet.object_point_count} "
            f"dropped_capture={packet.dropped_capture_frames} "
            f"dropped_seg={packet.dropped_seg_frames} "
            f"dropped_pcd={self.render_slot.dropped_count}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        return RealtimeMaskedEdgeTamPcdDemo(args).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
