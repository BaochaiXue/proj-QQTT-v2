"""CLI: parser, derived-mode accessors, and validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qqtt.env.camera.table_calibration import (
    TableCalibrationLoadError,
    load_table_calibration_transforms,
)
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_NONE,
    TRACKER_BACKENDS,
    normalize_tracker_backend,
)

from demo_v6_2.shape_prior import warmup as shape_prior_warmup
# config/default.yaml (via main_config) is the single source of runtime
# defaults; this parser must never carry its own copy of a config value.
from demo_v6_2.orchestration.main_config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_LOSSLESS_MAX_BACKLOG_SECONDS,
    DEFAULT_CAMERA_SOURCE_REPLAY_FPS,
    DEFAULT_DATA_PROCESS_BASE_PATH,
    DEFAULT_DEPTH_BACKEND,
    DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_PERCEPTION_DEVICE,
    DEFAULT_SHAPE_PRIOR_CACHE_ROOT,
    DEFAULT_SHAPE_PRIOR_OBJECT_PROMPT,
    DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
    DEFAULT_TRACKER_DEVICE,
    DEFAULT_VOLUME_SAMPLE_SIZE_M,
    SHAPE_PRIOR_CASE_DIR_NAME,
)
from demo_v6_2.mdp.constants import (
    CONTROLLER_COLOR_RGB,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    DEFAULT_PCD_MODE,
    DEFAULT_PROFILE,
    DEFAULT_TRACK_MODE,
    DEFAULT_TRACKER_BACKEND,
    DEPTH_BACKEND_TO_DEPTH_SOURCE,
    DEPTH_SOURCES,
    EDGE_TAM_OBJECT_LABELS,
    HAND_A_ID,
    HAND_B_ID,
    INPUT_SOURCE_FAKE_LIVE,
    INPUT_SOURCE_LIVE,
    INPUT_SOURCES,
    OBJECT_COLOR_RGB,
    OBJECT_ID,
    PCD_MODES,
    REPO_ROOT,
    TRACK_MODE_CONTROLLER_OBJECT,
    TRACK_MODE_CONTROLLER_ONLY,
    TRACK_MODE_NONE,
    TRACK_MODE_OBJECT_ONLY,
    TRACK_MODES,
)
from demo_v6_2.utils.camera import SUPPORTED_CAPTURE_FPS, parse_profile
from demo_v6_2.utils.ffs_align import validate_ffs_paths


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    """Parse RGB triplet."""
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


def _is_fake_live_input_source(input_source: str) -> bool:
    """Return whether the input source is the fake-live recorded replay."""
    return str(input_source) == INPUT_SOURCE_FAKE_LIVE


def depth_backend_label(args: argparse.Namespace) -> str:
    """Return the depth backend label."""
    label = getattr(args, "depth_backend_label", None)
    if label is not None and str(label):
        return str(label)
    return str(args.depth_source)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Single-D455 realtime HF EdgeTAM masked point-cloud demo. Captures live "
            "RealSense color plus FFS stereo depth by default, tracks controller/object "
            "or object-only with one HF EdgeTAM streaming session, and writes headless "
            "masked-PCD/tracking capture products."
        )
    )
    parser.add_argument(
        "--fps",
        choices=SUPPORTED_CAPTURE_FPS,
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Capture FPS.",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help=(
            "RealSense serial for live capture. The default picks the first "
            "sorted D400-series device, which with several cameras connected "
            "can select one without the requested RGB profile (e.g. a D405) "
            "or one that does not match table_calibrate.pkl."
        ),
    )
    parser.add_argument(
        "--warmup-rgb-preview",
        dest="warmup_rgb_preview",
        action="store_true",
        help=(
            "Show the live RGB camera-input window during warm-up (every "
            "downstream mode); it closes when warm-up finishes and "
            "immediately on failure/cancel/early exit."
        ),
    )
    parser.add_argument(
        "--no-warmup-rgb-preview",
        dest="warmup_rgb_preview",
        action="store_false",
        help="Disable the warm-up live RGB input preview window.",
    )
    parser.set_defaults(warmup_rgb_preview=True)
    parser.add_argument(
        "--input-source",
        choices=INPUT_SOURCES,
        default=INPUT_SOURCE_LIVE,
        help=(
            "Frame source. fake-live replays a raw single-camera data_collect "
            "case at camera cadence, dropping source frames to preserve "
            "recording time when replay FPS is lower."
        ),
    )
    parser.add_argument(
        "--fake-live-case",
        dest="recording_case",
        type=Path,
        default=None,
        help="Raw data_collect case; required for --input-source fake-live.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help=(
            "Replay FPS for --input-source fake-live. This is the emitted sample "
            "cadence; lower values drop source frames rather than slow motion. "
            "Use 0 to read metadata fps."
        ),
    )
    parser.add_argument(
        "--lossless-max-backlog-seconds",
        type=float,
        default=DEFAULT_CAMERA_LOSSLESS_MAX_BACKLOG_SECONDS,
        help=(
            "Maximum strict lossless input-FPS backlog window before treating "
            "the run as stalled."
        ),
    )
    parser.add_argument(
        "--lossless-input-fps",
        type=float,
        default=DEFAULT_CAMERA_SOURCE_REPLAY_FPS,
        help=(
            "Strict lossless camera/fake-live cadence used by "
            "tracker-synchronized masked PCD replay."
        ),
    )
    parser.add_argument(
        "--table-calibrate",
        type=Path,
        default=None,
        help=(
            "Required single-camera calibration pickle used to transform PCD "
            "and 3D tracker markers into table_world_z0."
        ),
    )
    parser.add_argument(
        "--depth-source",
        choices=DEPTH_SOURCES,
        default=DEPTH_BACKEND_TO_DEPTH_SOURCE[DEFAULT_DEPTH_BACKEND],
        help="Depth source. ffs streams color+IR stereo and runs local TensorRT FFS.",
    )
    parser.add_argument(
        "--depth-backend-label",
        default=None,
        help=argparse.SUPPRESS,
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
        "--color-exposure",
        type=float,
        default=None,
        help="Optional manual RealSense RGB exposure. When set, RGB auto exposure is disabled.",
    )
    parser.add_argument(
        "--color-gain",
        type=float,
        default=None,
        help="Optional manual RealSense RGB gain. When set, RGB auto exposure is disabled.",
    )
    parser.add_argument(
        "--track-mode",
        choices=TRACK_MODES,
        default=DEFAULT_TRACK_MODE,
        help="Objects tracked by EdgeTAM. Use none for capture/depth isolation profiling.",
    )
    parser.add_argument(
        "--pcd-mode",
        choices=PCD_MODES,
        default=DEFAULT_PCD_MODE,
        help="Point-cloud stage mode. Use none for EdgeTAM/depth isolation profiling.",
    )
    parser.add_argument(
        "--tracker-backend",
        choices=TRACKER_BACKENDS,
        default=DEFAULT_TRACKER_BACKEND,
        help=(
            "Point-tracker backend. Masked PCD requires tapnextpp; none is "
            "valid only when --pcd-mode none."
        ),
    )
    parser.add_argument(
        "--tracker-device",
        default=DEFAULT_TRACKER_DEVICE,
        help="Device for the point-tracker backend.",
    )
    parser.add_argument(
        "--tracker-overlay-max-points",
        type=int,
        default=512,
        help=(
            "Maximum visible tracker markers rendered per frame. "
            "0 renders all visible selected points."
        ),
    )
    parser.add_argument(
        "--phystwin-strict-output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for the strict PhysTwin tracking product. "
            "Defaults to <headless-capture-dir>/phystwin_like."
        ),
    )
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Enable the optional SAM3D shape-prior warmup request path.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable the optional SAM3D shape-prior warmup request path.",
    )
    parser.set_defaults(shape_prior_warmup=False)
    parser.add_argument(
        "--shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_true",
        help=(
            "Spawn pre-warmed one-shot upscale/generate/align workers at boot "
            "so shape-prior model loading happens before frame 0."
        ),
    )
    parser.add_argument(
        "--no-shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_false",
        help="Load shape-prior stage models only when the frame-0 request runs.",
    )
    parser.set_defaults(shape_prior_prewarm_stage_workers=True)
    parser.add_argument(
        "--shape-prior-timeout-ms",
        type=int,
        default=DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-case-root",
        type=Path,
        default=DEFAULT_DATA_PROCESS_BASE_PATH / SHAPE_PRIOR_CASE_DIR_NAME,
    )
    parser.add_argument(
        "--shape-prior-warmup-cuda-visible-devices",
        default=DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument(
        "--shape-prior-controller-name",
        default=None,
    )
    parser.add_argument(
        "--shape-prior-object",
        default=None,
        help=(
            "Canonical-mesh cache identity (instance + asset version). Absent "
            "disables the cache. Never pass 'none'/'null'; omit the flag instead."
        ),
    )
    parser.add_argument(
        "--shape-prior-object-prompt",
        default=DEFAULT_SHAPE_PRIOR_OBJECT_PROMPT,
        help="SAM3.1 semantic label for the object in view.",
    )
    parser.add_argument(
        "--shape-prior-cache-root",
        type=Path,
        default=DEFAULT_SHAPE_PRIOR_CACHE_ROOT,
        help="Persistent cache root for canonical meshes (not under run output).",
    )
    parser.add_argument("--shape-prior-sam3d-root", type=Path, default=None)
    parser.add_argument("--shape-prior-config", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-skip-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_true",
    )
    parser.add_argument(
        "--shape-prior-render-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_false",
    )
    parser.set_defaults(shape_prior_skip_route_visualizations=True)
    parser.add_argument(
        "--device",
        default=DEFAULT_PERCEPTION_DEVICE,
        help="Inference device, usually cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_INFERENCE_DTYPE,
        help="Inference dtype.",
    )
    parser.add_argument(
        "--edgetam-mask-logit-threshold",
        type=float,
        default=DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
        help=(
            "Logit threshold used to binarize EdgeTAM masks. "
            "Lower values make masks more permissive."
        ),
    )
    parser.add_argument(
        "--pcd-color-mode",
        choices=("rgb", "class"),
        default="rgb",
        help=(
            "Point-cloud colors. rgb uses the live color frame; class uses "
            "fixed controller/object colors."
        ),
    )
    parser.add_argument(
        "--headless-capture-dir",
        type=Path,
        default=None,
        help=(
            "Save canonical processed masks, color-aligned depth, PCD, and "
            "TAPNext++ trajectory artifacts here."
        ),
    )
    parser.add_argument(
        "--headless-prepared-only",
        action="store_true",
        help=(
            "For strict PhysTwin chunk preprocessing, save prepared_phystwin "
            "frames and frames.jsonl without legacy per-frame artifacts."
        ),
    )
    parser.add_argument(
        "--write-input-rgb-timeline",
        action="store_true",
        help=(
            "Write input_rgb/*.png and input_frames.jsonl for Demo v6.2 "
            "realtime side-by-side viewing."
        ),
    )
    parser.add_argument(
        "--controller-color",
        type=_parse_rgb_triplet,
        default=CONTROLLER_COLOR_RGB,
        help="Controller RGB color.",
    )
    parser.add_argument(
        "--object-color",
        type=_parse_rgb_triplet,
        default=OBJECT_COLOR_RGB,
        help="Object RGB color.",
    )
    return parser


def headless_capture_enabled(args: argparse.Namespace) -> bool:
    """Return whether headless capture is enabled."""
    return args.headless_capture_dir is not None


def validate_and_normalize_args(args: argparse.Namespace) -> None:
    """Validate args."""
    parse_profile(DEFAULT_PROFILE)
    if args.input_source not in INPUT_SOURCES:
        raise ValueError(f"--input-source must be one of {', '.join(INPUT_SOURCES)}")
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if float(args.replay_fps) < 0.0:
        raise ValueError("--replay-fps must be >= 0")
    if float(args.lossless_max_backlog_seconds) <= 0.0:
        raise ValueError("--lossless-max-backlog-seconds must be positive")
    if float(args.lossless_input_fps) <= 0.0:
        raise ValueError("--lossless-input-fps must be positive")
    if args.table_calibrate is not None:
        table_path = Path(args.table_calibrate).expanduser()
        if not table_path.is_absolute():
            table_path = REPO_ROOT / table_path
        table_path = table_path.resolve(strict=False)
        try:
            load_table_calibration_transforms(table_path)
        except TableCalibrationLoadError as exc:
            message = str(exc)
            if "Missing table calibration file" in message:
                raise ValueError(message) from exc
            raise ValueError(f"Invalid table calibration file: {message}") from exc
        args.table_calibrate = table_path
    if _is_fake_live_input_source(str(args.input_source)):
        if args.recording_case is None:
            raise ValueError("--input-source fake-live requires --fake-live-case")
    elif args.recording_case is not None:
        raise ValueError("--fake-live-case requires --input-source fake-live")
    if (
        bool(args.shape_prior_warmup)
        and not str(args.shape_prior_controller_name or "").strip()
    ):
        raise ValueError(
            "--shape-prior-controller-name is required when --shape-prior-warmup "
            "is enabled"
        )
    object_prompt = str(args.shape_prior_object_prompt or "").strip()
    if bool(args.shape_prior_warmup) or args.track_mode in {
        TRACK_MODE_CONTROLLER_OBJECT,
        TRACK_MODE_OBJECT_ONLY,
    }:
        if not object_prompt:
            raise ValueError(
                "--shape-prior-object-prompt is required when shape-prior "
                "warmup or object tracking is enabled"
            )
    args.shape_prior_object_prompt = object_prompt
    if not np.isfinite(float(args.edgetam_mask_logit_threshold)):
        raise ValueError("--edgetam-mask-logit-threshold must be finite")
    if int(args.shape_prior_timeout_ms) <= 0:
        raise ValueError("--shape-prior-timeout-ms must be positive")
    if args.table_calibrate is None:
        raise ValueError("formal runtime requires --table-calibrate")
    if args.depth_source == "none" and args.pcd_mode == "masked":
        raise ValueError("--depth-source none requires --pcd-mode none")
    if headless_capture_enabled(args):
        if args.depth_source not in {"ffs", "realsense"}:
            raise ValueError(
                "--headless-capture-dir requires --depth-source ffs or realsense"
            )
    args.tracker_backend = normalize_tracker_backend(str(args.tracker_backend))
    if args.pcd_mode == "masked" and not tracker_enabled(args):
        raise ValueError("--pcd-mode masked requires --tracker-backend tapnextpp")
    if args.pcd_mode == "masked" and args.track_mode == TRACK_MODE_NONE:
        raise ValueError("--pcd-mode masked requires an enabled --track-mode")
    if args.headless_capture_dir is None:
        raise ValueError("formal runtime requires --headless-capture-dir")
    if args.phystwin_strict_output_dir is None:
        args.phystwin_strict_output_dir = (
            Path(args.headless_capture_dir) / "phystwin_like"
        )
    if bool(args.shape_prior_warmup):
        from demo_v6_2.shape_prior.mesh_cache import (  # noqa: PLC0415
            ShapePriorMeshCacheError,
            normalize_object_id,
            validate_cache_root,
        )

        try:
            args.shape_prior_object = normalize_object_id(args.shape_prior_object)
        except ShapePriorMeshCacheError as exc:
            raise ValueError(str(exc)) from exc
        try:
            args.shape_prior_cache_root = validate_cache_root(
                args.shape_prior_cache_root,
                forbidden_root=Path(args.shape_prior_case_root).parent,
            )
        except ShapePriorMeshCacheError as exc:
            raise ValueError(str(exc)) from exc
    if (
        getattr(args, "color_exposure", None) is not None
        and float(args.color_exposure) <= 0.0
    ):
        raise ValueError("--color-exposure must be positive")
    if getattr(args, "color_gain", None) is not None and float(args.color_gain) < 0.0:
        raise ValueError("--color-gain must be >= 0")
    if tracker_enabled(args):
        if args.depth_source == "none":
            raise ValueError(
                "--tracker-backend tapnextpp requires RGB-D depth for 3D marker lift"
            )
    if args.depth_source == "ffs":
        validate_ffs_paths(
            ffs_repo=Path(args.ffs_repo), model_dir=Path(args.ffs_trt_model_dir)
        )


def active_object_id_labels(args: argparse.Namespace) -> dict[int, str]:
    """Return the active object id labels."""
    track_mode = str(args.track_mode)
    if track_mode == TRACK_MODE_NONE:
        return {}
    if track_mode == TRACK_MODE_OBJECT_ONLY:
        return {OBJECT_ID: EDGE_TAM_OBJECT_LABELS[OBJECT_ID]}
    if track_mode == TRACK_MODE_CONTROLLER_ONLY:
        return {
            HAND_A_ID: EDGE_TAM_OBJECT_LABELS[HAND_A_ID],
            HAND_B_ID: EDGE_TAM_OBJECT_LABELS[HAND_B_ID],
        }
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
        return dict(EDGE_TAM_OBJECT_LABELS)
    raise ValueError(f"unsupported track mode: {track_mode}")


def active_object_ids(args: argparse.Namespace) -> list[int]:
    """Return the active object ids."""
    return list(active_object_id_labels(args).keys())


def tracker_enabled(args: argparse.Namespace) -> bool:
    """Return whether tracker is enabled."""
    return (
        normalize_tracker_backend(str(args.tracker_backend))
        != TRACKER_BACKEND_NONE
    )


@dataclass(frozen=True)
class RunMode:
    """Immutable run-mode snapshot derived once from validated args.

    Built right after ``validate_and_normalize_args`` and shared by every worker thread, so
    run modes are decided exactly once per process instead of being re-derived
    from the argparse Namespace at every call site.
    """

    tracker_enabled: bool
    lossless_enabled: bool
    lossless_input_fps: float
    controller_tracking_enabled: bool
    object_tracking_enabled: bool
    fake_live_input: bool  # recorded fake-live replay (the only replay source)
    depth_backend_label: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunMode:
        """Derive the snapshot from a validated argparse Namespace."""
        track_mode = str(args.track_mode)
        is_tracker_enabled = tracker_enabled(args)
        return cls(
            tracker_enabled=is_tracker_enabled,
            lossless_enabled=bool(is_tracker_enabled and args.pcd_mode == "masked"),
            lossless_input_fps=float(args.lossless_input_fps),
            controller_tracking_enabled=track_mode
            in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_CONTROLLER_ONLY},
            object_tracking_enabled=track_mode
            in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_OBJECT_ONLY},
            fake_live_input=_is_fake_live_input_source(str(args.input_source)),
            depth_backend_label=depth_backend_label(args),
        )


__all__ = [
    "_is_fake_live_input_source",
    "depth_backend_label",
    "build_parser",
    "headless_capture_enabled",
    "validate_and_normalize_args",
    "active_object_id_labels",
    "active_object_ids",
    "tracker_enabled",
    "RunMode",
]
