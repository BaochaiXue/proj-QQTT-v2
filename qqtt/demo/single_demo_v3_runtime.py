from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from qqtt.demo import realtime_masked_edgetam_pcd as masked_pcd
from qqtt.demo import realtime_single_camera_pointcloud as single_pcd
from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    TableCalibrationLoadError,
    load_table_calibration_transforms,
)


DEMO_VERSION_3 = "3"
DEMO_VERSION_3_1 = "3.1"
DEMO_VERSION_3_2 = "3.2"
DEMO_VERSION_3_3 = "3.3"
DEMO_VERSIONS = (DEMO_VERSION_3, DEMO_VERSION_3_1, DEMO_VERSION_3_2, DEMO_VERSION_3_3)

DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_FFS = "ffs"

INPUT_SOURCE_LIVE = "live"
INPUT_SOURCE_FAKE_LIVE = "fake-live"
INPUT_SOURCE_RECORDING = "recording"
INPUT_SOURCES = (INPUT_SOURCE_LIVE, INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_RECORDING)

MODE_EXP = "exp"
MODE_DEMO = "demo"
MODES = (MODE_EXP, MODE_DEMO)

TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_NONE = "none"
TRACK_MODES = (TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_NONE)
DEMO_VISUAL_MODE_PCD = "pcd"
DEMO_VISUAL_MODE_TRACKING = "tracking"
DEMO_VISUAL_MODES = (DEMO_VISUAL_MODE_PCD, DEMO_VISUAL_MODE_TRACKING)
DEFAULT_DEMO_VISUAL_MODE = DEMO_VISUAL_MODE_TRACKING

DEFAULT_OBJECT_PROMPT = "stuffed animal"
DEFAULT_EXP_CONTROLLER_PROMPT = "towel"
DEFAULT_DEMO_CONTROLLER_PROMPT = "human hand"
DEFAULT_DEMO_CONTROLLER_LABEL = "hand"
DEFAULT_MODE = MODE_EXP
DEFAULT_TRACKER_BACKEND = masked_pcd.TRACKER_BACKEND_TAPNEXTPP
DEFAULT_TRACKER_DEVICE = "cuda:1"
DEFAULT_TABLE_CALIBRATE_PATH = Path("table_calibrate.pkl")
DEFAULT_FAKE_LIVE_CASE = Path("data_collect/sloth_both_eval_2min_e45_g35_20260614_155543")
DEFAULT_DEMO32_FAKE_LIVE_CASE = Path("data_collect/sloth_both_eval_3min_e70_g60_20260621_202627")
DEFAULT_FAKE_LIVE_REPLAY_FPS = 5.0
DEFAULT_RECORDING_FPS = 30.0
FFS_SURFACE_FILTER_RADIUS_M = 0.015
FFS_SURFACE_FILTER_NB_POINTS = 8
FFS_SURFACE_COMPONENT_VOXEL_SIZE_M = 0.015
FFS_SURFACE_FILTER_EVERY_N = 1
FFS_SURFACE_FILTER_MAX_AGE_FRAMES = 1
FFS_SURFACE_OBJECT_MASK_ERODE_PIXELS = 0
FFS_SURFACE_CONTROLLER_MASK_ERODE_PIXELS = 0
HEADLESS_CAPTURE_SAVED_PCD_SOURCE = "none_filtered"

DEFAULT_OUTPUT_ROOTS = {
    DEMO_VERSION_3: Path("result/single_demo_v3_realsense_masked_pcd"),
    DEMO_VERSION_3_1: Path("result/single_demo_v3_1_realsense_masked_pcd"),
    DEMO_VERSION_3_2: Path("result/single_demo_v3_2_ffs_masked_pcd"),
    DEMO_VERSION_3_3: Path("result/single_demo_v3_3_ffs_masked_pcd"),
}

DEFAULT_DEPTH_SOURCES = {
    DEMO_VERSION_3: DEPTH_SOURCE_REALSENSE,
    DEMO_VERSION_3_1: DEPTH_SOURCE_REALSENSE,
    DEMO_VERSION_3_2: DEPTH_SOURCE_FFS,
    DEMO_VERSION_3_3: DEPTH_SOURCE_FFS,
}

VERSION_LABELS = {
    DEMO_VERSION_3: "Single Demo 3",
    DEMO_VERSION_3_1: "Single Demo 3.1",
    DEMO_VERSION_3_2: "Single Demo 3.2",
    DEMO_VERSION_3_3: "Single Demo 3.3",
}


def default_fake_live_case_for_version(version: str) -> Path:
    if normalize_demo_version(version) == DEMO_VERSION_3_2:
        return DEFAULT_DEMO32_FAKE_LIVE_CASE
    return DEFAULT_FAKE_LIVE_CASE

ConnectedSerialsProvider = Callable[[], Sequence[str]]
REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_demo_version(demo_version: str) -> str:
    normalized = str(demo_version).strip().lower().removeprefix("demo").removeprefix("v")
    if normalized in {"3_1", "3-1"}:
        normalized = DEMO_VERSION_3_1
    elif normalized in {"3_2", "3-2"}:
        normalized = DEMO_VERSION_3_2
    elif normalized in {"3_3", "3-3"}:
        normalized = DEMO_VERSION_3_3
    if normalized not in DEMO_VERSIONS:
        raise ValueError(f"Unsupported single demo version: {demo_version}")
    return normalized


def _explicit_cli_options(argv: Sequence[str] | None) -> set[str]:
    if argv is None:
        argv = sys.argv[1:]
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


def _mode_prompts(mode: str) -> dict[str, str]:
    normalized = str(mode).strip().lower()
    if normalized == MODE_EXP:
        return {
            "semantic_mode": MODE_EXP,
            "controller_prompt": DEFAULT_EXP_CONTROLLER_PROMPT,
            "controller_label": DEFAULT_EXP_CONTROLLER_PROMPT,
        }
    if normalized == MODE_DEMO:
        return {
            "semantic_mode": MODE_DEMO,
            "controller_prompt": DEFAULT_DEMO_CONTROLLER_PROMPT,
            "controller_label": DEFAULT_DEMO_CONTROLLER_LABEL,
        }
    raise ValueError(f"Unsupported single demo mode: {mode}")


def _effective_object_pcd_mask_erode_pixels(args: argparse.Namespace) -> int:
    value = getattr(args, "object_pcd_mask_erode_pixels", None)
    if value is None:
        value = getattr(args, "pcd_mask_erode_pixels", masked_pcd.DEFAULT_PCD_MASK_ERODE_PIXELS)
    return int(value)


def _effective_controller_pcd_mask_erode_pixels(args: argparse.Namespace) -> int:
    value = getattr(args, "controller_pcd_mask_erode_pixels", None)
    if value is None:
        value = getattr(args, "pcd_mask_erode_pixels", masked_pcd.DEFAULT_PCD_MASK_ERODE_PIXELS)
    return int(value)


def _edgetam_tracking_identities(args: argparse.Namespace) -> list[str]:
    track_mode = str(args.track_mode)
    two_hands = str(args.controller_instance_mode) == masked_pcd.CONTROLLER_INSTANCE_MODE_TWO_HANDS
    if track_mode == TRACK_MODE_NONE:
        return []
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
        return ["hand_a", "object", "hand_b"] if two_hands else ["controller", "object"]
    if track_mode == masked_pcd.TRACK_MODE_OBJECT_ONLY:
        return ["object"]
    if track_mode == masked_pcd.TRACK_MODE_CONTROLLER_ONLY:
        return ["hand_a", "hand_b"] if two_hands else ["controller"]
    return []


def _is_replay_input_source(input_source: str) -> bool:
    return str(input_source) in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_RECORDING}


def _supports_headless_capture(version: str) -> bool:
    return normalize_demo_version(version) in {DEMO_VERSION_3_2, DEMO_VERSION_3_3}


def _supports_filtered_visual_modes(version: str) -> bool:
    return normalize_demo_version(version) in {DEMO_VERSION_3_2, DEMO_VERSION_3_3}


def _render_mode_requests_visual_policy(args: argparse.Namespace) -> bool:
    return str(getattr(args, "render_mode", "pointcloud")) in {"pointcloud", "panel"}


def _requires_table_world_default(version: str) -> bool:
    return normalize_demo_version(version) in {DEMO_VERSION_3_1, DEMO_VERSION_3_2, DEMO_VERSION_3_3}


def _filtered_visual_mode_requested(args: argparse.Namespace, version: str | None = None) -> bool:
    resolved_version = normalize_demo_version(version or getattr(args, "single_demo_version", DEMO_VERSION_3))
    return bool(
        _supports_filtered_visual_modes(resolved_version)
        and str(getattr(args, "demo_visual_mode", DEFAULT_DEMO_VISUAL_MODE)) in DEMO_VISUAL_MODES
        and _render_mode_requests_visual_policy(args)
    )


def _demo_visual_mode_policy_requested(args: argparse.Namespace, version: str | None = None) -> bool:
    resolved_version = normalize_demo_version(version or getattr(args, "single_demo_version", DEMO_VERSION_3))
    return bool(
        _supports_filtered_visual_modes(resolved_version)
        and str(getattr(args, "demo_visual_mode", DEFAULT_DEMO_VISUAL_MODE)) in DEMO_VISUAL_MODES
        and (_render_mode_requests_visual_policy(args) or _headless_capture_requested(args, resolved_version))
    )


def _table_z_filter_visual_default_requested(args: argparse.Namespace, version: str | None = None) -> bool:
    resolved_version = normalize_demo_version(version or getattr(args, "single_demo_version", DEMO_VERSION_3))
    return bool(
        _requires_table_world_default(resolved_version)
        and str(getattr(args, "demo_visual_mode", DEFAULT_DEMO_VISUAL_MODE)) in DEMO_VISUAL_MODES
        and (_render_mode_requests_visual_policy(args) or _headless_capture_requested(args, resolved_version))
    )


def _visual_mode_required_filter(args: argparse.Namespace) -> str:
    return masked_pcd.PCD_FILTER_NONE


def _visual_mode_required_preset(args: argparse.Namespace) -> str:
    return masked_pcd.PCD_FILTER_PRESET_ORIGINAL


def _effective_pcd_filter_preset(args: argparse.Namespace) -> str | None:
    preset = getattr(args, "pcd_filter_preset", None)
    if preset is not None:
        return str(preset)
    if not bool(getattr(args, "enable_pcd_filter", False)):
        return None
    object_filter = str(getattr(args, "object_filter", ""))
    controller_filter = str(getattr(args, "controller_filter", ""))
    if object_filter != controller_filter:
        return None
    if object_filter == masked_pcd.PCD_FILTER_NONE:
        return masked_pcd.PCD_FILTER_PRESET_ORIGINAL
    if object_filter == masked_pcd.PCD_FILTER_PT_FILTER:
        return masked_pcd.PCD_FILTER_PRESET_PT
    if object_filter == masked_pcd.PCD_FILTER_ENHANCED_PT:
        return masked_pcd.PCD_FILTER_PRESET_ENHANCED_PT
    return None


def _headless_capture_requested(args: argparse.Namespace, version: str | None = None) -> bool:
    resolved_version = normalize_demo_version(version or getattr(args, "single_demo_version", DEMO_VERSION_3))
    return bool(
        _supports_headless_capture(resolved_version)
        and str(args.input_source) == INPUT_SOURCE_FAKE_LIVE
        and str(args.render_mode) == "none"
    )


def _interactive_tracking_render_requested(args: argparse.Namespace) -> bool:
    return str(getattr(args, "render_mode", "")) in {"pointcloud", "panel"}


def _default_headless_capture_dir(args: argparse.Namespace, version: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(getattr(args, "output_root", DEFAULT_OUTPUT_ROOTS[normalize_demo_version(version)]))
    return output_root / f"headless_capture_{stamp}"


def _headless_capture_saved_pcd_source(args: argparse.Namespace) -> str:
    return masked_pcd.headless_capture_saved_pcd_source(args)


def _get_connected_realsense_serials() -> list[str]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required to discover connected RealSense cameras.") from exc
    ctx = rs.context()
    serials: list[str] = []
    for device in ctx.query_devices():
        try:
            serials.append(str(device.get_info(rs.camera_info.serial_number)))
        except Exception:
            continue
    return sorted(serials)


def build_arg_parser(*, demo_version: str = DEMO_VERSION_3) -> argparse.ArgumentParser:
    version = normalize_demo_version(demo_version)
    label = VERSION_LABELS[version]
    depth_source = DEFAULT_DEPTH_SOURCES[version]
    parser = argparse.ArgumentParser(
        description=(
            f"{label}: one RealSense camera, one masked point-cloud stream, "
            f"and {depth_source} depth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(single_demo_version=version)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved single-camera contract and exit.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Run duration. 0 means until closed.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-json-output", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOTS[version])
    parser.add_argument("--serial", default=None, help="Single RealSense serial to open. Defaults to first detected serial.")
    parser.add_argument("--profile", choices=single_pcd.SUPPORTED_PROFILES, default=single_pcd.DEFAULT_PROFILE)
    parser.add_argument("--fps", type=int, choices=single_pcd.SUPPORTED_CAPTURE_FPS, default=single_pcd.DEFAULT_FPS)
    parser.add_argument(
        "--input-source",
        choices=INPUT_SOURCES,
        default=INPUT_SOURCE_LIVE,
        help=(
            "Frame source. fake-live replays a raw single-camera data_collect case at camera cadence "
            "and uses demo mode; fake-live drops source frames to preserve recording time when replay FPS "
            "is lower. recording is kept as a compatibility alias for explicit replay cases."
        ),
    )
    parser.add_argument(
        "--recording-case",
        type=Path,
        default=None,
        help="Raw data_collect case folder for --input-source recording or fake-live.",
    )
    parser.add_argument(
        "--fake-live-case",
        dest="recording_case",
        type=Path,
        default=None,
        help=f"Alias for --recording-case. fake-live defaults to {default_fake_live_case_for_version(version)}.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help=(
            "Replay FPS for --input-source recording or fake-live. Omitted fake-live replay defaults to "
            f"{DEFAULT_FAKE_LIVE_REPLAY_FPS:g} fps; fake-live drops source frames to preserve recording time "
            "instead of slow motion. use 0 to read metadata fps."
        ),
    )
    parser.add_argument(
        "--table-calibrate",
        type=Path,
        default=None,
        help="Optional table Z=0 calibration file to validate and expose in demo contract.",
    )
    parser.add_argument(
        "--enable-table-z-filter",
        action="store_true",
        help="Enable table-world Z filter forwarded to the masked PCD delegate.",
    )
    parser.add_argument(
        "--disable-table-z-filter",
        action="store_true",
        help="Disable the table-world Z filter when a demo visual mode enables it by default.",
    )
    parser.add_argument(
        "--table-z-filter-threshold-m",
        type=float,
        default=masked_pcd.DEFAULT_TABLE_Z_FILTER_THRESHOLD_M,
        help="World-Z clearance threshold above table_z for --enable-table-z-filter.",
    )
    parser.add_argument(
        "--table-z-above-direction",
        choices=masked_pcd.TABLE_Z_ABOVE_DIRECTIONS,
        default=masked_pcd.DEFAULT_TABLE_Z_ABOVE_DIRECTION,
        help="Which table-world Z direction points away from the tabletop into the workspace.",
    )
    parser.add_argument(
        "--table-z-filter-classes",
        choices=masked_pcd.TABLE_Z_FILTER_CLASSES,
        default=masked_pcd.TABLE_Z_FILTER_CLASS_BOTH,
        help="Semantic classes affected by --enable-table-z-filter.",
    )
    if depth_source == DEPTH_SOURCE_FFS:
        parser.add_argument("--ffs-repo", type=Path, default=single_pcd.DEFAULT_FFS_REPO)
        parser.add_argument("--ffs-trt-model-dir", type=Path, default=single_pcd.DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR)
        parser.add_argument("--ffs-trt-root", type=Path, default=None)
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("--object-prompt", default=DEFAULT_OBJECT_PROMPT)
    parser.add_argument(
        "--controller-prompt",
        default=None,
        help="Override the controller prompt. Defaults to towel in exp mode and human hand in demo mode.",
    )
    parser.add_argument(
        "--controller-instance-mode",
        choices=masked_pcd.CONTROLLER_INSTANCE_MODES,
        default=masked_pcd.CONTROLLER_INSTANCE_MODE_SINGLE,
        help="Use two-hands to propagate hand_a and hand_b as separate EdgeTAM controller identities.",
    )
    parser.add_argument(
        "--track-mode",
        choices=TRACK_MODES,
        default=TRACK_MODE_CONTROLLER_OBJECT,
        help="Single-camera EdgeTAM mask mode.",
    )
    parser.add_argument(
        "--render-mode",
        choices=masked_pcd.RENDER_MODES,
        default=masked_pcd.DEFAULT_RENDER_MODE,
        help="Render mode for the single-camera masked point-cloud delegate.",
    )
    parser.add_argument(
        "--panel-layout",
        choices=masked_pcd.PANEL_LAYOUTS,
        default=masked_pcd.PANEL_LAYOUT_SIDE_BY_SIDE,
        help="Runtime panel layout forwarded to the masked PCD delegate.",
    )
    parser.add_argument(
        "--panel-video-output",
        type=Path,
        default=None,
        help="Optional MP4 output path for --render-mode panel.",
    )
    parser.add_argument(
        "--tracking-background-mask",
        choices=("target-union", "rgb"),
        default="target-union",
        help="Tracking overlay background for side-by-side panel and offline render parity.",
    )
    parser.add_argument(
        "--demo-visual-mode",
        choices=DEMO_VISUAL_MODES,
        default=DEFAULT_DEMO_VISUAL_MODE,
        help="Demo 3.2/3.3 visual presentation mode: filtered PCD only, or filtered PCD plus rainbow query tracking.",
    )
    parser.add_argument(
        "--headless-capture-dir",
        type=Path,
        default=None,
        help=(
            "Directory for Demo 3.2/3.3 fake-live headless filtered PCD artifacts. "
            "Defaults to output-root/headless_capture_<timestamp> when --render-mode none is used. "
            "The saved PCD uses the same default 0 mm table-Z filter as visual PCD/tracking modes."
        ),
    )
    parser.add_argument(
        "--view-mode",
        choices=masked_pcd.VIEW_MODES,
        default=masked_pcd.DEFAULT_VIEW_MODE,
        help="Initial Open3D view. orbit starts from a third-person view; camera uses RealSense color intrinsics.",
    )
    parser.add_argument(
        "--tracker-backend",
        choices=masked_pcd.TRACKER_BACKENDS,
        default=DEFAULT_TRACKER_BACKEND,
        help="Point-tracker overlay backend. Demo 3.x fake-live replay uses TAPNext++ by default.",
    )
    parser.add_argument("--tracker-device", default=DEFAULT_TRACKER_DEVICE)
    parser.add_argument("--tracker-query-count", type=int, default=masked_pcd.DEFAULT_TRACKER_QUERY_COUNT)
    parser.add_argument("--tracker-seed", type=int, default=masked_pcd.DEFAULT_TRACKER_SEED)
    parser.add_argument(
        "--tracker-display-scope",
        choices=masked_pcd.TRACKER_DISPLAY_SCOPES,
        default=masked_pcd.DEFAULT_TRACKER_DISPLAY_SCOPE,
    )
    parser.add_argument("--tracker-overlay-max-points", type=int, default=512)
    parser.add_argument("--tracker-marker-point-size", type=float, default=masked_pcd.DEFAULT_TRACKER_MARKER_POINT_SIZE)
    parser.add_argument(
        "--tracker-retire-filtered-markers",
        dest="tracker_retire_filtered_markers",
        action="store_true",
        help="Opt in to permanently hiding any query marker after it fails the active PCD residual/table-Z gate.",
    )
    parser.add_argument(
        "--no-tracker-retire-filtered-markers",
        dest="tracker_retire_filtered_markers",
        action="store_false",
        help="Use the default per-frame marker gate; filtered markers may reappear later.",
    )
    parser.set_defaults(tracker_retire_filtered_markers=False)
    parser.add_argument("--tapnet-repo-dir", type=Path, default=masked_pcd.DEFAULT_TAPNET_REPO_DIR)
    parser.add_argument("--tapnextpp-checkpoint", type=Path, default=masked_pcd.DEFAULT_TAPNEXTPP_CHECKPOINT)
    parser.add_argument("--tapnextpp-image-size", default="256,256")
    parser.add_argument("--tapnextpp-autocast-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--tapnextpp-compile", action="store_true")
    parser.add_argument("--no-tapnextpp-fast-postprocess", dest="tapnextpp_fast_postprocess", action="store_false")
    parser.set_defaults(tapnextpp_fast_postprocess=True)
    parser.add_argument(
        "--edgetam-live-session-keep-frames",
        type=int,
        default=masked_pcd.DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
        help="Recent streamed EdgeTAM frames/outputs retained in live session state; 0 disables pruning.",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=1.5)
    parser.add_argument("--pcd-max-points", type=int, default=60000)
    parser.add_argument("--pcd-stride", type=int, default=1)
    parser.add_argument("--pcd-color-mode", choices=("rgb", "class"), default="rgb")
    parser.add_argument("--pcd-mask-erode-pixels", type=int, default=masked_pcd.DEFAULT_PCD_MASK_ERODE_PIXELS)
    parser.add_argument(
        "--object-pcd-mask-erode-pixels",
        type=int,
        default=masked_pcd.DEFAULT_OBJECT_PCD_MASK_ERODE_PIXELS,
    )
    parser.add_argument(
        "--controller-pcd-mask-erode-pixels",
        type=int,
        default=masked_pcd.DEFAULT_CONTROLLER_PCD_MASK_ERODE_PIXELS,
    )
    parser.add_argument(
        "--render-max-points-per-layer",
        type=int,
        default=masked_pcd.DEFAULT_RENDER_MAX_POINTS_PER_LAYER,
    )
    parser.add_argument("--enable-pcd-filter", action="store_true")
    parser.add_argument(
        "--pcd-filter-mode",
        choices=masked_pcd.PCD_FILTER_MODES,
        default="async",
        help="Point-cloud filter scheduling mode forwarded to the masked PCD delegate.",
    )
    parser.add_argument(
        "--pcd-filter-preset",
        choices=masked_pcd.PCD_FILTER_PRESETS,
        default=None,
        help=(
            "High-level PCD filter preset for Demo 3.x. When set, it controls both object/controller "
            "PCD filtering and TAPNext++ initial query sampling."
        ),
    )
    parser.add_argument("--object-filter", choices=masked_pcd.PCD_FILTERS, default=masked_pcd.DEFAULT_OBJECT_FILTER)
    parser.add_argument(
        "--controller-filter",
        choices=masked_pcd.PCD_FILTERS,
        default=masked_pcd.DEFAULT_CONTROLLER_FILTER,
    )
    parser.add_argument("--object-filter-cap", type=int, default=20_000)
    parser.add_argument("--controller-filter-cap", type=int, default=20_000)
    parser.add_argument(
        "--object-filter-keep-components",
        type=int,
        default=masked_pcd.DEFAULT_OBJECT_FILTER_KEEP_COMPONENTS,
        help="Enhanced PCD component count for object filtering.",
    )
    parser.add_argument(
        "--controller-filter-keep-components",
        type=int,
        default=masked_pcd.DEFAULT_CONTROLLER_FILTER_KEEP_COMPONENTS,
        help="Enhanced PCD component count for controller filtering; default keeps two hand components.",
    )
    parser.add_argument("--object-filter-voxel-m", type=float, default=0.004)
    parser.add_argument("--controller-filter-voxel-m", type=float, default=0.003)
    parser.add_argument("--filter-every-n", type=int, default=3)
    parser.add_argument(
        "--filter-max-age-frames",
        type=int,
        default=masked_pcd.DEFAULT_FILTER_MAX_AGE_FRAMES,
        help="Maximum async filtered-output age in frames before rendering raw current PCD instead.",
    )
    parser.add_argument("--filter-budget-ms", type=float, default=12.0)
    parser.add_argument("--filter-min-cap", type=int, default=5_000)
    parser.add_argument("--voxel-density-min-points", type=int, default=2)
    parser.add_argument("--filter-radius-m", type=float, default=masked_pcd.DEFAULT_FILTER_RADIUS_M)
    parser.add_argument("--filter-nb-points", type=int, default=masked_pcd.DEFAULT_FILTER_NB_POINTS)
    parser.add_argument(
        "--enhanced-component-voxel-size-m",
        type=float,
        default=masked_pcd.DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    )
    parser.add_argument(
        "--enhanced-keep-near-main-gap-m",
        type=float,
        default=masked_pcd.DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    return parser


def apply_preset_defaults(
    args: argparse.Namespace,
    *,
    explicit_options: set[str] | None = None,
) -> argparse.Namespace:
    explicit = set(explicit_options or set())
    args._explicit_options = explicit
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    if "--output-root" not in explicit:
        args.output_root = DEFAULT_OUTPUT_ROOTS[version]
    args.depth_source = DEFAULT_DEPTH_SOURCES[version]
    if (
        _requires_table_world_default(version)
        and "--table-calibrate" not in explicit
        and args.table_calibrate is None
    ):
        args.table_calibrate = DEFAULT_TABLE_CALIBRATE_PATH
    if str(args.input_source) == INPUT_SOURCE_FAKE_LIVE:
        args.mode = MODE_DEMO
    if "--controller-prompt" not in explicit or args.controller_prompt is None:
        args.controller_prompt = _mode_prompts(str(args.mode))["controller_prompt"]
    if "--controller-instance-mode" not in explicit:
        prompt = str(args.controller_prompt).strip().lower()
        if str(args.mode) == MODE_DEMO and "hand" in prompt:
            args.controller_instance_mode = masked_pcd.CONTROLLER_INSTANCE_MODE_TWO_HANDS
        else:
            args.controller_instance_mode = masked_pcd.CONTROLLER_INSTANCE_MODE_SINGLE
    headless_capture = _headless_capture_requested(args, version)
    filtered_visual = _demo_visual_mode_policy_requested(args, version)
    if headless_capture:
        if "--track-mode" not in explicit:
            args.track_mode = TRACK_MODE_CONTROLLER_OBJECT
        if "--tracker-backend" not in explicit:
            args.tracker_backend = masked_pcd.TRACKER_BACKEND_TAPNEXTPP
        if "--tracker-overlay-max-points" not in explicit:
            args.tracker_overlay_max_points = 0
        if "--enable-pcd-filter" not in explicit:
            args.enable_pcd_filter = True
        if "--pcd-filter-mode" not in explicit:
            args.pcd_filter_mode = "sync"
        if "--pcd-filter-preset" not in explicit:
            args.pcd_filter_preset = masked_pcd.PCD_FILTER_PRESET_ORIGINAL
        if "--headless-capture-dir" not in explicit and args.headless_capture_dir is None:
            args.headless_capture_dir = _default_headless_capture_dir(args, version)
    if filtered_visual:
        visual_preset = _visual_mode_required_preset(args)
        if "--track-mode" not in explicit:
            args.track_mode = TRACK_MODE_CONTROLLER_OBJECT
        if "--enable-pcd-filter" not in explicit:
            args.enable_pcd_filter = True
        if "--pcd-filter-mode" not in explicit:
            args.pcd_filter_mode = "sync"
        if "--pcd-filter-preset" not in explicit:
            args.pcd_filter_preset = visual_preset
        if "--pcd-color-mode" not in explicit:
            args.pcd_color_mode = "rgb"
        if "--tracker-backend" not in explicit:
            args.tracker_backend = masked_pcd.TRACKER_BACKEND_TAPNEXTPP
        if "--tracker-overlay-max-points" not in explicit:
            args.tracker_overlay_max_points = 0
    if (
        _table_z_filter_visual_default_requested(args, version)
        and "--enable-table-z-filter" not in explicit
        and "--disable-table-z-filter" not in explicit
    ):
        args.enable_table_z_filter = True
    if "--track-mode" not in explicit and str(args.render_mode) == "none" and not headless_capture:
        args.track_mode = TRACK_MODE_NONE
        if "--controller-instance-mode" not in explicit:
            args.controller_instance_mode = masked_pcd.CONTROLLER_INSTANCE_MODE_SINGLE
    if "--tracker-backend" not in explicit and (
        (str(args.render_mode) == "none" and not headless_capture) or str(args.track_mode) == TRACK_MODE_NONE
    ):
        args.tracker_backend = masked_pcd.TRACKER_BACKEND_NONE
    if str(args.input_source) == INPUT_SOURCE_FAKE_LIVE and args.recording_case is None:
        args.recording_case = default_fake_live_case_for_version(version)
    if str(args.input_source) == INPUT_SOURCE_FAKE_LIVE and "--replay-fps" not in explicit:
        args.replay_fps = DEFAULT_FAKE_LIVE_REPLAY_FPS
        args.fake_live_replay_fps_defaulted = True
    else:
        args.fake_live_replay_fps_defaulted = False
    preset_filter = masked_pcd.pcd_filter_preset_to_filter(getattr(args, "pcd_filter_preset", None))
    if preset_filter is not None:
        args.enable_pcd_filter = True
        if "--pcd-filter-mode" not in explicit:
            args.pcd_filter_mode = "sync"
        if "--object-filter" not in explicit:
            args.object_filter = preset_filter
        if "--controller-filter" not in explicit:
            args.controller_filter = preset_filter
        if str(args.pcd_filter_preset) == masked_pcd.PCD_FILTER_PRESET_ORIGINAL:
            if "--object-filter-cap" not in explicit:
                args.object_filter_cap = 0
            if "--controller-filter-cap" not in explicit:
                args.controller_filter_cap = 0
    if version in {DEMO_VERSION_3_2, DEMO_VERSION_3_3} and bool(args.enable_pcd_filter):
        if "--filter-radius-m" not in explicit:
            args.filter_radius_m = FFS_SURFACE_FILTER_RADIUS_M
        if "--filter-nb-points" not in explicit:
            args.filter_nb_points = FFS_SURFACE_FILTER_NB_POINTS
        if "--enhanced-component-voxel-size-m" not in explicit:
            args.enhanced_component_voxel_size_m = FFS_SURFACE_COMPONENT_VOXEL_SIZE_M
        if "--filter-every-n" not in explicit:
            args.filter_every_n = FFS_SURFACE_FILTER_EVERY_N
        if "--filter-max-age-frames" not in explicit:
            args.filter_max_age_frames = FFS_SURFACE_FILTER_MAX_AGE_FRAMES
        if "--pcd-mask-erode-pixels" in explicit:
            if "--object-pcd-mask-erode-pixels" not in explicit:
                args.object_pcd_mask_erode_pixels = int(args.pcd_mask_erode_pixels)
            if "--controller-pcd-mask-erode-pixels" not in explicit:
                args.controller_pcd_mask_erode_pixels = int(args.pcd_mask_erode_pixels)
        else:
            if "--object-pcd-mask-erode-pixels" not in explicit:
                args.object_pcd_mask_erode_pixels = FFS_SURFACE_OBJECT_MASK_ERODE_PIXELS
            if "--controller-pcd-mask-erode-pixels" not in explicit:
                args.controller_pcd_mask_erode_pixels = FFS_SURFACE_CONTROLLER_MASK_ERODE_PIXELS
    return args


def validate_args(args: argparse.Namespace) -> None:
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    explicit = set(getattr(args, "_explicit_options", set()))
    args.depth_source = DEFAULT_DEPTH_SOURCES[version]
    args.tracker_backend = masked_pcd.normalize_tracker_backend(str(args.tracker_backend))
    if str(args.input_source) not in INPUT_SOURCES:
        raise ValueError(f"--input-source must be one of {INPUT_SOURCES}")
    headless_capture = _headless_capture_requested(args, version)
    if args.headless_capture_dir is not None and not headless_capture:
        raise ValueError("--headless-capture-dir requires Demo 3.2/3.3 --input-source fake-live --render-mode none")
    if float(args.replay_fps) < 0.0:
        raise ValueError("--replay-fps must be >= 0")
    if _is_replay_input_source(str(args.input_source)):
        if args.recording_case is None:
            raise ValueError(f"--input-source {args.input_source} requires --recording-case or --fake-live-case")
        if not _interactive_tracking_render_requested(args) and not headless_capture:
            if str(args.input_source) == INPUT_SOURCE_RECORDING:
                raise ValueError(f"--input-source {args.input_source} requires --render-mode pointcloud")
            raise ValueError(f"--input-source {args.input_source} requires --render-mode pointcloud or panel")
        if str(args.track_mode) == TRACK_MODE_NONE:
            raise ValueError(f"--input-source {args.input_source} requires --track-mode controller-object")
        if str(args.tracker_backend) != masked_pcd.TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError(f"--input-source {args.input_source} requires --tracker-backend tapnextpp")
    elif args.recording_case is not None:
        raise ValueError("--recording-case/--fake-live-case requires --input-source recording or fake-live")
    if str(args.render_mode) == masked_pcd.RENDER_MODE_PANEL:
        if str(args.input_source) != INPUT_SOURCE_FAKE_LIVE:
            raise ValueError("--render-mode panel requires --input-source fake-live")
        if str(args.depth_source) != DEPTH_SOURCE_FFS:
            raise ValueError("--render-mode panel requires --depth-source ffs")
        if str(args.track_mode) != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("--render-mode panel requires --track-mode controller-object")
        if hasattr(args, "pcd_mode") and str(args.pcd_mode) != "masked":
            raise ValueError("--render-mode panel requires --pcd-mode masked")
        if str(args.tracker_backend) != masked_pcd.TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("--render-mode panel requires --tracker-backend tapnextpp")
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
    if float(args.duration_s) < 0.0:
        raise ValueError("--duration-s must be >= 0")
    if int(args.fps) not in single_pcd.SUPPORTED_CAPTURE_FPS:
        raise ValueError(f"--fps must be one of {single_pcd.SUPPORTED_CAPTURE_FPS}")
    if str(args.profile) not in single_pcd.SUPPORTED_PROFILES:
        raise ValueError(f"--profile must be one of {single_pcd.SUPPORTED_PROFILES}")
    if str(args.track_mode) not in TRACK_MODES:
        raise ValueError(f"--track-mode must be one of {TRACK_MODES}")
    if str(args.demo_visual_mode) not in DEMO_VISUAL_MODES:
        raise ValueError(f"--demo-visual-mode must be one of {DEMO_VISUAL_MODES}")
    if str(args.view_mode) not in masked_pcd.VIEW_MODES:
        raise ValueError(f"--view-mode must be one of {masked_pcd.VIEW_MODES}")
    if int(args.tracker_query_count) < 0:
        raise ValueError("--tracker-query-count must be >= 0")
    if int(args.tracker_overlay_max_points) < 0:
        raise ValueError("--tracker-overlay-max-points must be >= 0")
    if float(args.tracker_marker_point_size) <= 0:
        raise ValueError("--tracker-marker-point-size must be positive")
    if float(args.depth_min_m) < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if float(args.depth_max_m) > 0 and float(args.depth_max_m) <= float(args.depth_min_m):
        raise ValueError("--depth-max-m must be <=0 or greater than --depth-min-m")
    if float(args.table_z_filter_threshold_m) < 0:
        raise ValueError("--table-z-filter-threshold-m must be >= 0")
    if bool(args.enable_table_z_filter) and bool(getattr(args, "disable_table_z_filter", False)):
        raise ValueError("--enable-table-z-filter conflicts with --disable-table-z-filter")
    if str(args.table_z_above_direction) not in masked_pcd.TABLE_Z_ABOVE_DIRECTIONS:
        raise ValueError(f"--table-z-above-direction must be one of {masked_pcd.TABLE_Z_ABOVE_DIRECTIONS}")
    if str(args.table_z_filter_classes) not in masked_pcd.TABLE_Z_FILTER_CLASSES:
        raise ValueError(f"--table-z-filter-classes must be one of {masked_pcd.TABLE_Z_FILTER_CLASSES}")
    if int(args.pcd_max_points) < 0:
        raise ValueError("--pcd-max-points must be >= 0")
    if int(args.pcd_stride) < 1:
        raise ValueError("--pcd-stride must be >= 1")
    if int(args.render_max_points_per_layer) < 0:
        raise ValueError("--render-max-points-per-layer must be >= 0")
    if int(args.edgetam_live_session_keep_frames) < 0:
        raise ValueError("--edgetam-live-session-keep-frames must be >= 0")
    if str(args.pcd_filter_mode) not in masked_pcd.PCD_FILTER_MODES:
        raise ValueError(f"--pcd-filter-mode must be one of {masked_pcd.PCD_FILTER_MODES}")
    preset_filter = masked_pcd.pcd_filter_preset_to_filter(getattr(args, "pcd_filter_preset", None))
    if preset_filter is not None and "--pcd-filter-preset" in explicit:
        if "--object-filter" in explicit and str(args.object_filter) != preset_filter:
            raise ValueError("--pcd-filter-preset conflicts with --object-filter")
        if "--controller-filter" in explicit and str(args.controller_filter) != preset_filter:
            raise ValueError("--pcd-filter-preset conflicts with --controller-filter")
        if str(args.pcd_filter_preset) == masked_pcd.PCD_FILTER_PRESET_ORIGINAL:
            if "--object-filter-cap" in explicit and int(args.object_filter_cap) != 0:
                raise ValueError("--pcd-filter-preset original requires --object-filter-cap 0")
            if "--controller-filter-cap" in explicit and int(args.controller_filter_cap) != 0:
                raise ValueError("--pcd-filter-preset original requires --controller-filter-cap 0")
    if str(args.object_filter) not in masked_pcd.PCD_FILTERS:
        raise ValueError(f"--object-filter must be one of {masked_pcd.PCD_FILTERS}")
    if str(args.controller_filter) not in masked_pcd.PCD_FILTERS:
        raise ValueError(f"--controller-filter must be one of {masked_pcd.PCD_FILTERS}")
    if int(args.object_filter_cap) < 0:
        raise ValueError("--object-filter-cap must be >= 0")
    if int(args.controller_filter_cap) < 0:
        raise ValueError("--controller-filter-cap must be >= 0")
    if int(args.object_filter_keep_components) < 1:
        raise ValueError("--object-filter-keep-components must be >= 1")
    if int(args.controller_filter_keep_components) < 1:
        raise ValueError("--controller-filter-keep-components must be >= 1")
    if float(args.object_filter_voxel_m) <= 0:
        raise ValueError("--object-filter-voxel-m must be positive")
    if float(args.controller_filter_voxel_m) <= 0:
        raise ValueError("--controller-filter-voxel-m must be positive")
    if int(args.filter_every_n) < 1:
        raise ValueError("--filter-every-n must be >= 1")
    if int(args.filter_max_age_frames) < 0:
        raise ValueError("--filter-max-age-frames must be >= 0")
    if int(args.pcd_mask_erode_pixels) < 0:
        raise ValueError("--pcd-mask-erode-pixels must be >= 0")
    if args.object_pcd_mask_erode_pixels is not None and int(args.object_pcd_mask_erode_pixels) < 0:
        raise ValueError("--object-pcd-mask-erode-pixels must be >= 0")
    if args.controller_pcd_mask_erode_pixels is not None and int(args.controller_pcd_mask_erode_pixels) < 0:
        raise ValueError("--controller-pcd-mask-erode-pixels must be >= 0")
    if float(args.filter_budget_ms) < 0:
        raise ValueError("--filter-budget-ms must be >= 0")
    if int(args.filter_min_cap) < 0:
        raise ValueError("--filter-min-cap must be >= 0")
    if int(args.object_filter_cap) > 0 and int(args.filter_min_cap) > int(args.object_filter_cap):
        raise ValueError("--filter-min-cap must be <= --object-filter-cap when object cap is enabled")
    if int(args.controller_filter_cap) > 0 and int(args.filter_min_cap) > int(args.controller_filter_cap):
        raise ValueError("--filter-min-cap must be <= --controller-filter-cap when controller cap is enabled")
    if int(args.voxel_density_min_points) < 1:
        raise ValueError("--voxel-density-min-points must be >= 1")
    if float(args.filter_radius_m) <= 0:
        raise ValueError("--filter-radius-m must be positive")
    if int(args.filter_nb_points) < 1:
        raise ValueError("--filter-nb-points must be >= 1")
    if float(args.enhanced_component_voxel_size_m) <= 0:
        raise ValueError("--enhanced-component-voxel-size-m must be positive")
    if float(args.enhanced_keep_near_main_gap_m) < 0:
        raise ValueError("--enhanced-keep-near-main-gap-m must be >= 0")
    if float(args.point_size) <= 0:
        raise ValueError("--point-size must be positive")
    if bool(args.enable_pcd_filter) and str(args.track_mode) == TRACK_MODE_NONE:
        raise ValueError("--enable-pcd-filter requires --track-mode controller-object")
    if headless_capture:
        if DEFAULT_DEPTH_SOURCES[version] != DEPTH_SOURCE_FFS:
            raise ValueError("headless capture requires Demo 3.2/3.3 FFS depth")
        if not bool(args.enable_pcd_filter):
            raise ValueError("headless capture requires --enable-pcd-filter")
        if str(args.pcd_filter_mode) != "sync":
            raise ValueError("headless capture requires --pcd-filter-mode sync")
        if str(args.object_filter) not in masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
            allowed = ", ".join(masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
            raise ValueError(f"headless capture requires --object-filter one of {allowed}")
        if str(args.controller_filter) not in masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
            allowed = ", ".join(masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
            raise ValueError(f"headless capture requires --controller-filter one of {allowed}")
    if _demo_visual_mode_policy_requested(args, version):
        if not bool(args.enable_pcd_filter):
            raise ValueError("--demo-visual-mode requires --enable-pcd-filter for Demo 3.2/3.3")
        if str(args.pcd_filter_mode) != "sync":
            raise ValueError("--demo-visual-mode requires --pcd-filter-mode sync for Demo 3.2/3.3")
        visual_filter = preset_filter or _visual_mode_required_filter(args)
        if str(args.demo_visual_mode) == DEMO_VISUAL_MODE_PCD:
            if str(args.object_filter) != visual_filter:
                raise ValueError(f"--demo-visual-mode pcd requires --object-filter {visual_filter} for Demo 3.2/3.3")
            if str(args.controller_filter) != visual_filter:
                raise ValueError(f"--demo-visual-mode pcd requires --controller-filter {visual_filter} for Demo 3.2/3.3")
        elif headless_capture:
            if str(args.object_filter) not in masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
                allowed = ", ".join(masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
                raise ValueError(f"--demo-visual-mode requires --object-filter one of {allowed} for headless Demo 3.2/3.3")
            if str(args.controller_filter) not in masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
                allowed = ", ".join(masked_pcd.HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
                raise ValueError(
                    f"--demo-visual-mode requires --controller-filter one of {allowed} for headless Demo 3.2/3.3"
                )
        else:
            if str(args.object_filter) != visual_filter:
                raise ValueError(f"--demo-visual-mode tracking requires --object-filter {visual_filter} for Demo 3.2/3.3")
            if str(args.controller_filter) != visual_filter:
                raise ValueError(f"--demo-visual-mode tracking requires --controller-filter {visual_filter} for Demo 3.2/3.3")
        if str(args.pcd_color_mode) != "rgb":
            raise ValueError("--demo-visual-mode requires --pcd-color-mode rgb for Demo 3.2/3.3")
        if str(args.track_mode) != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("--demo-visual-mode requires --track-mode controller-object for Demo 3.2/3.3")
        if str(args.tracker_backend) != masked_pcd.TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("--demo-visual-mode requires --tracker-backend tapnextpp for full-pipeline FPS")
        if int(args.tracker_overlay_max_points) != 0:
            raise ValueError("--demo-visual-mode requires --tracker-overlay-max-points 0")
    if str(args.tracker_backend) != masked_pcd.TRACKER_BACKEND_NONE:
        if str(args.tracker_backend) != masked_pcd.TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("single demo tracker backend currently supports only tapnextpp")
        if str(args.track_mode) != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("--tracker-backend tapnextpp requires --track-mode controller-object")
        if not _interactive_tracking_render_requested(args) and not headless_capture:
            raise ValueError("--tracker-backend tapnextpp requires --render-mode pointcloud or panel")


def _read_recording_metadata_fps(case_path: Path | None) -> float | None:
    if case_path is None:
        return None
    metadata_path = Path(case_path).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = REPO_ROOT / metadata_path
    metadata_path = metadata_path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        fps = float(metadata.get("fps", 0.0))
    except (TypeError, ValueError):
        return None
    return fps if fps > 0.0 else None


def _contract_recording_fps(args: argparse.Namespace) -> tuple[float | None, str | None]:
    if not _is_replay_input_source(str(args.input_source)):
        return None, None
    metadata_fps = _read_recording_metadata_fps(args.recording_case)
    if metadata_fps is not None:
        return metadata_fps, "metadata"
    return DEFAULT_RECORDING_FPS, "default_30fps"


def _contract_replay_fps(args: argparse.Namespace) -> tuple[float | None, str | None]:
    if not _is_replay_input_source(str(args.input_source)):
        return None, None
    requested = float(args.replay_fps)
    if requested > 0.0:
        if bool(getattr(args, "fake_live_replay_fps_defaulted", False)):
            return requested, "default_fake_live"
        return requested, "cli"
    recording_fps, recording_fps_source = _contract_recording_fps(args)
    if recording_fps is not None:
        return recording_fps, recording_fps_source
    return None, None


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    depth_source = DEFAULT_DEPTH_SOURCES[version]
    uses_ffs = depth_source == DEPTH_SOURCE_FFS
    depth_pipeline = "ffs_tensorrt_batch1" if uses_ffs else "realsense_native"
    prompts = _mode_prompts(str(args.mode))
    controller_prompt = str(getattr(args, "controller_prompt", None) or prompts["controller_prompt"])
    controller_label = DEFAULT_DEMO_CONTROLLER_LABEL if str(args.mode) == MODE_DEMO else controller_prompt
    input_source = str(args.input_source)
    if input_source == INPUT_SOURCE_FAKE_LIVE:
        contract_input_source = "fake_live_recorded_single_camera"
    elif input_source == INPUT_SOURCE_RECORDING:
        contract_input_source = "recording_single_camera"
    else:
        contract_input_source = "live_realsense_single_camera"
    replay_fps, replay_fps_source = _contract_replay_fps(args)
    recording_fps, recording_fps_source = _contract_recording_fps(args)
    headless_capture = _headless_capture_requested(args, version)
    tracker_on = str(args.tracker_backend) != masked_pcd.TRACKER_BACKEND_NONE
    tracker_visualization_mode = "phystwin_rainbow_identity_3d_lift" if tracker_on else "none"
    tracker_sync_policy = "strict_same_seq_lossless_5fps" if tracker_on and str(args.track_mode) != TRACK_MODE_NONE else "none"
    query_display_policy = "visible_3d_lifted_all" if tracker_on else "none"
    query_color_mode = "phystwin_rainbow_identity" if tracker_on else "none"
    contract: dict[str, Any] = {
        "demo": f"single-demo{version}",
        "demo_version": version,
        "demo_display_name": VERSION_LABELS[version],
        "runtime_module": "qqtt.demo.single_demo_v3_runtime",
        "live_delegate_module": "qqtt.demo.realtime_masked_edgetam_pcd",
        "input_source": contract_input_source,
        "input_source_mode": input_source,
        "recording_case": None if args.recording_case is None else str(args.recording_case),
        "table_calibration_path": None if args.table_calibrate is None else str(args.table_calibrate),
        "table_world_frame_kind": None if args.table_calibrate is None else TABLE_WORLD_FRAME_KIND,
        "replay_fps": replay_fps,
        "replay_fps_source": replay_fps_source,
        "recording_fps": recording_fps,
        "recording_fps_source": recording_fps_source,
        "fake_live_frame_selection_policy": (
            masked_pcd.FAKE_LIVE_FRAME_SELECTION_POLICY if input_source == INPUT_SOURCE_FAKE_LIVE else None
        ),
        "camera_count": 1,
        "serial": None if _is_replay_input_source(input_source) or args.serial is None else str(args.serial),
        "coordinate_frame": single_pcd.COORDINATE_FRAME,
        "pcd_coordinate_frame": TABLE_WORLD_FRAME_KIND if args.table_calibrate is not None else single_pcd.COORDINATE_FRAME,
        "camera_coordinate_frame": single_pcd.COORDINATE_FRAME,
        "table_z_m": 0.0 if args.table_calibrate is not None else None,
        "world_z_diagnostic_thresholds_m": [float(value) for value in masked_pcd.DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M],
        "table_z_filter_enabled": bool(args.enable_table_z_filter),
        "table_z_filter_threshold_m": float(args.table_z_filter_threshold_m),
        "table_z_above_direction": str(args.table_z_above_direction),
        "table_z_filter_classes": str(args.table_z_filter_classes),
        "depth_source": depth_source,
        "depth_pipeline": depth_pipeline,
        "uses_ffs": uses_ffs,
        "ffs_trt_batch_size": 1 if uses_ffs else None,
        "mask_source": "hf_edgetam" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "init_mode": "sam31_first_frame" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "mask_propagation": "hf_edgetam_online" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "semantic_mode": str(prompts["semantic_mode"]),
        "object_prompt": str(args.object_prompt),
        "controller_prompt": controller_prompt,
        "controller_label": controller_label,
        "controller_instance_mode": str(args.controller_instance_mode),
        "edgetam_tracking_identities": _edgetam_tracking_identities(args),
        "track_mode": str(args.track_mode),
        "render_mode": str(args.render_mode),
        "panel_layout": str(getattr(args, "panel_layout", masked_pcd.PANEL_LAYOUT_SIDE_BY_SIDE)),
        "panel_video_output": None if getattr(args, "panel_video_output", None) is None else str(args.panel_video_output),
        "tracking_background_mask": str(getattr(args, "tracking_background_mask", "target-union")),
        "panel_backend": (
            masked_pcd.PANEL_BACKEND_OPEN3D_MULTI_VIEWPORT
            if str(getattr(args, "render_mode", "")) == "panel"
            else "none"
        ),
        "panel_sync_policy": (
            masked_pcd.PANEL_SYNC_POLICY_STRICT_SAME_SEQ
            if str(getattr(args, "render_mode", "")) == "panel"
            else "none"
        ),
        "demo_visual_mode": str(args.demo_visual_mode),
        "headless_capture_enabled": bool(headless_capture),
        "headless_capture_dir": None if not headless_capture or args.headless_capture_dir is None else str(args.headless_capture_dir),
        "saved_pcd_source": _headless_capture_saved_pcd_source(args) if headless_capture else None,
        "view_mode": str(args.view_mode),
        "tracker_backend": str(args.tracker_backend),
        "tracker_backend_family": (
            "tapnext" if str(args.tracker_backend) == masked_pcd.TRACKER_BACKEND_TAPNEXTPP else "none"
        ),
        "tracker_device": str(args.tracker_device),
        "tracker_query_count": int(args.tracker_query_count),
        "tracker_query_source": (
            masked_pcd.tracker_query_source(args) if tracker_on else None
        ),
        "tracker_marker_gate": (
            masked_pcd.tracker_marker_gate(args) if tracker_on else None
        ),
        "tracker_retire_filtered_markers": (
            masked_pcd.tracker_retire_filtered_markers(args) if tracker_on else None
        ),
        "tracker_marker_retirement_policy": (
            masked_pcd.tracker_marker_retirement_policy(args) if tracker_on else None
        ),
        "tracker_display_scope": str(args.tracker_display_scope),
        "tracker_visualization_mode": tracker_visualization_mode,
        "tracker_sync_policy": tracker_sync_policy,
        "query_display_policy": query_display_policy,
        "query_color_mode": query_color_mode,
        "tracker_overlay_max_points": int(args.tracker_overlay_max_points),
        "tracker_marker_point_size": float(args.tracker_marker_point_size),
        "tapnet_repo_dir": str(args.tapnet_repo_dir),
        "tapnextpp_checkpoint": str(args.tapnextpp_checkpoint),
        "tapnextpp_image_size": str(args.tapnextpp_image_size),
        "tapnextpp_autocast_dtype": str(args.tapnextpp_autocast_dtype),
        "tapnextpp_compile": bool(args.tapnextpp_compile),
        "tapnextpp_fast_postprocess": bool(args.tapnextpp_fast_postprocess),
        "edgetam_live_session_keep_frames": int(args.edgetam_live_session_keep_frames),
        "depth_min_m": float(args.depth_min_m),
        "depth_max_m": float(args.depth_max_m),
        "pcd_max_points": int(args.pcd_max_points),
        "pcd_stride": int(args.pcd_stride),
        "pcd_color_mode": str(args.pcd_color_mode),
        "pcd_mask_erode_pixels": int(args.pcd_mask_erode_pixels),
        "object_pcd_mask_erode_pixels": _effective_object_pcd_mask_erode_pixels(args),
        "controller_pcd_mask_erode_pixels": _effective_controller_pcd_mask_erode_pixels(args),
        "render_max_points_per_layer": int(args.render_max_points_per_layer),
        "pcd_filter_enabled": bool(args.enable_pcd_filter),
        "pcd_filter_mode": str(args.pcd_filter_mode if bool(args.enable_pcd_filter) else masked_pcd.PCD_FILTER_NONE),
        "pcd_filter_preset": _effective_pcd_filter_preset(args),
        "object_filter": str(args.object_filter),
        "controller_filter": str(args.controller_filter),
        "object_filter_cap": int(args.object_filter_cap),
        "controller_filter_cap": int(args.controller_filter_cap),
        "object_filter_keep_components": int(args.object_filter_keep_components),
        "controller_filter_keep_components": int(args.controller_filter_keep_components),
        "object_filter_voxel_m": float(args.object_filter_voxel_m),
        "controller_filter_voxel_m": float(args.controller_filter_voxel_m),
        "filter_every_n": int(args.filter_every_n),
        "filter_max_age_frames": int(args.filter_max_age_frames),
        "filter_budget_ms": float(args.filter_budget_ms),
        "filter_min_cap": int(args.filter_min_cap),
        "voxel_density_min_points": int(args.voxel_density_min_points),
        "filter_radius_m": float(args.filter_radius_m),
        "filter_nb_points": int(args.filter_nb_points),
        "enhanced_component_voxel_size_m": float(args.enhanced_component_voxel_size_m),
        "enhanced_keep_near_main_gap_m": float(args.enhanced_keep_near_main_gap_m),
        "point_size": float(args.point_size),
        "profile": str(args.profile),
        "fps": int(args.fps),
        "duration_s": float(args.duration_s),
        "output_root": str(args.output_root),
    }
    contract["profile_summary_fields"] = {
        "camera_count": 1,
        "depth_source": depth_source,
        "depth_pipeline": depth_pipeline,
        "uses_ffs": uses_ffs,
        "ffs_trt_batch_size": contract["ffs_trt_batch_size"],
        "rendered_fps": 0.0,
        "capture_fps": 0.0,
        "mask_fps": 0.0,
        "tracker_fps": 0.0,
    }
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "demo_display_name",
        "input_source",
        "recording_case",
        "table_calibration_path",
        "table_world_frame_kind",
        "replay_fps",
        "recording_fps",
        "fake_live_frame_selection_policy",
        "camera_count",
        "serial",
        "pcd_coordinate_frame",
        "depth_source",
        "depth_pipeline",
        "uses_ffs",
        "ffs_trt_batch_size",
        "mask_source",
        "track_mode",
        "tracker_backend",
        "tracker_device",
        "tracker_query_count",
        "tracker_query_source",
        "tracker_marker_gate",
        "tracker_retire_filtered_markers",
        "tracker_marker_retirement_policy",
        "tracker_display_scope",
        "object_prompt",
        "controller_prompt",
        "controller_instance_mode",
        "edgetam_tracking_identities",
        "render_mode",
        "panel_layout",
        "panel_video_output",
        "tracking_background_mask",
        "panel_backend",
        "panel_sync_policy",
        "demo_visual_mode",
        "headless_capture_enabled",
        "headless_capture_dir",
        "saved_pcd_source",
        "view_mode",
        "pcd_max_points",
        "pcd_stride",
        "pcd_mask_erode_pixels",
        "object_pcd_mask_erode_pixels",
        "controller_pcd_mask_erode_pixels",
        "render_max_points_per_layer",
        "pcd_filter_enabled",
        "pcd_filter_mode",
        "pcd_filter_preset",
        "object_filter",
        "controller_filter",
        "tracker_visualization_mode",
        "tracker_sync_policy",
        "query_display_policy",
        "query_color_mode",
        "filter_radius_m",
        "filter_nb_points",
        "enhanced_component_voxel_size_m",
        "filter_every_n",
        "filter_max_age_frames",
        "edgetam_live_session_keep_frames",
        "point_size",
    )
    lines = []
    for key in keys:
        value = contract[key]
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"{key} = {rendered}")
    lines.append(json.dumps(contract, indent=2, sort_keys=True))
    return "\n".join(lines)


def validate_live_contract(
    args: argparse.Namespace,
    *,
    connected_serials_provider: ConnectedSerialsProvider | None = None,
) -> dict[str, Any]:
    validate_args(args)
    provider = connected_serials_provider or _get_connected_realsense_serials
    connected_serials = list(provider())
    requested = str(args.serial or "")
    if requested:
        if requested not in connected_serials:
            raise RuntimeError(f"single demo requested RealSense serial is not connected: {requested}")
        active_serial = requested
    else:
        if not connected_serials:
            raise RuntimeError("single demo requires at least one connected RealSense camera.")
        active_serial = connected_serials[0]
    return {
        "connected_serials": connected_serials,
        "active_serial": active_serial,
    }


def build_live_delegate_argv(args: argparse.Namespace, *, active_serial: str | None = None) -> list[str]:
    validate_args(args)
    argv = [
        "--profile",
        str(args.profile),
        "--fps",
        str(int(args.fps)),
        "--input-source",
        str(args.input_source),
        "--depth-source",
        str(args.depth_source),
        "--duration-s",
        str(float(args.duration_s)),
        "--track-mode",
        str(args.track_mode),
        "--render-mode",
        str(args.render_mode),
        "--panel-layout",
        str(args.panel_layout),
        "--tracking-background-mask",
        str(args.tracking_background_mask),
        "--demo-visual-mode",
        str(args.demo_visual_mode),
        "--view-mode",
        str(args.view_mode),
        "--tracker-backend",
        str(args.tracker_backend),
        "--tracker-device",
        str(args.tracker_device),
        "--tracker-query-count",
        str(int(args.tracker_query_count)),
        "--tracker-seed",
        str(int(args.tracker_seed)),
        "--tracker-display-scope",
        str(args.tracker_display_scope),
        "--tracker-overlay-max-points",
        str(int(args.tracker_overlay_max_points)),
        "--tracker-marker-point-size",
        str(float(args.tracker_marker_point_size)),
        "--tapnet-repo-dir",
        str(args.tapnet_repo_dir),
        "--tapnextpp-checkpoint",
        str(args.tapnextpp_checkpoint),
        "--tapnextpp-image-size",
        str(args.tapnextpp_image_size),
        "--tapnextpp-autocast-dtype",
        str(args.tapnextpp_autocast_dtype),
        "--edgetam-live-session-keep-frames",
        str(int(args.edgetam_live_session_keep_frames)),
        "--depth-min-m",
        str(float(args.depth_min_m)),
        "--depth-max-m",
        str(float(args.depth_max_m)),
        "--pcd-max-points",
        str(int(args.pcd_max_points)),
        "--pcd-stride",
        str(int(args.pcd_stride)),
        "--pcd-color-mode",
        str(args.pcd_color_mode),
        "--pcd-mask-erode-pixels",
        str(int(args.pcd_mask_erode_pixels)),
        "--object-pcd-mask-erode-pixels",
        str(_effective_object_pcd_mask_erode_pixels(args)),
        "--controller-pcd-mask-erode-pixels",
        str(_effective_controller_pcd_mask_erode_pixels(args)),
        "--render-max-points-per-layer",
        str(int(args.render_max_points_per_layer)),
        "--pcd-filter-mode",
        str(args.pcd_filter_mode),
        "--object-filter",
        str(args.object_filter),
        "--controller-filter",
        str(args.controller_filter),
        "--object-filter-cap",
        str(int(args.object_filter_cap)),
        "--controller-filter-cap",
        str(int(args.controller_filter_cap)),
        "--object-filter-keep-components",
        str(int(args.object_filter_keep_components)),
        "--controller-filter-keep-components",
        str(int(args.controller_filter_keep_components)),
        "--object-filter-voxel-m",
        str(float(args.object_filter_voxel_m)),
        "--controller-filter-voxel-m",
        str(float(args.controller_filter_voxel_m)),
        "--filter-every-n",
        str(int(args.filter_every_n)),
        "--filter-max-age-frames",
        str(int(args.filter_max_age_frames)),
        "--filter-budget-ms",
        str(float(args.filter_budget_ms)),
        "--filter-min-cap",
        str(int(args.filter_min_cap)),
        "--voxel-density-min-points",
        str(int(args.voxel_density_min_points)),
        "--filter-radius-m",
        str(float(args.filter_radius_m)),
        "--filter-nb-points",
        str(int(args.filter_nb_points)),
        "--enhanced-component-voxel-size-m",
        str(float(args.enhanced_component_voxel_size_m)),
        "--enhanced-keep-near-main-gap-m",
        str(float(args.enhanced_keep_near_main_gap_m)),
        "--table-z-filter-threshold-m",
        str(float(args.table_z_filter_threshold_m)),
        "--table-z-above-direction",
        str(args.table_z_above_direction),
        "--table-z-filter-classes",
        str(args.table_z_filter_classes),
        "--object-prompt",
        str(args.object_prompt),
        "--controller-prompt",
        str(args.controller_prompt),
        "--controller-instance-mode",
        str(args.controller_instance_mode),
        "--point-size",
        str(float(args.point_size)),
    ]
    if _is_replay_input_source(str(args.input_source)):
        argv.extend(["--recording-case", str(args.recording_case)])
        if float(args.replay_fps) > 0.0:
            argv.extend(["--replay-fps", str(float(args.replay_fps))])
    if not _is_replay_input_source(str(args.input_source)):
        serial = active_serial or args.serial
        if serial:
            argv.extend(["--serial", str(serial)])
    if str(args.depth_source) == DEPTH_SOURCE_FFS:
        argv.extend(["--ffs-repo", str(args.ffs_repo), "--ffs-trt-model-dir", str(args.ffs_trt_model_dir)])
        if args.ffs_trt_root is not None:
            argv.extend(["--ffs-trt-root", str(args.ffs_trt_root)])
    if bool(args.tapnextpp_compile):
        argv.append("--tapnextpp-compile")
    if not bool(args.tapnextpp_fast_postprocess):
        argv.append("--no-tapnextpp-fast-postprocess")
    if bool(args.tracker_retire_filtered_markers):
        argv.append("--tracker-retire-filtered-markers")
    else:
        argv.append("--no-tracker-retire-filtered-markers")
    if bool(args.enable_pcd_filter):
        argv.append("--enable-pcd-filter")
    if bool(args.enable_table_z_filter):
        argv.append("--enable-table-z-filter")
    if bool(getattr(args, "disable_table_z_filter", False)):
        argv.append("--disable-table-z-filter")
    pcd_filter_preset = _effective_pcd_filter_preset(args)
    if pcd_filter_preset is not None:
        argv.extend(["--pcd-filter-preset", str(pcd_filter_preset)])
    if args.headless_capture_dir is not None:
        argv.extend(["--headless-capture-dir", str(args.headless_capture_dir)])
    if args.panel_video_output is not None:
        argv.extend(["--panel-video-output", str(args.panel_video_output)])
    if args.table_calibrate is not None:
        argv.extend(["--table-calibrate", str(args.table_calibrate)])
    if bool(args.debug):
        argv.append("--debug")
    if str(args.track_mode) == TRACK_MODE_NONE:
        argv.extend(["--pcd-mode", "none"])
    return argv


def _write_profile(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    demo_version: str = DEMO_VERSION_3,
    connected_serials_provider: ConnectedSerialsProvider | None = None,
) -> int:
    parser = build_arg_parser(demo_version=demo_version)
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=_explicit_cli_options(argv))
        validate_args(args)
        contract = build_contract(args)
        if args.dry_run:
            print(format_contract(contract))
            _write_profile(args.profile_json_output, {"contract": contract, "summary": contract["profile_summary_fields"]})
            return 0
        if _is_replay_input_source(str(args.input_source)):
            delegate_argv = build_live_delegate_argv(args)
        else:
            live_validation = validate_live_contract(args, connected_serials_provider=connected_serials_provider)
            delegate_argv = build_live_delegate_argv(args, active_serial=live_validation["active_serial"])
        return int(masked_pcd.main(delegate_argv))
    except (FileNotFoundError, RuntimeError, ValueError, argparse.ArgumentTypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "DEMO_VERSION_3",
    "DEMO_VERSION_3_1",
    "DEMO_VERSION_3_2",
    "DEMO_VERSION_3_3",
    "INPUT_SOURCE_LIVE",
    "INPUT_SOURCE_FAKE_LIVE",
    "INPUT_SOURCE_RECORDING",
    "MODE_DEMO",
    "MODE_EXP",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_live_delegate_argv",
    "format_contract",
    "main",
    "validate_args",
    "validate_live_contract",
]
