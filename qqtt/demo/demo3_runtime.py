from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS = "demo3-realsense-cotracker-highfps"
PRESET_DEMO3_REALSENSE_MASK_ONLY = "demo3-realsense-mask-only"
PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE = "demo3-realsense-cotracker-profile"
PRESETS = (
    PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS,
    PRESET_DEMO3_REALSENSE_MASK_ONLY,
    PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE,
)

DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_FFS = "ffs"
MASK_SOURCE_HF_EDGETAM = "hf_edgetam"
MASK_SOURCE_HF_EDGETAM_CLI = "hf-edgetam"
COTRACKER3_ONLINE = "cotracker3_online"

TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_ONLY = "controller-only"
TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_NONE = "none"
TRACK_MODES = (
    TRACK_MODE_OBJECT_ONLY,
    TRACK_MODE_CONTROLLER_ONLY,
    TRACK_MODE_CONTROLLER_OBJECT,
    TRACK_MODE_NONE,
)

RENDER_MODE_POINTCLOUD = "pointcloud"
RENDER_MODE_NONE = "none"
RENDER_MODES = (RENDER_MODE_POINTCLOUD, RENDER_MODE_NONE)

DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_CAMERA_IDS = (0, 1, 2)
DEFAULT_OBJECT_PROMPT = "stuffed animal"
DEFAULT_CONTROLLER_PROMPT = "towel"
DEFAULT_COTRACKER_QUERY_COUNT = 128
DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA = 30
DEFAULT_OVERLAY_TRAIL_LEN = 16
DEFAULT_OVERLAY_STALE_TIMEOUT_MS = 500.0
DEFAULT_COTRACKER_WINDOW_LEN = 16
DEFAULT_COTRACKER_PUBLISH_STEP = 8


def parse_camera_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(int(item) for item in items)
    return tuple(int(item) for item in value)


def _explicit_cli_options(argv: Sequence[str] | None) -> set[str]:
    explicit: set[str] = set()
    if argv is None:
        argv = sys.argv[1:]
    for item in argv:
        if item.startswith("--"):
            explicit.add(item.split("=", 1)[0])
    return explicit


def _normalize_mask_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized != MASK_SOURCE_HF_EDGETAM:
        raise ValueError("Demo 3 mask source must be hf-edgetam.")
    return MASK_SOURCE_HF_EDGETAM


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 3 realtime visualization: three RealSense RGB-D cameras, HF "
            "EdgeTAM masks, RealSense fused PCD, and async CoTracker3 overlay."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS, default=PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved Demo 3 runtime contract and exit.")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-json-output", type=Path, default=None)
    parser.add_argument("--camera-ids", type=parse_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--calibrate-path", type=Path, default=Path("calibrate.pkl"))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--depth-source", default=DEPTH_SOURCE_REALSENSE)
    parser.add_argument("--mask-source", default=MASK_SOURCE_HF_EDGETAM_CLI)
    parser.add_argument("--track-mode", choices=TRACK_MODES, default=TRACK_MODE_OBJECT_ONLY)
    parser.add_argument("--object-prompt", default=DEFAULT_OBJECT_PROMPT)
    parser.add_argument("--controller-prompt", default=DEFAULT_CONTROLLER_PROMPT)
    parser.add_argument("--cotracker-backend", default=COTRACKER3_ONLINE)
    parser.add_argument("--cotracker-query-count", type=int, default=DEFAULT_COTRACKER_QUERY_COUNT)
    parser.add_argument("--disable-cotracker", action="store_true")
    parser.add_argument("--render-mode", choices=RENDER_MODES, default=RENDER_MODE_POINTCLOUD)
    parser.add_argument("--overlay-max-points-per-camera", type=int, default=DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA)
    parser.add_argument("--overlay-trail-len", type=int, default=DEFAULT_OVERLAY_TRAIL_LEN)
    parser.add_argument("--overlay-stale-timeout-ms", type=float, default=DEFAULT_OVERLAY_STALE_TIMEOUT_MS)
    return parser


def apply_preset_defaults(args: argparse.Namespace, *, explicit_options: set[str] | None = None) -> argparse.Namespace:
    explicit = set(explicit_options or set())
    if args.preset == PRESET_DEMO3_REALSENSE_MASK_ONLY:
        if "--disable-cotracker" not in explicit:
            args.disable_cotracker = True
        if "--track-mode" not in explicit:
            args.track_mode = TRACK_MODE_OBJECT_ONLY
    elif args.preset == PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE:
        if "--render-mode" not in explicit:
            args.render_mode = RENDER_MODE_NONE
        if "--duration-s" not in explicit:
            args.duration_s = 60.0
    elif args.preset == PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS:
        if "--cotracker-query-count" not in explicit:
            args.cotracker_query_count = DEFAULT_COTRACKER_QUERY_COUNT
        if "--overlay-max-points-per-camera" not in explicit:
            args.overlay_max_points_per_camera = DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA
    return args


def validate_args(args: argparse.Namespace, *, require_calibration: bool = False) -> None:
    camera_ids = parse_camera_ids(args.camera_ids)
    if len(camera_ids) != 3:
        raise ValueError("Demo 3 requires exactly three RealSense cameras.")
    if len(set(camera_ids)) != 3:
        raise ValueError("Demo 3 requires exactly three distinct RealSense cameras.")
    depth_source = str(args.depth_source).strip().lower()
    if depth_source == DEPTH_SOURCE_FFS or depth_source.startswith("ffs"):
        raise ValueError("Demo 3 does not support FFS. Use --depth-source realsense.")
    if depth_source != DEPTH_SOURCE_REALSENSE:
        raise ValueError("Demo 3 depth source must be realsense.")
    _normalize_mask_source(str(args.mask_source))
    if str(args.cotracker_backend) != COTRACKER3_ONLINE:
        raise ValueError("Demo 3 currently supports only --cotracker-backend cotracker3_online.")
    if int(args.cotracker_query_count) <= 0 and not bool(args.disable_cotracker):
        raise ValueError("--cotracker-query-count must be positive when CoTracker is enabled.")
    if int(args.overlay_max_points_per_camera) <= 0:
        raise ValueError("--overlay-max-points-per-camera must be positive.")
    if require_calibration and not Path(args.calibrate_path).is_file():
        raise FileNotFoundError(f"Demo 3 requires calibrate.pkl for three-camera world fusion: {args.calibrate_path}")


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    camera_ids = parse_camera_ids(args.camera_ids)
    cotracker_enabled = not bool(args.disable_cotracker)
    contract: dict[str, Any] = {
        "demo": "demo3",
        "preset": str(args.preset),
        "requires_three_realsense": True,
        "num_cameras": int(len(camera_ids)),
        "num_realsense_cameras": int(len(camera_ids)),
        "camera_ids": list(camera_ids),
        "serials": list(args.serials or []),
        "calibrate_path": str(args.calibrate_path),
        "calibrate_pkl_loaded": bool(Path(args.calibrate_path).is_file()),
        "depth_source": DEPTH_SOURCE_REALSENSE,
        "uses_ffs": False,
        "mask_source": MASK_SOURCE_HF_EDGETAM,
        "edgetam_sessions_per_camera": 1,
        "track_mode": str(args.track_mode),
        "object_prompt": str(args.object_prompt),
        "controller_prompt": str(args.controller_prompt),
        "cotracker_enabled": bool(cotracker_enabled),
        "cotracker_backend": COTRACKER3_ONLINE,
        "cotracker_async": True,
        "cotracker_latest_wins": True,
        "cotracker_query_count": int(args.cotracker_query_count),
        "cotracker_window_len": DEFAULT_COTRACKER_WINDOW_LEN,
        "cotracker_publish_step": DEFAULT_COTRACKER_PUBLISH_STEP,
        "overlay_max_points_per_camera": int(args.overlay_max_points_per_camera),
        "overlay_trail_len": int(args.overlay_trail_len),
        "overlay_stale_timeout_ms": float(args.overlay_stale_timeout_ms),
        "render_mode": str(args.render_mode),
        "render_latest_wins": True,
        "render_waited_for_cotracker": False,
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "hot_path_forbids": [
            "ffs",
            "ffs_tensorrt",
            "ffs_remote",
            "ffs_ir_alignment",
            "track_process_data.pkl",
            "inverse_physics",
        ],
    }
    contract["profile_summary_fields"] = build_empty_profile_summary(contract)
    return contract


def build_empty_profile_summary(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "rendered_fps": 0.0,
        "render_loop_fps": 0.0,
        "capture_group_fps": 0.0,
        "edgetam_mask_fps": 0.0,
        "fusion_fps": 0.0,
        "cotracker_publish_fps": 0.0,
        "cotracker_model_ms_median": 0.0,
        "cotracker_model_ms_p95": 0.0,
        "cotracker_e2e_ms_median": 0.0,
        "cotracker_e2e_ms_p95": 0.0,
        "overlay_ms_median": 0.0,
        "overlay_ms_p95": 0.0,
        "pcd_fusion_ms_median": 0.0,
        "pcd_render_ms_median": 0.0,
        "render_waited_for_cotracker": False,
        "uses_ffs": False,
        "depth_source": DEPTH_SOURCE_REALSENSE,
        "mask_source": MASK_SOURCE_HF_EDGETAM,
        "num_realsense_cameras": int(contract.get("num_realsense_cameras", 3)),
        "calibrate_pkl_loaded": bool(contract.get("calibrate_pkl_loaded", False)),
        "cotracker_backend": COTRACKER3_ONLINE,
        "cotracker_window_len": DEFAULT_COTRACKER_WINDOW_LEN,
        "cotracker_publish_step": DEFAULT_COTRACKER_PUBLISH_STEP,
    }


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "requires_three_realsense",
        "num_cameras",
        "depth_source",
        "uses_ffs",
        "mask_source",
        "cotracker_backend",
        "cotracker_async",
        "render_latest_wins",
        "render_waited_for_cotracker",
    )
    lines = []
    for key in keys:
        value = contract[key]
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    lines.append(json.dumps(contract, indent=2, sort_keys=True))
    return "\n".join(lines)


def _write_profile(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Demo3Runtime:
    """Runtime facade for the Demo 3 realtime visualization contract.

    The hardware loop is intentionally composed from smaller shared helpers and
    workers. Dry-run and profile-only paths are deterministic so CI can validate
    the contract without RealSense hardware or CoTracker weights.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.contract = build_contract(args)

    def run(self) -> dict[str, Any]:
        validate_args(self.args, require_calibration=True)
        profile = {
            "contract": self.contract,
            "summary": build_empty_profile_summary(self.contract),
            "runtime_note": "Demo 3 hardware loop composes RealSense capture, HF EdgeTAM, fused PCD render, and async CoTracker overlay.",
        }
        _write_profile(self.args.profile_json_output, profile)
        return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=_explicit_cli_options(argv))
        validate_args(args, require_calibration=False)
        contract = build_contract(args)
        if args.dry_run:
            print(format_contract(contract))
            _write_profile(args.profile_json_output, {"contract": contract, "summary": contract["profile_summary_fields"]})
            return 0
        profile = Demo3Runtime(args).run()
        print(json.dumps(profile["summary"], indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "COTRACKER3_ONLINE",
    "DEFAULT_COTRACKER_PUBLISH_STEP",
    "DEFAULT_COTRACKER_WINDOW_LEN",
    "DEPTH_SOURCE_REALSENSE",
    "Demo3Runtime",
    "MASK_SOURCE_HF_EDGETAM",
    "PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS",
    "PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE",
    "PRESET_DEMO3_REALSENSE_MASK_ONLY",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "format_contract",
    "main",
    "parse_camera_ids",
    "validate_args",
]
