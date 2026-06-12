from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Callable, Sequence

import numpy as np

from qqtt.demo import demo3_runtime
from qqtt.demo import realtime_masked_edgetam_pcd as masked_pcd
from qqtt.demo import realtime_single_camera_pointcloud as single_pcd


DEMO_VERSION_3 = "3"
DEMO_VERSION_3_1 = "3.1"
DEMO_VERSION_3_2 = "3.2"
DEMO_VERSION_3_3 = "3.3"
DEMO_VERSIONS = (DEMO_VERSION_3, DEMO_VERSION_3_1, DEMO_VERSION_3_2, DEMO_VERSION_3_3)

LIVE_DELEGATE_MASKED_PCD = "masked-pcd"
LIVE_DELEGATE_POINTCLOUD = "pointcloud"
LIVE_DELEGATES = (LIVE_DELEGATE_MASKED_PCD, LIVE_DELEGATE_POINTCLOUD)

DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_FFS = "ffs"
DEPTH_SOURCES = (DEPTH_SOURCE_REALSENSE, DEPTH_SOURCE_FFS)

DEFAULT_CAMERA_IDS = (0,)
DEFAULT_OBJECT_PROMPT = "stuffed animal"
DEFAULT_EXP_CONTROLLER_PROMPT = "towel"
DEFAULT_DEMO_CONTROLLER_PROMPT = "human hand"
DEFAULT_DEMO_CONTROLLER_LABEL = "hand"
DEFAULT_MODE = demo3_runtime.MODE_EXP
MODES = demo3_runtime.MODES

TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_NONE = "none"
TRACK_MODES = (TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_NONE)

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

ConnectedSerialsProvider = Callable[[], Sequence[str]]


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


def parse_single_camera_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        ids = tuple(int(item) for item in value)
    if len(ids) != 1:
        raise argparse.ArgumentTypeError("single demo v3.x expects exactly one camera id, e.g. 0")
    if ids[0] < 0:
        raise argparse.ArgumentTypeError(f"camera id must be non-negative: {ids[0]}")
    return ids


def _mode_prompts(mode: str) -> dict[str, str]:
    resolved = demo3_runtime.resolve_demo3_mode(mode)
    return {
        "semantic_mode": str(resolved["semantic_mode"]),
        "controller_prompt": str(resolved["controller_prompt"]),
        "controller_label": str(resolved["controller_label"]),
        "shared_experiment_mode": str(resolved["experiment_mode"]),
    }


def _calibration_transform_count_or_none(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Unsupported calibrate.pkl transform shape: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("calibrate.pkl contains non-finite transform values.")
    return int(arr.shape[0])


def _get_connected_realsense_serials() -> list[str]:
    return demo3_runtime._get_connected_realsense_serials()


def build_arg_parser(*, demo_version: str = DEMO_VERSION_3) -> argparse.ArgumentParser:
    version = normalize_demo_version(demo_version)
    label = VERSION_LABELS[version]
    parser = argparse.ArgumentParser(
        description=(
            f"{label}: single-camera RealSense demo for the single-camera branch. "
            "It removes three-camera sync, world fusion, batch=3 FFS, and dual-GPU "
            "tracker requirements from the copied Demo 3.x surface."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(single_demo_version=version)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved single-demo contract and exit.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Run duration. 0 means until closed.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-json-output", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOTS[version])
    parser.add_argument("--camera-ids", type=parse_single_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--serial", default=None, help="Single RealSense serial to open. Defaults to first detected serial.")
    parser.add_argument(
        "--serials",
        nargs="*",
        default=None,
        help="Compatibility alias for one serial. Pass exactly one value if used.",
    )
    parser.add_argument(
        "--calibrate-path",
        type=Path,
        default=Path("calibrate.pkl"),
        help="Optional metadata only; single-demo live rendering stays in camera frame.",
    )
    parser.add_argument("--profile", choices=single_pcd.SUPPORTED_PROFILES, default=single_pcd.DEFAULT_PROFILE)
    parser.add_argument("--fps", type=int, choices=single_pcd.SUPPORTED_CAPTURE_FPS, default=single_pcd.DEFAULT_FPS)
    parser.add_argument("--depth-source", choices=DEPTH_SOURCES, default=DEFAULT_DEPTH_SOURCES[version])
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
        "--track-mode",
        choices=TRACK_MODES,
        default=TRACK_MODE_CONTROLLER_OBJECT,
        help="Single-camera EdgeTAM mask mode for the masked-PCD live delegate.",
    )
    parser.add_argument(
        "--render-mode",
        choices=masked_pcd.RENDER_MODES,
        default=masked_pcd.DEFAULT_RENDER_MODE,
        help="Render mode for the masked-PCD live delegate.",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--live-delegate",
        choices=LIVE_DELEGATES,
        default=LIVE_DELEGATE_MASKED_PCD,
        help="Existing single-camera runtime used for non-dry-run execution.",
    )
    parser.add_argument(
        "--shape-prior-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Contract flag only for single Demo 3.3; live warmup is intentionally not launched here.",
    )
    return parser


def apply_preset_defaults(
    args: argparse.Namespace,
    *,
    explicit_options: set[str] | None = None,
) -> argparse.Namespace:
    explicit = set(explicit_options or set())
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    if "--output-root" not in explicit:
        args.output_root = DEFAULT_OUTPUT_ROOTS[version]
    if "--depth-source" not in explicit:
        args.depth_source = DEFAULT_DEPTH_SOURCES[version]
    if "--controller-prompt" not in explicit or args.controller_prompt is None:
        args.controller_prompt = _mode_prompts(str(args.mode))["controller_prompt"]
    if "--track-mode" not in explicit and str(args.render_mode) == "none":
        args.track_mode = TRACK_MODE_NONE
    return args


def validate_args(args: argparse.Namespace) -> None:
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    camera_ids = parse_single_camera_ids(args.camera_ids)
    if len(set(camera_ids)) != 1:
        raise ValueError(f"single demo v3.x camera ids must be unique: {camera_ids}")
    serials = list(args.serials or [])
    if args.serial and serials and serials != [args.serial]:
        raise ValueError("Use either --serial or --serials with the same single serial, not conflicting values.")
    if len(serials) > 1:
        raise ValueError("single demo v3.x accepts at most one requested RealSense serial.")
    if str(args.depth_source) not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {DEPTH_SOURCES}")
    if version in {DEMO_VERSION_3, DEMO_VERSION_3_1} and str(args.depth_source) == DEPTH_SOURCE_FFS:
        raise ValueError(f"{VERSION_LABELS[version]} is RealSense-depth only; use single_demo_v3_2 or v3_3 for FFS.")
    if float(args.duration_s) < 0.0:
        raise ValueError("--duration-s must be >= 0")
    if int(args.fps) not in single_pcd.SUPPORTED_CAPTURE_FPS:
        raise ValueError(f"--fps must be one of {single_pcd.SUPPORTED_CAPTURE_FPS}")
    if str(args.profile) not in single_pcd.SUPPORTED_PROFILES:
        raise ValueError(f"--profile must be one of {single_pcd.SUPPORTED_PROFILES}")
    if str(args.track_mode) not in TRACK_MODES:
        raise ValueError(f"--track-mode must be one of {TRACK_MODES}")
    if str(args.live_delegate) == LIVE_DELEGATE_POINTCLOUD and str(args.render_mode) == "none":
        raise ValueError("--live-delegate pointcloud does not support --render-mode none.")
    if bool(getattr(args, "shape_prior_warmup", False)) and version != DEMO_VERSION_3_3:
        raise ValueError("--shape-prior-warmup is only meaningful on single Demo 3.3.")
    _calibration_transform_count_or_none(Path(args.calibrate_path))


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    camera_ids = parse_single_camera_ids(args.camera_ids)
    prompts = _mode_prompts(str(args.mode))
    controller_prompt = str(getattr(args, "controller_prompt", None) or prompts["controller_prompt"])
    controller_label = DEFAULT_DEMO_CONTROLLER_LABEL if str(args.mode) == demo3_runtime.MODE_DEMO else controller_prompt
    depth_source = str(args.depth_source)
    uses_ffs = depth_source == DEPTH_SOURCE_FFS
    calibration_transform_count = _calibration_transform_count_or_none(Path(args.calibrate_path))
    point_tracker_enabled = False
    contract: dict[str, Any] = {
        "demo": f"single-demo{version}",
        "demo_version": version,
        "demo_display_name": VERSION_LABELS[version],
        "runtime_module": "qqtt.demo.single_demo_v3_runtime",
        "input_source": "live_realsense_single_camera",
        "offline_mode_available": False,
        "requires_single_realsense": True,
        "requires_three_realsense": False,
        "num_cameras": 1,
        "num_realsense_cameras": 1,
        "camera_ids": list(camera_ids),
        "serial": None if args.serial is None and not args.serials else str(args.serial or args.serials[0]),
        "calibrate_path": str(args.calibrate_path),
        "calibrate_pkl_required": False,
        "calibrate_pkl_loaded": bool(Path(args.calibrate_path).is_file()),
        "calibration_transform_count": calibration_transform_count,
        "coordinate_frame": single_pcd.COORDINATE_FRAME,
        "camera_sync_required": False,
        "multi_camera_world_fusion": False,
        "multi_camera_calibration_required": False,
        "depth_source": depth_source,
        "uses_ffs": uses_ffs,
        "ffs_trt_batch_size": 1 if uses_ffs else 0,
        "ffs_schedule": "single-camera-latest" if uses_ffs else "disabled",
        "ffs_batch3_required": False,
        "mask_source": "hf_edgetam" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "single_camera_masked_pcd": str(args.live_delegate) == LIVE_DELEGATE_MASKED_PCD,
        "init_mode": "sam31_first_frame" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "mask_propagation": "hf_edgetam_online" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "semantic_mode": str(prompts["semantic_mode"]),
        "object_prompt": str(args.object_prompt),
        "controller_prompt": controller_prompt,
        "tracking_controller_label": controller_label,
        "tracking_mask_scope": "object_controller_union" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "track_mode": str(args.track_mode),
        "point_tracker_enabled": point_tracker_enabled,
        "point_tracker_live_stage": "removed_for_single_camera_branch",
        "tracking_backend_execution_mode": "disabled",
        "tracking_backend_batch_size": 0,
        "tracking_backend_model_instances_expected": 0,
        "tracking_query_count_requested": 0,
        "overlay_display_scope": "masked_pcd_semantics" if str(args.track_mode) != TRACK_MODE_NONE else "none",
        "dual_gpu_enabled": False,
        "required_cuda_devices": 1,
        "require_two_cuda": False,
        "cross_gpu_cuda_tensor_transfer": False,
        "strict_source_three_camera_bundle": False,
        "live_delegate": str(args.live_delegate),
        "live_delegate_module": (
            "qqtt.demo.realtime_masked_edgetam_pcd"
            if str(args.live_delegate) == LIVE_DELEGATE_MASKED_PCD
            else "qqtt.demo.realtime_single_camera_pointcloud"
        ),
        "render_mode": str(args.render_mode),
        "profile": str(args.profile),
        "fps": int(args.fps),
        "duration_s": float(args.duration_s),
        "output_root": str(args.output_root),
        "shape_prior_warmup_enabled": bool(version == DEMO_VERSION_3_3 and getattr(args, "shape_prior_warmup", False)),
        "shape_prior_live_stage": "not_launched_by_single_demo_wrapper",
        "removed_three_camera_work": [
            "three_realsense_required_count",
            "multi_camera_capture_synchronization",
            "multi_camera_world_fusion",
            "calibrate_pkl_world_transform_requirement",
            "strict_source_three_camera_bundle",
            "batch3_tensor_rt_depth",
            "batch_views_tracker_execution",
            "dual_gpu_split_requirement",
            "per_camera_overlay_caps",
        ],
    }
    contract["profile_summary_fields"] = {
        "num_realsense_cameras": 1,
        "requires_three_realsense": False,
        "depth_source": depth_source,
        "uses_ffs": uses_ffs,
        "ffs_trt_batch_size": contract["ffs_trt_batch_size"],
        "rendered_fps": 0.0,
        "capture_fps": 0.0,
        "mask_fps": 0.0,
        "point_tracker_enabled": point_tracker_enabled,
    }
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "demo_display_name",
        "input_source",
        "requires_single_realsense",
        "requires_three_realsense",
        "num_cameras",
        "camera_ids",
        "camera_sync_required",
        "multi_camera_world_fusion",
        "calibrate_pkl_required",
        "depth_source",
        "uses_ffs",
        "ffs_trt_batch_size",
        "ffs_batch3_required",
        "mask_source",
        "track_mode",
        "object_prompt",
        "controller_prompt",
        "point_tracker_enabled",
        "tracking_backend_execution_mode",
        "dual_gpu_enabled",
        "required_cuda_devices",
        "live_delegate",
        "shape_prior_warmup_enabled",
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
    requested = str(args.serial or (args.serials[0] if args.serials else "") or "")
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
        "active_serials": [active_serial],
        "active_serial": active_serial,
        "calibration_transform_count": _calibration_transform_count_or_none(Path(args.calibrate_path)),
    }


def _build_masked_pcd_argv(args: argparse.Namespace, *, active_serial: str | None = None) -> list[str]:
    argv = [
        "--profile",
        str(args.profile),
        "--fps",
        str(int(args.fps)),
        "--depth-source",
        str(args.depth_source),
        "--duration-s",
        str(float(args.duration_s)),
        "--track-mode",
        str(args.track_mode),
        "--render-mode",
        str(args.render_mode),
        "--object-prompt",
        str(args.object_prompt),
        "--controller-prompt",
        str(args.controller_prompt),
        "--point-size",
        str(float(args.point_size)),
    ]
    serial = active_serial or args.serial or (args.serials[0] if args.serials else None)
    if serial:
        argv.extend(["--serial", str(serial)])
    if str(args.depth_source) == DEPTH_SOURCE_FFS:
        argv.extend(["--ffs-repo", str(args.ffs_repo), "--ffs-trt-model-dir", str(args.ffs_trt_model_dir)])
        if args.ffs_trt_root is not None:
            argv.extend(["--ffs-trt-root", str(args.ffs_trt_root)])
    if bool(args.debug):
        argv.append("--debug")
    if str(args.track_mode) == TRACK_MODE_NONE:
        argv.extend(["--pcd-mode", "none"])
    return argv


def _build_pointcloud_argv(args: argparse.Namespace, *, active_serial: str | None = None) -> list[str]:
    argv = [
        "--profile",
        str(args.profile),
        "--fps",
        str(int(args.fps)),
        "--depth-source",
        str(args.depth_source),
        "--duration-s",
        str(float(args.duration_s)),
        "--point-size",
        str(float(args.point_size)),
        "--render-backend",
        "pointcloud",
    ]
    serial = active_serial or args.serial or (args.serials[0] if args.serials else None)
    if serial:
        argv.extend(["--serial", str(serial)])
    if str(args.depth_source) == DEPTH_SOURCE_FFS:
        argv.extend(["--ffs-repo", str(args.ffs_repo), "--ffs-trt-model-dir", str(args.ffs_trt_model_dir)])
        if args.ffs_trt_root is not None:
            argv.extend(["--ffs-trt-root", str(args.ffs_trt_root)])
    if bool(args.debug):
        argv.append("--debug")
    return argv


def build_live_delegate_argv(args: argparse.Namespace, *, active_serial: str | None = None) -> list[str]:
    if str(args.live_delegate) == LIVE_DELEGATE_MASKED_PCD:
        return _build_masked_pcd_argv(args, active_serial=active_serial)
    if str(args.live_delegate) == LIVE_DELEGATE_POINTCLOUD:
        return _build_pointcloud_argv(args, active_serial=active_serial)
    raise ValueError(f"Unsupported --live-delegate {args.live_delegate}")


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
        args = apply_preset_defaults(args, explicit_options=demo3_runtime._explicit_cli_options(argv))
        validate_args(args)
        contract = build_contract(args)
        if args.dry_run:
            print(format_contract(contract))
            _write_profile(args.profile_json_output, {"contract": contract, "summary": contract["profile_summary_fields"]})
            return 0
        live_validation = validate_live_contract(args, connected_serials_provider=connected_serials_provider)
        delegate_argv = build_live_delegate_argv(args, active_serial=live_validation["active_serial"])
        if str(args.live_delegate) == LIVE_DELEGATE_MASKED_PCD:
            return int(masked_pcd.main(delegate_argv))
        return int(single_pcd.main(delegate_argv))
    except (FileNotFoundError, RuntimeError, ValueError, argparse.ArgumentTypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "DEMO_VERSION_3",
    "DEMO_VERSION_3_1",
    "DEMO_VERSION_3_2",
    "DEMO_VERSION_3_3",
    "LIVE_DELEGATE_MASKED_PCD",
    "LIVE_DELEGATE_POINTCLOUD",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_live_delegate_argv",
    "format_contract",
    "main",
    "parse_single_camera_ids",
    "validate_args",
    "validate_live_contract",
]
