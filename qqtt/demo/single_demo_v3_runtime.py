from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from qqtt.demo import realtime_masked_edgetam_pcd as masked_pcd
from qqtt.demo import realtime_single_camera_pointcloud as single_pcd


DEMO_VERSION_3 = "3"
DEMO_VERSION_3_1 = "3.1"
DEMO_VERSION_3_2 = "3.2"
DEMO_VERSION_3_3 = "3.3"
DEMO_VERSIONS = (DEMO_VERSION_3, DEMO_VERSION_3_1, DEMO_VERSION_3_2, DEMO_VERSION_3_3)

DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_FFS = "ffs"

INPUT_SOURCE_LIVE = "live"
INPUT_SOURCE_RECORDING = "recording"
INPUT_SOURCES = (INPUT_SOURCE_LIVE, INPUT_SOURCE_RECORDING)

MODE_EXP = "exp"
MODE_DEMO = "demo"
MODES = (MODE_EXP, MODE_DEMO)

TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_NONE = "none"
TRACK_MODES = (TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_NONE)

DEFAULT_OBJECT_PROMPT = "stuffed animal"
DEFAULT_EXP_CONTROLLER_PROMPT = "towel"
DEFAULT_DEMO_CONTROLLER_PROMPT = "human hand"
DEFAULT_DEMO_CONTROLLER_LABEL = "hand"
DEFAULT_MODE = MODE_EXP

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
        help="Frame source. recording replays one raw RGB-D data_collect case as the camera stream.",
    )
    parser.add_argument(
        "--recording-case",
        type=Path,
        default=None,
        help="Raw data_collect case folder for --input-source recording.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help="Replay FPS for --input-source recording. Use 0 to read metadata fps.",
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
    parser.add_argument("--point-size", type=float, default=2.0)
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
    args.depth_source = DEFAULT_DEPTH_SOURCES[version]
    if "--controller-prompt" not in explicit or args.controller_prompt is None:
        args.controller_prompt = _mode_prompts(str(args.mode))["controller_prompt"]
    if "--track-mode" not in explicit and str(args.render_mode) == "none":
        args.track_mode = TRACK_MODE_NONE
    return args


def validate_args(args: argparse.Namespace) -> None:
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    args.depth_source = DEFAULT_DEPTH_SOURCES[version]
    if str(args.input_source) not in INPUT_SOURCES:
        raise ValueError(f"--input-source must be one of {INPUT_SOURCES}")
    if float(args.replay_fps) < 0.0:
        raise ValueError("--replay-fps must be >= 0")
    if str(args.input_source) == INPUT_SOURCE_RECORDING:
        if args.recording_case is None:
            raise ValueError("--input-source recording requires --recording-case")
        if args.depth_source != DEPTH_SOURCE_REALSENSE:
            raise ValueError("recording replay currently supports only Single Demo 3 / 3.1 RealSense RGB-D")
        if str(args.render_mode) != "pointcloud":
            raise ValueError("--input-source recording requires --render-mode pointcloud")
        if str(args.track_mode) == TRACK_MODE_NONE:
            raise ValueError("--input-source recording requires --track-mode controller-object")
    elif args.recording_case is not None:
        raise ValueError("--recording-case requires --input-source recording")
    if float(args.duration_s) < 0.0:
        raise ValueError("--duration-s must be >= 0")
    if int(args.fps) not in single_pcd.SUPPORTED_CAPTURE_FPS:
        raise ValueError(f"--fps must be one of {single_pcd.SUPPORTED_CAPTURE_FPS}")
    if str(args.profile) not in single_pcd.SUPPORTED_PROFILES:
        raise ValueError(f"--profile must be one of {single_pcd.SUPPORTED_PROFILES}")
    if str(args.track_mode) not in TRACK_MODES:
        raise ValueError(f"--track-mode must be one of {TRACK_MODES}")


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


def _contract_replay_fps(args: argparse.Namespace) -> tuple[float | None, str | None]:
    if str(args.input_source) != INPUT_SOURCE_RECORDING:
        return None, None
    requested = float(args.replay_fps)
    if requested > 0.0:
        return requested, "cli"
    metadata_fps = _read_recording_metadata_fps(args.recording_case)
    if metadata_fps is not None:
        return metadata_fps, "metadata"
    return None, "metadata_unresolved"


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    version = normalize_demo_version(getattr(args, "single_demo_version", DEMO_VERSION_3))
    depth_source = DEFAULT_DEPTH_SOURCES[version]
    uses_ffs = depth_source == DEPTH_SOURCE_FFS
    depth_pipeline = "ffs_tensorrt_batch1" if uses_ffs else "realsense_native"
    prompts = _mode_prompts(str(args.mode))
    controller_prompt = str(getattr(args, "controller_prompt", None) or prompts["controller_prompt"])
    controller_label = DEFAULT_DEMO_CONTROLLER_LABEL if str(args.mode) == MODE_DEMO else controller_prompt
    input_source = str(args.input_source)
    contract_input_source = (
        "recording_rgbd_single_camera" if input_source == INPUT_SOURCE_RECORDING else "live_realsense_single_camera"
    )
    replay_fps, replay_fps_source = _contract_replay_fps(args)
    contract: dict[str, Any] = {
        "demo": f"single-demo{version}",
        "demo_version": version,
        "demo_display_name": VERSION_LABELS[version],
        "runtime_module": "qqtt.demo.single_demo_v3_runtime",
        "live_delegate_module": "qqtt.demo.realtime_masked_edgetam_pcd",
        "input_source": contract_input_source,
        "input_source_mode": input_source,
        "recording_case": None if args.recording_case is None else str(args.recording_case),
        "replay_fps": replay_fps,
        "replay_fps_source": replay_fps_source,
        "camera_count": 1,
        "serial": None if input_source == INPUT_SOURCE_RECORDING or args.serial is None else str(args.serial),
        "coordinate_frame": single_pcd.COORDINATE_FRAME,
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
        "track_mode": str(args.track_mode),
        "render_mode": str(args.render_mode),
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
    }
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "demo_display_name",
        "input_source",
        "recording_case",
        "replay_fps",
        "camera_count",
        "serial",
        "depth_source",
        "depth_pipeline",
        "uses_ffs",
        "ffs_trt_batch_size",
        "mask_source",
        "track_mode",
        "object_prompt",
        "controller_prompt",
        "render_mode",
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
        "--object-prompt",
        str(args.object_prompt),
        "--controller-prompt",
        str(args.controller_prompt),
        "--point-size",
        str(float(args.point_size)),
    ]
    if str(args.input_source) == INPUT_SOURCE_RECORDING:
        argv.extend(["--recording-case", str(args.recording_case)])
        if float(args.replay_fps) > 0.0:
            argv.extend(["--replay-fps", str(float(args.replay_fps))])
    if str(args.input_source) != INPUT_SOURCE_RECORDING:
        serial = active_serial or args.serial
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
        if str(args.input_source) == INPUT_SOURCE_RECORDING:
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
