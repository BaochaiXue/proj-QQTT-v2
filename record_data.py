from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from shutil import copy2

from qqtt.env.camera.defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)
from qqtt.env.camera.preflight import evaluate_capture_preflight, format_capture_preflight_summary

_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> Path:
    return (_PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record synchronized multi-camera RealSense raw data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(_resolve_path("./data_collect")))
    parser.add_argument(
        "--calibrate_path",
        type=str,
        default=str(_resolve_path("./calibrate.pkl")),
        help="Calibration file to copy into the recorded case if it exists.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-cam", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument(
        "--capture_mode",
        type=str,
        choices=("rgbd", "stereo_ir", "both_eval"),
        default="rgbd",
    )
    parser.add_argument(
        "--emitter",
        type=str,
        choices=("on", "off", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Record this many frames per camera then stop automatically.",
    )
    parser.add_argument(
        "--disable-keyboard-listener",
        action="store_true",
        help="Disable the keyboard listener used for spacebar start/stop.",
    )
    return parser


def _print_preflight_summary(*, decision, stage_label: str) -> None:
    print(format_capture_preflight_summary(decision, stage_label=stage_label))


def _raise_if_preflight_blocked(*, decision, stage_label: str, camera_system=None) -> None:
    if decision.allowed_to_record:
        return
    if camera_system is not None:
        camera_system.realsense.stop()
    raise RuntimeError(
        f"Recording preflight blocked this capture profile {stage_label}. "
        f"{decision.reason} See {decision.probe_results_md}."
    )

def main() -> int:
    args = build_parser().parse_args()
    from qqtt.env import CameraSystem

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    case_name = args.case_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_root / case_name
    selected_serials = args.serials if args.serials else None
    effective_serials = selected_serials or []
    if selected_serials is None:
        # CameraSystem will pick the first num_cam connected devices in sorted order.
        pass
    initial_preflight = evaluate_capture_preflight(
        capture_mode=args.capture_mode,
        serials=None if not effective_serials else effective_serials,
        width=args.width,
        height=args.height,
        fps=args.fps,
        emitter=args.emitter,
    )
    _print_preflight_summary(
        decision=initial_preflight,
        stage_label="before camera discovery" if selected_serials is None else "before camera startup",
    )
    if effective_serials:
        _raise_if_preflight_blocked(decision=initial_preflight, stage_label="before camera startup")

    camera_system = CameraSystem(
        WH=[args.width, args.height],
        fps=args.fps,
        num_cam=args.num_cam,
        serial_numbers=args.serials,
        capture_mode=args.capture_mode,
        emitter=args.emitter,
        enable_keyboard_listener=not args.disable_keyboard_listener,
    )
    if not effective_serials:
        effective_serials = camera_system.serial_numbers
    final_preflight = evaluate_capture_preflight(
        capture_mode=args.capture_mode,
        serials=effective_serials,
        width=args.width,
        height=args.height,
        fps=args.fps,
        emitter=args.emitter,
    )
    _print_preflight_summary(decision=final_preflight, stage_label="after camera discovery")
    _raise_if_preflight_blocked(
        decision=final_preflight,
        stage_label="after camera discovery",
        camera_system=camera_system,
    )
    if final_preflight.operator_status == "experimental_warning":
        print(
            "[record] warning: preflight policy allows this unsupported profile experimentally; "
            "recording will still be attempted."
        )
    elif final_preflight.operator_status == "unknown":
        print(
            "[record] warning: preflight support is unknown for this exact profile; "
            "recording will still be attempted under current repo policy."
        )
    camera_system.record(output_path=str(output_path), max_frames=args.max_frames)

    calibrate_path = Path(args.calibrate_path).resolve()
    if calibrate_path.exists():
        copy2(calibrate_path, output_path / "calibrate.pkl")
    else:
        print(f"[record] warning: calibrate file not found, skipping copy: {calibrate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
