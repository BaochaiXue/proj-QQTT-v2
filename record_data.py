from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from shutil import copy2
import json

from qqtt.env.camera.defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)

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


def probe_supports_mode(
    *,
    capture_mode: str,
    serials: list[str],
    width: int,
    height: int,
    fps: int,
    emitter: str,
) -> bool | None:
    probe_path = _resolve_path("./docs/generated/d455_stream_probe_results.json")
    if not probe_path.exists():
        return None
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    topology_type = "single" if len(serials) == 1 else "three_camera" if len(serials) == 3 else None
    stream_set = {"stereo_ir": "rgb_ir_pair", "both_eval": "rgbd_ir_pair"}.get(capture_mode)
    if topology_type is None or stream_set is None:
        return None

    ordered_serials = serials
    for case in data.get("cases", []):
        if (
            case.get("topology_type") == topology_type
            and case.get("stream_set") == stream_set
            and case.get("serials") == ordered_serials
            and case.get("width") == width
            and case.get("height") == height
            and case.get("fps") == fps
            and case.get("emitter_request") == emitter
        ):
            return bool(case.get("success"))
    return None


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
    support = probe_supports_mode(
        capture_mode=args.capture_mode,
        serials=effective_serials if effective_serials else [],
        width=args.width,
        height=args.height,
        fps=args.fps,
        emitter=args.emitter,
    ) if effective_serials else None
    if args.capture_mode == "both_eval" and support is False:
        raise RuntimeError(
            "both_eval is blocked by the latest D455 stream probe on this machine "
            f"for serials={effective_serials}, {args.width}x{args.height}@{args.fps}, emitter={args.emitter}."
        )

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
        support = probe_supports_mode(
            capture_mode=args.capture_mode,
            serials=effective_serials,
            width=args.width,
            height=args.height,
            fps=args.fps,
            emitter=args.emitter,
        )
    if args.capture_mode == "both_eval" and support is False:
        camera_system.realsense.stop()
        raise RuntimeError(
            "both_eval is blocked by the latest D455 stream probe on this machine "
            f"for serials={effective_serials}, {args.width}x{args.height}@{args.fps}, emitter={args.emitter}. "
            f"See {_resolve_path('./docs/generated/d455_stream_probe_results.md')}."
        )
    if args.capture_mode == "stereo_ir" and support is False:
        print(
            "[record] warning: latest D455 stream probe marked this stereo_ir profile as unstable; "
            "recording will still be attempted."
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
