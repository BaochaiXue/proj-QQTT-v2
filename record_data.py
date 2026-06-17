from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from shutil import copy2

from qqtt.env.camera.defaults import (
    DEFAULT_EXPOSURE,
    DEFAULT_FPS,
    DEFAULT_GAIN,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)
from qqtt.env.camera.calibration_metadata import (
    calibration_metadata_path_for,
    load_calibration_reference_serials,
)
from qqtt.env.camera.preflight import (
    evaluate_capture_preflight,
    format_capture_preflight_summary,
)
from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    table_calibration_metadata_path_for,
)

_PROJECT_ROOT = next(
    (
        p
        for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
        if (p / ".git").exists()
    ),
    Path(__file__).resolve().parent,
)
DEFAULT_CAMERA_START_TIMEOUT_S = 30.0


def _resolve_path(path: str) -> Path:
    return (_PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record RealSense raw data. The single-camera branch defaults to "
            "one camera; pass --num-cam or --serials for explicit multi-camera runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default=str(_resolve_path("./data_collect"))
    )
    parser.add_argument(
        "--calibrate_path",
        type=str,
        default=str(_resolve_path("./calibrate.pkl")),
        help="Calibration file to copy into the recorded case if it exists.",
    )
    parser.add_argument(
        "--table-calibrate",
        type=str,
        default=None,
        help="Optional table Z=0 calibration file to copy into the recorded case.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-cam", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument(
        "--exposure",
        type=float,
        default=DEFAULT_EXPOSURE,
        help="Base manual RGB exposure. Known lab-rig serials use shared per-camera overrides.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_GAIN,
        help="Base manual RGB gain. Known lab-rig serials use shared per-camera overrides.",
    )
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
        "--camera-start-timeout-s",
        type=float,
        default=DEFAULT_CAMERA_START_TIMEOUT_S,
        help="Fail camera startup if no first frame arrives within this many seconds.",
    )
    parser.add_argument(
        "--disable-keyboard-listener",
        action="store_true",
        help="Disable the keyboard listener used for spacebar start/stop.",
    )
    return parser


def _print_preflight_summary(*, decision, stage_label: str) -> None:
    print(format_capture_preflight_summary(decision, stage_label=stage_label))


def _stop_camera_system(camera_system) -> None:
    stop = getattr(camera_system, "stop", None)
    if callable(stop):
        try:
            stop()
            return
        except Exception:
            pass
    realsense = getattr(camera_system, "realsense", None)
    if realsense is not None:
        try:
            realsense.stop()
        except Exception:
            pass


def _raise_if_preflight_blocked(
    *, decision, stage_label: str, camera_system=None
) -> None:
    if decision.allowed_to_record:
        return
    if camera_system is not None:
        _stop_camera_system(camera_system)
    raise RuntimeError(
        f"Recording preflight blocked this capture profile {stage_label}. "
        f"{decision.reason} See {decision.probe_results_md}."
    )


def _update_case_metadata(case_metadata_path: Path, updates: dict[str, object]) -> None:
    metadata = json.loads(case_metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    case_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def validate_table_calibration_for_case(
    *,
    table_calibrate_path: Path,
    serial_numbers: list[str],
) -> dict[str, object]:
    table_path = Path(table_calibrate_path).expanduser().resolve()
    load_table_calibration_transforms(table_path, serial_numbers=list(serial_numbers))
    table_metadata = load_table_calibration_metadata(table_path)
    return {
        "path": table_path,
        "metadata": table_metadata,
        "metadata_sidecar_path": table_calibration_metadata_path_for(table_path),
    }


def copy_table_calibration_into_case(
    *,
    table_calibrate_path: Path,
    output_path: Path,
    serial_numbers: list[str],
    validated_table_calibration: dict[str, object] | None = None,
) -> None:
    table_calibration = validated_table_calibration
    if table_calibration is None:
        table_calibration = validate_table_calibration_for_case(
            table_calibrate_path=table_calibrate_path,
            serial_numbers=serial_numbers,
        )
    table_path = Path(table_calibration["path"])
    table_metadata = table_calibration["metadata"]
    metadata_sidecar_path = Path(table_calibration["metadata_sidecar_path"])

    output_path = Path(output_path)
    copied_table_name = "table_calibrate.pkl"
    copied_metadata_name = "table_calibrate_metadata.json"
    copy2(table_path, output_path / copied_table_name)
    copy2(metadata_sidecar_path, output_path / copied_metadata_name)
    _update_case_metadata(
        output_path / "metadata.json",
        {
            "table_calibration_path": copied_table_name,
            "table_calibration_metadata_path": copied_metadata_name,
            "table_world_frame_kind": TABLE_WORLD_FRAME_KIND,
            "table_calibration_reference_serials": list(
                table_metadata["table_calibration_reference_serials"]
            ),
        },
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
        stage_label=(
            "before camera discovery"
            if selected_serials is None
            else "before camera startup"
        ),
    )
    if effective_serials:
        _raise_if_preflight_blocked(
            decision=initial_preflight, stage_label="before camera startup"
        )

    calibrate_path = Path(args.calibrate_path).resolve()
    calibration_reference_serials = None
    if calibrate_path.exists():
        calibration_reference_serials = load_calibration_reference_serials(
            calibrate_path
        )
        if calibration_reference_serials is None:
            print(
                "[record] warning: calibration metadata sidecar was not found next to "
                f"{calibrate_path}. If the camera rig was physically moved or cameras were swapped, "
                "rerun cameras_calibrate.py before recording."
            )

    camera_system = CameraSystem(
        WH=[args.width, args.height],
        fps=args.fps,
        num_cam=args.num_cam,
        serial_numbers=args.serials,
        capture_mode=args.capture_mode,
        emitter=args.emitter,
        exposure=args.exposure,
        gain=args.gain,
        calibration_reference_serials=calibration_reference_serials,
        enable_keyboard_listener=not args.disable_keyboard_listener,
        camera_start_timeout_s=args.camera_start_timeout_s,
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
    _print_preflight_summary(
        decision=final_preflight, stage_label="after camera discovery"
    )
    _raise_if_preflight_blocked(
        decision=final_preflight,
        stage_label="after camera discovery",
        camera_system=camera_system,
    )
    validated_table_calibration = None
    if args.table_calibrate is not None:
        try:
            validated_table_calibration = validate_table_calibration_for_case(
                table_calibrate_path=Path(args.table_calibrate),
                serial_numbers=list(effective_serials),
            )
        except Exception:
            _stop_camera_system(camera_system)
            raise
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

    if calibrate_path.exists():
        copy2(calibrate_path, output_path / "calibrate.pkl")
        sidecar_path = calibration_metadata_path_for(calibrate_path)
        if sidecar_path.exists():
            copy2(sidecar_path, output_path / sidecar_path.name)
    else:
        print(
            f"[record] warning: calibrate file not found, skipping copy: {calibrate_path}"
        )
    if args.table_calibrate is not None:
        copy_table_calibration_into_case(
            table_calibrate_path=Path(args.table_calibrate),
            output_path=output_path,
            serial_numbers=list(effective_serials),
            validated_table_calibration=validated_table_calibration,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
