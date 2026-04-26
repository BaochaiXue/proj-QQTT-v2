"""Trim and align a raw multi-camera recording into an aligned case."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any


_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_process.aligned_case_metadata import write_split_aligned_metadata
from data_process.visualization.calibration_io import load_calibration_transforms

RAW_IMAGE_STREAMS = ("color", "ir_left", "ir_right")


def _resolve_path(path: str) -> Path:
    return (_PROJECT_ROOT / path).resolve()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Align and trim a raw recording into a case under ./data.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_path", type=Path, default=_resolve_path("./data_collect"))
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--output_path", type=Path, default=_resolve_path("./data"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS for optional mp4 generation. Defaults to the recording metadata fps.",
    )
    parser.add_argument(
        "--write_mp4",
        action="store_true",
        help="Generate mp4 videos from aligned frames if ffmpeg is available.",
    )
    parser.add_argument(
        "--depth_backend",
        type=str,
        choices=("realsense", "ffs", "both"),
        default="realsense",
    )
    parser.add_argument("--ffs_repo", type=str, default=None)
    parser.add_argument("--ffs_model_path", type=str, default=None)
    parser.add_argument("--ffs_scale", type=float, default=1.0)
    parser.add_argument("--ffs_valid_iters", type=int, default=8)
    parser.add_argument("--ffs_max_disp", type=int, default=192)
    parser.add_argument("--ffs_radius_outlier_filter", action="store_true")
    parser.add_argument("--ffs_radius_outlier_radius_m", type=float, default=0.01)
    parser.add_argument("--ffs_radius_outlier_nb_points", type=int, default=40)
    parser.add_argument(
        "--ffs_native_like_postprocess",
        action="store_true",
    )
    parser.add_argument(
        "--ffs_confidence_mode",
        choices=("none", "margin", "max_softmax", "entropy", "variance"),
        default="none",
        help="Optional PyTorch logits confidence mode for filtering FFS depth before uint16 encoding.",
    )
    parser.add_argument("--ffs_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--ffs_confidence_depth_min_m", type=float, default=0.2)
    parser.add_argument("--ffs_confidence_depth_max_m", type=float, default=1.5)
    parser.add_argument("--write_ffs_confidence_debug", action="store_true")
    parser.add_argument("--write_ffs_valid_mask_debug", action="store_true")
    parser.add_argument("--write_ffs_float_m", action="store_true")
    parser.add_argument("--fail_if_no_ir_stereo", action="store_true")
    return parser


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def validate_args(args: Any) -> None:
    if not args.case_name.strip():
        raise ValueError("--case_name must be non-empty")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.end < 0:
        raise ValueError("--end must be >= 0")
    if args.start > args.end:
        raise ValueError("--start must be <= --end")
    if args.depth_backend in {"ffs", "both"}:
        if not args.ffs_repo:
            raise ValueError("--ffs_repo is required for --depth_backend ffs|both")
        if not args.ffs_model_path:
            raise ValueError("--ffs_model_path is required for --depth_backend ffs|both")
    if bool(_optional_arg(args, "ffs_radius_outlier_filter", False)) and args.depth_backend not in {"ffs", "both"}:
        raise ValueError("--ffs_radius_outlier_filter requires --depth_backend ffs|both")
    if float(_optional_arg(args, "ffs_radius_outlier_radius_m", 0.01)) <= 0.0:
        raise ValueError("--ffs_radius_outlier_radius_m must be > 0")
    if int(_optional_arg(args, "ffs_radius_outlier_nb_points", 40)) <= 0:
        raise ValueError("--ffs_radius_outlier_nb_points must be > 0")
    confidence_mode = str(_optional_arg(args, "ffs_confidence_mode", "none"))
    if confidence_mode not in {"none", "margin", "max_softmax", "entropy", "variance"}:
        raise ValueError(f"Unsupported --ffs_confidence_mode: {confidence_mode}")
    if confidence_mode != "none" and args.depth_backend not in {"ffs", "both"}:
        raise ValueError("--ffs_confidence_mode requires --depth_backend ffs|both")
    if float(_optional_arg(args, "ffs_confidence_threshold", 0.0)) < 0.0:
        raise ValueError("--ffs_confidence_threshold must be >= 0")
    if float(_optional_arg(args, "ffs_confidence_threshold", 0.0)) > 1.0:
        raise ValueError("--ffs_confidence_threshold must be <= 1")
    confidence_depth_min_m = float(_optional_arg(args, "ffs_confidence_depth_min_m", 0.2))
    confidence_depth_max_m = float(_optional_arg(args, "ffs_confidence_depth_max_m", 1.5))
    if confidence_depth_min_m > confidence_depth_max_m:
        raise ValueError("--ffs_confidence_depth_min_m must be <= --ffs_confidence_depth_max_m")
    if confidence_mode == "none" and bool(_optional_arg(args, "write_ffs_confidence_debug", False)):
        raise ValueError("--write_ffs_confidence_debug requires --ffs_confidence_mode != none")
    if confidence_mode == "none" and bool(_optional_arg(args, "write_ffs_valid_mask_debug", False)):
        raise ValueError("--write_ffs_valid_mask_debug requires --ffs_confidence_mode != none")


def _optional_arg(args: Any, name: str, default: Any):
    values = getattr(args, "__dict__", None)
    if isinstance(values, dict) and name in values:
        return values[name]
    return default


def find_ffmpeg() -> str | None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        return ffmpeg_bin

    local_appdata = Path.home() / "AppData" / "Local"
    user_bin = Path.home() / "bin" / "ffmpeg.exe"
    candidates = [user_bin]
    winget_root = local_appdata / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        candidates.extend(winget_root.glob("Gyan.FFmpeg.Essentials_*/*/bin/ffmpeg.exe"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def is_formal_different_types_export(output_case_dir: Path) -> bool:
    return output_case_dir.parent.name == "different_types"


def write_aligned_calibration_file(*, case_dir: Path, output_case_dir: Path, metadata: dict[str, Any]) -> None:
    output_path = output_case_dir / "calibrate.pkl"
    if is_formal_different_types_export(output_case_dir):
        transforms = load_calibration_transforms(
            case_dir / "calibrate.pkl",
            serial_numbers=list(metadata["serial_numbers"]),
            calibration_reference_serials=metadata.get("calibration_reference_serials", metadata["serial_numbers"]),
        )
        with output_path.open("wb") as handle:
            pickle.dump(transforms, handle)
        return
    shutil.copy2(case_dir / "calibrate.pkl", output_path)


def load_metadata(case_dir: Path) -> dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    require_file(metadata_path, "metadata.json")
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    serial_numbers = data.get("serial_numbers")
    recording = data.get("recording")
    if not isinstance(serial_numbers, list) or not serial_numbers:
        raise ValueError(f"metadata.json missing valid serial_numbers: {metadata_path}")
    if not isinstance(recording, dict) or not recording:
        raise ValueError(f"metadata.json missing valid recording map: {metadata_path}")
    return data


def get_camera_count(metadata: dict[str, Any]) -> int:
    return len(metadata["serial_numbers"])


def discover_available_streams(case_dir: Path) -> list[str]:
    streams = []
    for stream_name in ("color", "depth", "ir_left", "ir_right"):
        if (case_dir / stream_name).is_dir():
            streams.append(stream_name)
    return streams


def get_metadata_list(metadata: dict[str, Any], key: str, num_cameras: int, default=None):
    value = metadata.get(key, default)
    if isinstance(value, list):
        return value
    return [value for _ in range(num_cameras)]


def get_color_intrinsics(metadata: dict[str, Any], num_cameras: int):
    if "K_color" in metadata:
        return get_metadata_list(metadata, "K_color", num_cameras)
    return get_metadata_list(metadata, "intrinsics", num_cameras)


def match_frames(recording: dict[str, dict[str, float]], num_cameras: int, start_step: int, end_step: int) -> list[list[int]]:
    if "0" not in recording:
        raise ValueError("recording metadata does not contain camera 0")

    final_frames: list[list[int]] = []
    for step_str, timestamp in recording["0"].items():
        step_idx = int(step_str)
        if step_idx < start_step or step_idx > end_step:
            continue
        current_frame = [step_idx]
        valid_match = True
        for camera_idx in range(1, num_cameras):
            camera_key = str(camera_idx)
            if camera_key not in recording:
                raise ValueError(f"recording metadata missing camera {camera_idx}")
            min_diff = 10.0
            best_idx: int | None = None
            for offset in range(-3, 4):
                candidate = step_idx + offset
                candidate_key = str(candidate)
                if candidate_key not in recording[camera_key]:
                    continue
                diff = abs(float(recording[camera_key][candidate_key]) - float(timestamp))
                if diff < min_diff:
                    min_diff = diff
                    best_idx = candidate
            if best_idx is None:
                valid_match = False
                break
            current_frame.append(best_idx)
        if valid_match:
            final_frames.append(current_frame)
    return final_frames


def prepare_output_case(output_case_dir: Path, num_cameras: int, stream_dirs: list[str]) -> None:
    if output_case_dir.exists():
        shutil.rmtree(output_case_dir)
    for stream_name in stream_dirs:
        (output_case_dir / stream_name).mkdir(parents=True, exist_ok=True)
        if stream_name.endswith("_float_m"):
            continue
        for camera_idx in range(num_cameras):
            (output_case_dir / stream_name / str(camera_idx)).mkdir(parents=True, exist_ok=True)
    for stream_name in stream_dirs:
        if stream_name.endswith("_float_m"):
            for camera_idx in range(num_cameras):
                (output_case_dir / stream_name / str(camera_idx)).mkdir(parents=True, exist_ok=True)


def write_mp4s(ffmpeg_bin: str, output_case_dir: Path, num_cameras: int, fps: int) -> None:
    for camera_idx in range(num_cameras):
        cmd = [
            ffmpeg_bin,
            "-y",
            "-r",
            str(fps),
            "-start_number",
            "0",
            "-f",
            "image2",
            "-i",
            str(output_case_dir / "color" / str(camera_idx) / "%d.png"),
            "-vcodec",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_case_dir / "color" / f"{camera_idx}.mp4"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def should_write_color_mp4_sidecars(*, output_case_dir: Path, explicit_write_mp4: bool) -> bool:
    if bool(explicit_write_mp4):
        return True
    return output_case_dir.parent.name == "different_types"


def copy_image_stream(case_dir: Path, output_case_dir: Path, stream_name: str, camera_idx: int, src_step: int, dst_idx: int) -> None:
    src_path = case_dir / stream_name / str(camera_idx) / f"{src_step}.png"
    require_file(src_path, f"{stream_name} frame camera={camera_idx} step={src_step}")
    shutil.copy2(src_path, output_case_dir / stream_name / str(camera_idx) / f"{dst_idx}.png")


def copy_depth_stream(case_dir: Path, output_case_dir: Path, stream_name: str, camera_idx: int, src_step: int, dst_idx: int) -> None:
    src_path = case_dir / stream_name / str(camera_idx) / f"{src_step}.npy"
    require_file(src_path, f"{stream_name} frame camera={camera_idx} step={src_step}")
    shutil.copy2(src_path, output_case_dir / stream_name / str(camera_idx) / f"{dst_idx}.npy")


def require_ffs_geometry(metadata: dict[str, Any], num_cameras: int) -> None:
    required_keys = ("K_ir_left", "T_ir_left_to_color", "ir_baseline_m")
    for key in required_keys:
        values = get_metadata_list(metadata, key, num_cameras)
        if any(value is None for value in values):
            raise ValueError(f"FFS backend requires metadata field {key}")
    if any(value is None for value in get_color_intrinsics(metadata, num_cameras)):
        raise ValueError("FFS backend requires color intrinsics metadata")
    if any(value is None for value in get_metadata_list(metadata, "depth_scale_m_per_unit", num_cameras)):
        raise ValueError("FFS backend requires depth_scale_m_per_unit for compatibility encoding")


def _resolve_ffs_stream_dirs(
    *,
    depth_backend: str,
    write_ffs_float_m: bool,
    ffs_native_like_postprocess_enabled: bool,
    ffs_radius_outlier_filter_enabled: bool,
    ffs_confidence_mode: str,
    write_ffs_confidence_debug: bool,
    write_ffs_valid_mask_debug: bool,
    archive_u16_dir: str,
    archive_float_dir: str,
) -> list[str]:
    streams_to_write = ["color"]
    if depth_backend == "ffs":
        streams_to_write.append("depth")
        if write_ffs_float_m:
            streams_to_write.append("depth_ffs_float_m")
    elif depth_backend == "both":
        streams_to_write.extend(["depth", "depth_ffs"])
        if write_ffs_float_m:
            streams_to_write.append("depth_ffs_float_m")
    else:
        streams_to_write.append("depth")
    if ffs_radius_outlier_filter_enabled:
        streams_to_write.append(archive_u16_dir)
        if write_ffs_float_m:
            streams_to_write.append(archive_float_dir)
    if depth_backend in {"ffs", "both"} and ffs_confidence_mode != "none":
        if write_ffs_confidence_debug:
            streams_to_write.append("confidence_ffs")
        if write_ffs_valid_mask_debug:
            streams_to_write.append("valid_mask_ffs")
    if depth_backend in {"ffs", "both"} and ffs_native_like_postprocess_enabled:
        streams_to_write.extend(["depth_ffs_native_like_postprocess", "depth_ffs_native_like_postprocess_float_m"])
    return streams_to_write


def summarize_numeric_stats(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {"sample_count": 0.0}
    keys = sorted({key for sample in samples for key in sample})
    summary: dict[str, float] = {"sample_count": float(len(samples))}
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if not values:
            continue
        summary[f"{key}_mean"] = float(sum(values) / len(values))
        summary[f"{key}_min"] = float(min(values))
        summary[f"{key}_max"] = float(max(values))
    return summary


def align_case(args: Any, runner_factory=None) -> dict[str, Any]:
    validate_args(args)

    base_path = args.base_path.resolve()
    output_root = args.output_path.resolve()
    case_dir = base_path / args.case_name
    output_case_dir = output_root / args.case_name

    require_file(case_dir / "calibrate.pkl", "calibrate.pkl")
    metadata = load_metadata(case_dir)
    num_cameras = get_camera_count(metadata)
    available_streams = discover_available_streams(case_dir)
    fps = args.fps if args.fps is not None else int(metadata.get("fps", 30))
    final_frames = match_frames(metadata["recording"], num_cameras, args.start, args.end)
    ffs_native_like_postprocess_enabled = bool(_optional_arg(args, "ffs_native_like_postprocess", False))
    ffs_radius_outlier_filter_enabled = bool(_optional_arg(args, "ffs_radius_outlier_filter", False))
    ffs_radius_outlier_radius_m = float(_optional_arg(args, "ffs_radius_outlier_radius_m", 0.01))
    ffs_radius_outlier_nb_points = int(_optional_arg(args, "ffs_radius_outlier_nb_points", 40))
    ffs_confidence_mode = str(_optional_arg(args, "ffs_confidence_mode", "none"))
    ffs_confidence_threshold = float(_optional_arg(args, "ffs_confidence_threshold", 0.0))
    ffs_confidence_depth_min_m = float(_optional_arg(args, "ffs_confidence_depth_min_m", 0.2))
    ffs_confidence_depth_max_m = float(_optional_arg(args, "ffs_confidence_depth_max_m", 1.5))
    write_ffs_confidence_debug = bool(_optional_arg(args, "write_ffs_confidence_debug", False))
    write_ffs_valid_mask_debug = bool(_optional_arg(args, "write_ffs_valid_mask_debug", False))
    if not final_frames:
        raise RuntimeError(f"No aligned frames found for case={args.case_name} in range {args.start} -> {args.end}")

    if args.fail_if_no_ir_stereo and not {"ir_left", "ir_right"}.issubset(set(available_streams)):
        raise RuntimeError("Raw case does not contain ir_left/ir_right but --fail_if_no_ir_stereo was requested")

    if args.depth_backend in {"realsense", "both"} and "depth" not in available_streams:
        raise RuntimeError(f"Raw case {args.case_name} does not contain depth data required for backend={args.depth_backend}")
    if args.depth_backend in {"ffs", "both"}:
        if not {"ir_left", "ir_right"}.issubset(set(available_streams)):
            raise RuntimeError(f"Raw case {args.case_name} does not contain ir_left/ir_right required for backend={args.depth_backend}")
        require_ffs_geometry(metadata, num_cameras)

    streams_to_write = ["color"]
    for optional_stream in ("ir_left", "ir_right"):
        if optional_stream in available_streams:
            streams_to_write.append(optional_stream)
    if args.depth_backend in {"ffs", "both"}:
        from data_process.depth_backends import (
            FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND,
            FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND,
            FFS_FLOAT_ARCHIVE_DIR,
        )

        archive_u16_dir = (
            FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND
            if args.depth_backend == "ffs"
            else FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND
        )
        archive_float_dir = FFS_FLOAT_ARCHIVE_DIR
        streams_to_write.extend(
            _resolve_ffs_stream_dirs(
                depth_backend=args.depth_backend,
                write_ffs_float_m=bool(args.write_ffs_float_m),
                ffs_native_like_postprocess_enabled=ffs_native_like_postprocess_enabled,
                ffs_radius_outlier_filter_enabled=ffs_radius_outlier_filter_enabled,
                ffs_confidence_mode=ffs_confidence_mode,
                write_ffs_confidence_debug=write_ffs_confidence_debug,
                write_ffs_valid_mask_debug=write_ffs_valid_mask_debug,
                archive_u16_dir=archive_u16_dir,
                archive_float_dir=archive_float_dir,
            )[1:]
        )
    else:
        streams_to_write.append("depth")

    print(f"[align] base_path={base_path}")
    print(f"[align] case_name={args.case_name}")
    print(f"[align] output_path={output_root}")
    print(f"[align] step range: {args.start} -> {args.end}")
    print(f"[align] num_cameras={num_cameras}")
    print(f"[align] matched frames={len(final_frames)}")
    print(f"[align] depth_backend={args.depth_backend}")
    print(f"[align] ffs_native_like_postprocess={ffs_native_like_postprocess_enabled}")
    print(
        "[align] "
        f"ffs_confidence_mode={ffs_confidence_mode} "
        f"threshold={ffs_confidence_threshold:.3f} "
        f"depth_range_m={ffs_confidence_depth_min_m:.3f}->{ffs_confidence_depth_max_m:.3f}"
    )
    print(
        "[align] "
        f"ffs_radius_outlier_filter={ffs_radius_outlier_filter_enabled} "
        f"radius_m={ffs_radius_outlier_radius_m:.4f} "
        f"nb_points={ffs_radius_outlier_nb_points}"
    )

    prepare_output_case(output_case_dir, num_cameras, streams_to_write)
    write_aligned_calibration_file(
        case_dir=case_dir,
        output_case_dir=output_case_dir,
        metadata=metadata,
    )

    color_intrinsics = get_color_intrinsics(metadata, num_cameras)
    aligned_metadata = {
        "schema_version": "qqtt_aligned_case_v2",
        "source_case_name": args.case_name,
        "serial_numbers": metadata["serial_numbers"],
        "calibration_reference_serials": metadata.get("calibration_reference_serials", metadata["serial_numbers"]),
        "logical_camera_names": metadata.get("logical_camera_names", [f"cam{i}" for i in range(num_cameras)]),
        "fps": fps,
        "WH": metadata["WH"],
        "frame_num": len(final_frames),
        "start_step": args.start,
        "end_step": args.end,
        "capture_mode": metadata.get("capture_mode", "rgbd"),
        "streams_present": streams_to_write,
        "depth_backend_used": args.depth_backend,
        "depth_source_for_depth_dir": "realsense" if args.depth_backend != "ffs" else "ffs",
        "ffs_native_like_postprocess_enabled": ffs_native_like_postprocess_enabled,
        "ffs_radius_outlier_filter_enabled": ffs_radius_outlier_filter_enabled,
        "ffs_radius_outlier_filter": {
            "mode": "per_camera_color_frame_radius_outlier",
            "radius_m": ffs_radius_outlier_radius_m,
            "nb_points": ffs_radius_outlier_nb_points,
            "archive_policy": "replace_main_and_archive_raw",
        },
        "ffs_confidence_filter": {
            "enabled": ffs_confidence_mode != "none",
            "mode": ffs_confidence_mode,
            "threshold": ffs_confidence_threshold,
            "depth_min_m": ffs_confidence_depth_min_m,
            "depth_max_m": ffs_confidence_depth_max_m,
            "debug_confidence_uint8": write_ffs_confidence_debug,
            "debug_valid_mask_uint8": write_ffs_valid_mask_debug,
            "depth_output_encoding": "uint16_depth_scale_invalid_zero",
        },
        "depth_scale_m_per_unit": get_metadata_list(metadata, "depth_scale_m_per_unit", num_cameras),
        "depth_encoding": metadata.get("depth_encoding") or "uint16_meters_scaled_invalid_zero",
        "intrinsics": color_intrinsics,
        "K_color": color_intrinsics,
        "K_ir_left": get_metadata_list(metadata, "K_ir_left", num_cameras),
        "K_ir_right": get_metadata_list(metadata, "K_ir_right", num_cameras),
        "T_ir_left_to_right": get_metadata_list(metadata, "T_ir_left_to_right", num_cameras),
        "T_ir_left_to_color": get_metadata_list(metadata, "T_ir_left_to_color", num_cameras),
        "ir_baseline_m": get_metadata_list(metadata, "ir_baseline_m", num_cameras),
        "source_streams_present": metadata.get("streams_present", available_streams),
    }

    runner = None
    if args.depth_backend in {"ffs", "both"}:
        from data_process.depth_backends import (
            FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND,
            FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND,
            FFS_FLOAT_ARCHIVE_DIR,
            FastFoundationStereoRunner,
            align_depth_to_color,
            align_ir_scalar_to_color,
            apply_ffs_radius_outlier_filter_u16,
            build_confidence_filtered_depth_uint16,
            quantize_depth_with_invalid_zero,
        )
        from qqtt.env.camera.realsense.depth_postprocess import (
            FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_DIR,
            FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_FLOAT_DIR,
            apply_ffs_native_like_depth_postprocess_float_m,
        )
        archive_u16_dir = (
            FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND
            if args.depth_backend == "ffs"
            else FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND
        )
        archive_float_dir = FFS_FLOAT_ARCHIVE_DIR
        ffs_main_u16_dir = "depth" if args.depth_backend == "ffs" else "depth_ffs"
        ffs_main_float_dir = "depth_ffs_float_m"

        # This is the production entrypoint for FFS inside the repo.
        # We load the external Fast-FoundationStereo model once here,
        # then reuse it for every aligned frame/camera pair below.
        runner_cls = runner_factory or FastFoundationStereoRunner
        runner = runner_cls(
            ffs_repo=args.ffs_repo,
            model_path=args.ffs_model_path,
            scale=args.ffs_scale,
            valid_iters=args.ffs_valid_iters,
            max_disp=args.ffs_max_disp,
        )
        aligned_metadata["ffs_config"] = {
            "ffs_repo": args.ffs_repo,
            "model_path": args.ffs_model_path,
            "scale": args.ffs_scale,
            "valid_iters": args.ffs_valid_iters,
            "max_disp": args.ffs_max_disp,
        }
    else:
        align_depth_to_color = None

    import cv2
    import numpy as np

    confidence_filter_stats_samples: list[dict[str, float]] = []
    for output_idx, frame in enumerate(final_frames):
        if output_idx % 100 == 0:
            print(f"[align] processing frame {output_idx}/{len(final_frames)}")
        for camera_idx in range(num_cameras):
            src_step = frame[camera_idx]

            copy_image_stream(case_dir, output_case_dir, "color", camera_idx, src_step, output_idx)
            if "ir_left" in available_streams:
                copy_image_stream(case_dir, output_case_dir, "ir_left", camera_idx, src_step, output_idx)
            if "ir_right" in available_streams:
                copy_image_stream(case_dir, output_case_dir, "ir_right", camera_idx, src_step, output_idx)

            if args.depth_backend in {"realsense", "both"}:
                copy_depth_stream(case_dir, output_case_dir, "depth", camera_idx, src_step, output_idx)

            if args.depth_backend in {"ffs", "both"}:
                left_image = cv2.imread(str(case_dir / "ir_left" / str(camera_idx) / f"{src_step}.png"), cv2.IMREAD_UNCHANGED)
                right_image = cv2.imread(str(case_dir / "ir_right" / str(camera_idx) / f"{src_step}.png"), cv2.IMREAD_UNCHANGED)
                if left_image is None or right_image is None:
                    raise RuntimeError(f"Failed to load IR stereo pair for camera={camera_idx} step={src_step}")

                # This is the actual FFS call: feed this camera's own
                # IR-left / IR-right pair into the external model and get
                # disparity + IR-left metric depth back.
                runner_kwargs = {
                    "K_ir_left": np.asarray(aligned_metadata["K_ir_left"][camera_idx], dtype=np.float32),
                    "baseline_m": float(aligned_metadata["ir_baseline_m"][camera_idx]),
                }
                if ffs_confidence_mode != "none":
                    ffs_output = runner.run_pair_with_confidence(left_image, right_image, **runner_kwargs)
                else:
                    ffs_output = runner.run_pair(left_image, right_image, **runner_kwargs)
                depth_ir = np.asarray(ffs_output["depth_ir_left_m"], dtype=np.float32)
                K_ir_left_used = np.asarray(ffs_output["K_ir_left_used"], dtype=np.float32)
                T_ir_left_to_color = np.asarray(aligned_metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
                K_color = np.asarray(aligned_metadata["K_color"][camera_idx], dtype=np.float32)
                output_shape = (int(metadata["WH"][1]), int(metadata["WH"][0]))
                depth_color = align_depth_to_color(
                    depth_ir,
                    K_ir_left_used,
                    T_ir_left_to_color,
                    K_color,
                    output_shape=output_shape,
                )
                scale_m = float(aligned_metadata["depth_scale_m_per_unit"][camera_idx])
                filtered_depth_u16, filtered_depth_m, _filter_stats = apply_ffs_radius_outlier_filter_u16(
                    depth_color,
                    K_color=K_color,
                    depth_scale_m_per_unit=scale_m,
                    radius_m=ffs_radius_outlier_radius_m,
                    nb_points=ffs_radius_outlier_nb_points,
                ) if ffs_radius_outlier_filter_enabled else (
                    quantize_depth_with_invalid_zero(depth_color, scale_m),
                    np.asarray(depth_color, dtype=np.float32),
                    {},
                )
                filtered_depth_u16[~(np.isfinite(filtered_depth_m) & (filtered_depth_m > 0))] = 0

                if ffs_confidence_mode != "none":
                    confidence_ir = np.asarray(
                        ffs_output[f"confidence_{ffs_confidence_mode}_ir_left"],
                        dtype=np.float32,
                    )
                    confidence_color = align_ir_scalar_to_color(
                        depth_ir,
                        confidence_ir,
                        K_ir_left_used,
                        T_ir_left_to_color,
                        K_color,
                        output_shape=output_shape,
                        invalid_value=0.0,
                    )
                    confidence_result = build_confidence_filtered_depth_uint16(
                        depth_m=filtered_depth_m,
                        confidence=confidence_color,
                        confidence_threshold=ffs_confidence_threshold,
                        depth_scale_m_per_unit=scale_m,
                        depth_min_m=ffs_confidence_depth_min_m,
                        depth_max_m=ffs_confidence_depth_max_m,
                    )
                    filtered_depth_u16 = np.asarray(confidence_result["depth_uint16"], dtype=np.uint16)
                    valid_mask_uint8 = np.asarray(confidence_result["valid_mask_uint8"], dtype=np.uint8)
                    confidence_uint8 = np.asarray(confidence_result["confidence_uint8"], dtype=np.uint8)
                    confidence_filter_stats_samples.append(
                        {
                            str(key): float(value)
                            for key, value in dict(confidence_result["stats"]).items()
                        }
                    )
                    filtered_depth_m = np.where(valid_mask_uint8.astype(bool), filtered_depth_m, 0.0).astype(np.float32)
                    if write_ffs_confidence_debug:
                        cv2.imwrite(
                            str(output_case_dir / "confidence_ffs" / str(camera_idx) / f"{output_idx}.png"),
                            confidence_uint8,
                        )
                    if write_ffs_valid_mask_debug:
                        cv2.imwrite(
                            str(output_case_dir / "valid_mask_ffs" / str(camera_idx) / f"{output_idx}.png"),
                            (valid_mask_uint8 * 255).astype(np.uint8),
                        )

                np.save(output_case_dir / ffs_main_u16_dir / str(camera_idx) / f"{output_idx}.npy", filtered_depth_u16)

                if ffs_radius_outlier_filter_enabled:
                    raw_depth_u16 = quantize_depth_with_invalid_zero(depth_color, scale_m)
                    np.save(output_case_dir / archive_u16_dir / str(camera_idx) / f"{output_idx}.npy", raw_depth_u16)

                if args.write_ffs_float_m:
                    np.save(output_case_dir / ffs_main_float_dir / str(camera_idx) / f"{output_idx}.npy", filtered_depth_m)
                    if ffs_radius_outlier_filter_enabled:
                        np.save(output_case_dir / archive_float_dir / str(camera_idx) / f"{output_idx}.npy", depth_color.astype(np.float32))

                if ffs_native_like_postprocess_enabled:
                    filtered_depth_u16, filtered_depth_m = apply_ffs_native_like_depth_postprocess_float_m(
                        filtered_depth_m,
                        depth_scale_m_per_unit=scale_m,
                        fps=int(fps),
                        frame_number=int(output_idx) + 1,
                    )
                    np.save(output_case_dir / FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_DIR / str(camera_idx) / f"{output_idx}.npy", filtered_depth_u16)
                    np.save(output_case_dir / FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_FLOAT_DIR / str(camera_idx) / f"{output_idx}.npy", filtered_depth_m)

    if ffs_confidence_mode != "none":
        aligned_metadata["ffs_confidence_filter"]["stats_summary"] = summarize_numeric_stats(confidence_filter_stats_samples)

    write_split_aligned_metadata(output_case_dir, aligned_metadata)

    write_color_mp4_sidecars = should_write_color_mp4_sidecars(
        output_case_dir=output_case_dir,
        explicit_write_mp4=bool(args.write_mp4),
    )
    if write_color_mp4_sidecars:
        ffmpeg_bin = find_ffmpeg()
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg not found but color mp4 sidecars are required for this aligned export")
        print(f"[align] ffmpeg={ffmpeg_bin}")
        write_mp4s(ffmpeg_bin, output_case_dir, num_cameras, fps)

    print("[align] done.")
    return aligned_metadata


def main() -> int:
    args = build_parser().parse_args()
    align_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
