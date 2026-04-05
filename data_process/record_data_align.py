"""Trim and align a raw multi-camera recording into an aligned case."""

from __future__ import annotations

import json
import shutil
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any


_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


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


def prepare_output_case(output_case_dir: Path, num_cameras: int) -> None:
    if output_case_dir.exists():
        shutil.rmtree(output_case_dir)
    (output_case_dir / "color").mkdir(parents=True, exist_ok=True)
    (output_case_dir / "depth").mkdir(parents=True, exist_ok=True)
    for camera_idx in range(num_cameras):
        (output_case_dir / "color" / str(camera_idx)).mkdir(parents=True, exist_ok=True)
        (output_case_dir / "depth" / str(camera_idx)).mkdir(parents=True, exist_ok=True)


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


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    base_path = args.base_path.resolve()
    output_root = args.output_path.resolve()
    case_dir = base_path / args.case_name
    output_case_dir = output_root / args.case_name

    require_file(case_dir / "calibrate.pkl", "calibrate.pkl")
    metadata = load_metadata(case_dir)

    num_cameras = len(metadata["serial_numbers"])
    fps = args.fps if args.fps is not None else int(metadata.get("fps", 30))
    final_frames = match_frames(metadata["recording"], num_cameras, args.start, args.end)
    if not final_frames:
        raise RuntimeError(
            f"No aligned frames found for case={args.case_name} in range {args.start} -> {args.end}"
        )

    print(f"[align] base_path={base_path}")
    print(f"[align] case_name={args.case_name}")
    print(f"[align] output_path={output_root}")
    print(f"[align] step range: {args.start} -> {args.end}")
    print(f"[align] num_cameras={num_cameras}")
    print(f"[align] matched frames={len(final_frames)}")

    prepare_output_case(output_case_dir, num_cameras)
    shutil.copy2(case_dir / "calibrate.pkl", output_case_dir / "calibrate.pkl")

    output_metadata = {
        "intrinsics": metadata["intrinsics"],
        "serial_numbers": metadata["serial_numbers"],
        "fps": fps,
        "WH": metadata["WH"],
        "frame_num": len(final_frames),
        "start_step": args.start,
        "end_step": args.end,
        "source_case_name": args.case_name,
    }
    with (output_case_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(output_metadata, f)

    for output_idx, frame in enumerate(final_frames):
        if output_idx % 100 == 0:
            print(f"[align] copying frame {output_idx}/{len(final_frames)}")
        for camera_idx in range(num_cameras):
            src_color = case_dir / "color" / str(camera_idx) / f"{frame[camera_idx]}.png"
            src_depth = case_dir / "depth" / str(camera_idx) / f"{frame[camera_idx]}.npy"
            require_file(src_color, f"color frame camera={camera_idx} step={frame[camera_idx]}")
            require_file(src_depth, f"depth frame camera={camera_idx} step={frame[camera_idx]}")
            dst_color = output_case_dir / "color" / str(camera_idx) / f"{output_idx}.png"
            dst_depth = output_case_dir / "depth" / str(camera_idx) / f"{output_idx}.npy"
            shutil.copy2(src_color, dst_color)
            shutil.copy2(src_depth, dst_depth)

    if args.write_mp4:
        ffmpeg_bin = find_ffmpeg()
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg not found but --write_mp4 was requested")
        print(f"[align] ffmpeg={ffmpeg_bin}")
        write_mp4s(ffmpeg_bin, output_case_dir, num_cameras, fps)

    print("[align] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
