from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "different_types"
REQUIRED_TOP_LEVEL_DIRS = ("color", "depth")
REQUIRED_TOP_LEVEL_FILES = ("calibrate.pkl", "metadata.json")
REQUIRED_CAMERA_DIRS = ("0", "1", "2")
OPTIONAL_COLOR_MP4_FILES = tuple(f"{camera_idx}.mp4" for camera_idx in REQUIRED_CAMERA_DIRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute cleanup for formal downstream cases under data/different_types/."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case_name", action="append", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply deletions in place. Default is dry-run and prints what would be removed.",
    )
    return parser.parse_args()


def _resolved_within_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _find_ffmpeg() -> str | None:
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


def _load_case_metadata(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))


def _generate_missing_color_mp4s(*, case_dir: Path, camera_ids: list[str], fps: int, ffmpeg_bin: str) -> list[str]:
    generated_paths: list[str] = []
    for camera_id in camera_ids:
        frame_dir = case_dir / "color" / camera_id
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"Missing color frame directory for camera {camera_id}: {frame_dir}")
        if not any(frame_dir.glob("*.png")):
            raise RuntimeError(f"Cannot generate color/{camera_id}.mp4 because {frame_dir} contains no PNG frames.")
        output_path = case_dir / "color" / f"{camera_id}.mp4"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-r",
            str(int(fps)),
            "-start_number",
            "0",
            "-f",
            "image2",
            "-i",
            str(frame_dir / "%d.png"),
            "-vcodec",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        generated_paths.append(str(output_path.resolve()))
    return generated_paths


def _inspect_case(case_dir: Path) -> dict[str, Any]:
    root_items = sorted(case_dir.iterdir(), key=lambda item: item.name)
    keep_paths: list[str] = []
    delete_paths: list[str] = []
    generate_paths: list[str] = []
    errors: list[str] = []

    for required_dir in REQUIRED_TOP_LEVEL_DIRS:
        path = case_dir / required_dir
        if not path.exists():
            errors.append(f"Missing required directory: {required_dir}")
        elif not path.is_dir():
            errors.append(f"Required path is not a directory: {required_dir}")
        else:
            keep_paths.append(str(path))
            for camera_dir in REQUIRED_CAMERA_DIRS:
                child = path / camera_dir
                if not child.exists():
                    errors.append(f"Missing required camera directory: {required_dir}/{camera_dir}")
                elif not child.is_dir():
                    errors.append(f"Required camera path is not a directory: {required_dir}/{camera_dir}")
                else:
                    keep_paths.append(str(child))
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if child.name in REQUIRED_CAMERA_DIRS:
                    continue
                if required_dir == "color" and child.name in OPTIONAL_COLOR_MP4_FILES:
                    if not child.is_file():
                        errors.append(f"Optional color mp4 path is not a file: {required_dir}/{child.name}")
                    else:
                        keep_paths.append(str(child))
                    continue
                if child.name not in REQUIRED_CAMERA_DIRS:
                    delete_paths.append(str(child))
            if required_dir == "color":
                for mp4_name in OPTIONAL_COLOR_MP4_FILES:
                    mp4_path = path / mp4_name
                    if not mp4_path.exists():
                        generate_paths.append(str(mp4_path))

    for required_file in REQUIRED_TOP_LEVEL_FILES:
        path = case_dir / required_file
        if not path.exists():
            errors.append(f"Missing required file: {required_file}")
        elif not path.is_file():
            errors.append(f"Required path is not a file: {required_file}")
        else:
            keep_paths.append(str(path))

    allowed_top_level = set(REQUIRED_TOP_LEVEL_DIRS) | set(REQUIRED_TOP_LEVEL_FILES)
    for item in root_items:
        if item.name not in allowed_top_level:
            delete_paths.append(str(item))

    seen_delete = sorted(dict.fromkeys(delete_paths))
    seen_keep = sorted(dict.fromkeys(keep_paths))
    return {
        "case_name": case_dir.name,
        "case_dir": str(case_dir.resolve()),
        "kept_paths": seen_keep,
        "delete_paths": seen_delete,
        "generate_paths": sorted(dict.fromkeys(generate_paths)),
        "errors": errors,
        "status": "error" if errors else "ready",
    }


def cleanup_cases(*, root: Path, case_names: list[str] | None, execute: bool) -> dict[str, Any]:
    root = Path(root).resolve()
    cases: list[Path] = []
    errors: list[str] = []
    if not root.exists():
        raise FileNotFoundError(f"Cleanup root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Cleanup root is not a directory: {root}")

    if case_names:
        for case_name in case_names:
            candidate = root / case_name
            if not candidate.exists():
                errors.append(f"Case not found: {case_name}")
                continue
            if not candidate.is_dir():
                errors.append(f"Case path is not a directory: {case_name}")
                continue
            cases.append(candidate)
    else:
        cases = sorted([item for item in root.iterdir() if item.is_dir()], key=lambda item: item.name)

    case_summaries: list[dict[str, Any]] = []
    any_case_errors = bool(errors)
    for case_dir in cases:
        summary = _inspect_case(case_dir)
        deleted_paths: list[str] = []
        generated_paths: list[str] = []
        if execute and not summary["errors"]:
            if summary["generate_paths"]:
                ffmpeg_bin = _find_ffmpeg()
                if ffmpeg_bin is None:
                    summary["errors"].append("Missing ffmpeg required to generate color/*.mp4 sidecars.")
                else:
                    metadata = _load_case_metadata(case_dir)
                    generated_paths = _generate_missing_color_mp4s(
                        case_dir=case_dir,
                        camera_ids=[Path(path_str).stem for path_str in summary["generate_paths"]],
                        fps=int(metadata.get("fps", 30)),
                        ffmpeg_bin=ffmpeg_bin,
                    )
                    summary["kept_paths"] = sorted(dict.fromkeys([*summary["kept_paths"], *generated_paths]))
            if not summary["errors"]:
                for path_str in summary["delete_paths"]:
                    target = _resolved_within_root(Path(path_str), root)
                    _remove_path(target)
                    deleted_paths.append(str(target))
        summary["deleted_paths"] = deleted_paths
        summary["generated_paths"] = generated_paths
        summary["mode"] = "execute" if execute else "dry-run"
        if execute and summary["errors"]:
            summary["status"] = "error"
        elif execute:
            summary["status"] = "executed"
        case_summaries.append(summary)
        any_case_errors = any_case_errors or bool(summary["errors"])

    result = {
        "root": str(root),
        "mode": "execute" if execute else "dry-run",
        "case_filter": list(case_names) if case_names else None,
        "errors": errors,
        "cases": case_summaries,
    }
    if any_case_errors:
        result["status"] = "error"
    elif execute:
        result["status"] = "executed"
    else:
        result["status"] = "ready"
    return result


def main() -> int:
    args = parse_args()
    result = cleanup_cases(
        root=args.root,
        case_names=list(args.case_name) if args.case_name else None,
        execute=bool(args.execute),
    )
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
