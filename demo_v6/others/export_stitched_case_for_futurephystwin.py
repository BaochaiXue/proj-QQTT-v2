#!/usr/bin/env python3
"""Export the stitched Demo v6 online result as a FuturePhysTwin case.

Takes the cross-chunk stitched tracking result (``outputs/data/final_data.pkl``),
the single-camera table calibration (``table_calibrate.pkl``), the capture
metadata (``outputs/capture/metadata.json``), and the per-frame input color
images, and materializes a case directory laid out the way
``~/FuturePhysTwin/train_warp.py`` expects:

    <case_dir>/final_data.pkl     copied stitched final data
    <case_dir>/calibrate.pkl      list with the single c2w 4x4 matrix
    <case_dir>/metadata.json      intrinsics / WH / fps for visualization+timing
    <case_dir>/split.json         {"frame_len": N, "train": [0, T], "test": [T, N]}
    <case_dir>/color/0/<i>.png    per-frame color overlay images

It also seeds ``FuturePhysTwin/experiments_optimization/<case>/optimal_params.pkl``
by copying the zero-order result of a donor case (no zero-order optimization was
run for demo_v6, so the first-order stage reuses the sloth demo_v4 params).
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_DATA_PATH = REPO_ROOT / "outputs/data/final_data.pkl"
DEFAULT_ONLINE_MANIFEST_PATH = REPO_ROOT / "outputs/online_data/manifest.json"
DEFAULT_TABLE_CALIBRATE_PATH = REPO_ROOT / "table_calibrate.pkl"
DEFAULT_CAPTURE_DIR = REPO_ROOT / "outputs/capture"
DEFAULT_FUTUREPHYSTWIN_ROOT = Path.home() / "FuturePhysTwin"
DEFAULT_CASE_NAME = "demo_v6_stitched_805"
DEFAULT_CASES_ROOT = REPO_ROOT / "demo_v6/others/futurephystwin_stitched/cases"
DEFAULT_DONOR_CASE = "demo_v4_native_single_gpu_unlimited_chunk_0028"
TRAIN_SPLIT_RATIO = 0.7
REQUIRED_FINAL_DATA_KEYS = (
    "object_points",
    "object_colors",
    "object_visibilities",
    "object_motions_valid",
    "controller_points",
    "surface_points",
    "interior_points",
)
REQUIRED_OPTIMAL_PARAM_KEYS = (
    "global_spring_Y",
    "object_radius",
    "object_max_neighbours",
    "controller_radius",
    "controller_max_neighbours",
    "collide_elas",
    "collide_fric",
    "collide_object_elas",
    "collide_object_fric",
    "collision_dist",
    "drag_damping",
    "dashpot_damping",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Export the stitched Demo v6 final_data as a FuturePhysTwin case "
            "and seed its zero-order optimal_params from a donor case."
        )
    )
    parser.add_argument("--final-data-path", type=Path, default=DEFAULT_FINAL_DATA_PATH)
    parser.add_argument(
        "--online-manifest-path", type=Path, default=DEFAULT_ONLINE_MANIFEST_PATH
    )
    parser.add_argument(
        "--table-calibrate-path", type=Path, default=DEFAULT_TABLE_CALIBRATE_PATH
    )
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--futurephystwin-root", type=Path, default=DEFAULT_FUTUREPHYSTWIN_ROOT
    )
    parser.add_argument("--case-name", type=str, default=DEFAULT_CASE_NAME)
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument(
        "--donor-case",
        type=str,
        default=DEFAULT_DONOR_CASE,
        help=(
            "Existing FuturePhysTwin experiments_optimization case whose "
            "optimal_params.pkl seeds the first-order stage."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing case directory / optimal_params seed.",
    )
    return parser


def _load_pickle(path: Path) -> Any:
    """Load pickle."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: type | None = None,
) -> np.ndarray:
    """Return validated array."""
    array = np.asarray(value) if dtype is None else np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def load_and_validate_final_data(path: Path, *, expected_frames: int) -> dict[str, Any]:
    """Load and validate final data."""
    final_data = dict(_load_pickle(path))
    missing = [key for key in REQUIRED_FINAL_DATA_KEYS if key not in final_data]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    object_points = np.asarray(final_data["object_points"], dtype=np.float64)
    if object_points.ndim != 3 or object_points.shape[2] != 3:
        raise ValueError(
            f"object_points must have shape (frames, points, 3), got {object_points.shape}"
        )
    frame_count, object_count = object_points.shape[:2]
    if frame_count != expected_frames:
        raise ValueError(
            f"final_data has {frame_count} frames, online manifest expects {expected_frames}"
        )
    controller_points = np.asarray(final_data["controller_points"], dtype=np.float64)
    if controller_points.ndim != 3 or controller_points.shape[2] != 3:
        raise ValueError(
            f"controller_points must have shape (frames, points, 3), "
            f"got {controller_points.shape}"
        )
    if controller_points.shape[0] != frame_count:
        raise ValueError("controller_points frame count mismatch")

    _require_array(
        final_data["object_colors"],
        name="object_colors",
        shape=(frame_count, object_count, 3),
    )
    _require_array(
        final_data["object_visibilities"],
        name="object_visibilities",
        shape=(frame_count, object_count),
        dtype=bool,
    )
    _require_array(
        final_data["object_motions_valid"],
        name="object_motions_valid",
        shape=(frame_count, object_count),
        dtype=bool,
    )
    for key in ("surface_points", "interior_points"):
        points = np.asarray(final_data[key], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"{key} must have shape (n, 3) with n > 0, got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError(f"{key} contains non-finite values")
    for name, points in (
        ("object_points", object_points),
        ("controller_points", controller_points),
    ):
        if not np.isfinite(points).all():
            raise ValueError(f"{name} contains non-finite values")
    return final_data


def load_c2ws(path: Path) -> list[np.ndarray]:
    """Load c2ws."""
    raw = _load_pickle(path)
    c2ws = [np.asarray(matrix, dtype=np.float64) for matrix in raw]
    if len(c2ws) != 1:
        raise ValueError(f"{path} must hold exactly one camera, got {len(c2ws)}")
    for matrix in c2ws:
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{path} contains an invalid 4x4 c2w matrix")
    return c2ws


def load_capture_metadata(capture_dir: Path) -> dict[str, Any]:
    """Load capture metadata."""
    metadata = json.loads((capture_dir / "metadata.json").read_text())
    intrinsics = np.asarray(metadata["k_color"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("capture metadata k_color must be a finite 3x3 matrix")
    width = int(metadata["width"])
    height = int(metadata["height"])
    fps = float(metadata["replay_fps"])
    if width <= 0 or height <= 0 or fps <= 0.0:
        raise ValueError("capture metadata width/height/replay_fps must be positive")
    return {
        "intrinsics": intrinsics,
        "WH": [width, height],
        "fps": fps,
        "serial": str(metadata.get("serial", "")),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_color_frame_paths(capture_dir: Path, *, frame_count: int) -> list[Path]:
    """Map final-data frame i -> the input RGB image of the same source frame."""
    strict_frames = _read_jsonl(capture_dir / "frames.jsonl")
    if len(strict_frames) < frame_count:
        raise ValueError(
            f"frames.jsonl has {len(strict_frames)} frames, need {frame_count}"
        )
    input_rgb_by_source: dict[int, str] = {}
    for record in _read_jsonl(capture_dir / "input_frames.jsonl"):
        input_rgb_by_source.setdefault(
            int(record["source_frame_index"]), str(record["input_rgb_path"])
        )

    color_paths: list[Path] = []
    for frame_idx in range(frame_count):
        record = strict_frames[frame_idx]
        if int(record["seq"]) != frame_idx:
            raise ValueError(f"frames.jsonl seq mismatch at line {frame_idx}")
        source_frame_index = int(record["source_frame_index"])
        rgb_relpath = input_rgb_by_source.get(source_frame_index)
        if rgb_relpath is None:
            raise ValueError(
                f"no input_rgb frame for source_frame_index {source_frame_index}"
            )
        rgb_path = capture_dir / rgb_relpath
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        color_paths.append(rgb_path)
    return color_paths


def materialize_color_dir(case_dir: Path, color_paths: list[Path]) -> None:
    """Materialize the exported color-frame directory."""
    color_dir = case_dir / "color" / "0"
    color_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, source_path in enumerate(color_paths):
        target = color_dir / f"{frame_idx}.png"
        if target.exists():
            target.unlink()
        try:
            # Hardlink to avoid duplicating ~800 PNGs; falls back to a copy.
            target.hardlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target)


def seed_optimal_params(
    futurephystwin_root: Path,
    *,
    donor_case: str,
    case_name: str,
    force: bool,
) -> Path:
    """Seed optimal params."""
    donor_path = (
        futurephystwin_root / "experiments_optimization" / donor_case / "optimal_params.pkl"
    )
    if not donor_path.is_file():
        raise FileNotFoundError(f"donor optimal_params not found: {donor_path}")
    donor_params = _load_pickle(donor_path)
    missing = [key for key in REQUIRED_OPTIMAL_PARAM_KEYS if key not in donor_params]
    if missing:
        raise ValueError(f"{donor_path} is missing keys: {missing}")

    target_dir = futurephystwin_root / "experiments_optimization" / case_name
    target_path = target_dir / "optimal_params.pkl"
    if target_path.exists() and not force:
        raise FileExistsError(f"{target_path} already exists; pass --force to overwrite")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(donor_path, target_path)
    (target_dir / "optimal_params_provenance.json").write_text(
        json.dumps(
            {
                "copied_from": str(donor_path),
                "donor_case": donor_case,
                "note": (
                    "No zero-order optimization was run for this stitched case; "
                    "the first-order stage reuses the donor's zero-order params."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return target_path


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)

    manifest = json.loads(args.online_manifest_path.read_text())
    if str(manifest.get("status", "")) != "finished":
        raise ValueError(
            f"online manifest status is {manifest.get('status')!r}, expected 'finished'"
        )
    frame_count = int(manifest["num_frames_total"])

    final_data = load_and_validate_final_data(
        args.final_data_path, expected_frames=frame_count
    )
    c2ws = load_c2ws(args.table_calibrate_path)
    capture = load_capture_metadata(args.capture_dir)
    color_paths = resolve_color_frame_paths(args.capture_dir, frame_count=frame_count)

    case_dir = args.cases_root / args.case_name
    if case_dir.exists():
        if not args.force:
            raise FileExistsError(f"{case_dir} already exists; pass --force to overwrite")
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    shutil.copy2(args.final_data_path, case_dir / "final_data.pkl")
    with (case_dir / "calibrate.pkl").open("wb") as handle:
        pickle.dump(c2ws, handle)

    train_frame = int(frame_count * TRAIN_SPLIT_RATIO)
    if not 1 < train_frame < frame_count:
        raise ValueError(f"invalid train split boundary {train_frame} for {frame_count}")
    metadata = {
        "intrinsics": [capture["intrinsics"].tolist()],
        "serial_numbers": [capture["serial"]],
        "fps": capture["fps"],
        "WH": capture["WH"],
        "frame_num": frame_count,
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata) + "\n")
    split = {
        "frame_len": frame_count,
        "train": [0, train_frame],
        "test": [train_frame, frame_count],
    }
    (case_dir / "split.json").write_text(json.dumps(split) + "\n")
    materialize_color_dir(case_dir, color_paths)

    optimal_params_path = seed_optimal_params(
        args.futurephystwin_root,
        donor_case=args.donor_case,
        case_name=args.case_name,
        force=args.force,
    )

    provenance = {
        "capture_dir": str(args.capture_dir),
        "case_dir": str(case_dir),
        "donor_case": args.donor_case,
        "final_data_path": str(args.final_data_path),
        "frame_count": frame_count,
        "online_manifest_path": str(args.online_manifest_path),
        "optimal_params_path": str(optimal_params_path),
        "query_schema_hash": str(manifest.get("query_schema_hash", "")),
        "table_calibrate_path": str(args.table_calibrate_path),
        "track_process_status": str(final_data.get("track_process_status", "")),
        "train_frame": train_frame,
    }
    (case_dir / "case_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"exported case: {case_dir}")


if __name__ == "__main__":
    main()
