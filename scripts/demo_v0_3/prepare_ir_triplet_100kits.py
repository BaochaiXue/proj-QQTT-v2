#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest_v03_100kits.json"
KITS_JSONL_NAME = "kits.jsonl"


@dataclass(frozen=True)
class SourceCamera:
    camera_idx: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SourceReplay:
    source_format: str
    source_dir: Path
    cameras: list[SourceCamera]
    kit_indices: list[int]
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a normalized Demo v0.3 fixed IR-triplet replay folder. "
            "This is a data-prep step only; it does not run RealSense, FFS, SAM, EdgeTAM, or networking."
        )
    )
    parser.add_argument("--src-replay-dir", type=Path, required=True)
    parser.add_argument("--out-replay-dir", type=Path, required=True)
    parser.add_argument("--num-kits", type=int, default=100)
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-kit-fps", type=float, default=15.0)
    parser.add_argument("--allow-cycle-if-needed", action="store_true")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Compatibility flag. Demo v0.3 always writes manifest_v03_100kits.json and kits.jsonl.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must decode to a JSON object")
    return data


def _camera_idx_from_name(name: str) -> int:
    if not name.startswith("cam"):
        raise ValueError(f"camera directory must be named camN, got {name!r}")
    return int(name[3:])


def _sorted_int_stems(paths: list[Path]) -> list[int]:
    stems: list[int] = []
    for path in paths:
        try:
            stems.append(int(path.stem))
        except ValueError:
            continue
    return sorted(set(stems))


def _detect_v02_replay(src_dir: Path, *, camera_count: int) -> SourceReplay:
    metadata_path = src_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing v0.2 replay metadata: {metadata_path}")
    metadata = _load_json(metadata_path)
    raw_cameras = metadata.get("cameras")
    if not isinstance(raw_cameras, list):
        raise ValueError(f"{metadata_path} must contain a cameras list")

    cameras: list[SourceCamera] = []
    for item in raw_cameras:
        if not isinstance(item, dict):
            continue
        camera_idx = int(item.get("camera_idx", len(cameras)))
        cameras.append(SourceCamera(camera_idx=camera_idx, metadata=dict(item)))
    cameras = sorted(cameras, key=lambda item: item.camera_idx)[:camera_count]
    if len(cameras) != camera_count:
        raise ValueError(f"expected {camera_count} cameras in {metadata_path}, found {len(cameras)}")

    common_indices: set[int] | None = None
    for camera in cameras:
        cam_dir = src_dir / f"cam{camera.camera_idx}"
        left_dir = cam_dir / "left"
        right_dir = cam_dir / "right"
        if not left_dir.is_dir() or not right_dir.is_dir():
            raise ValueError(f"camera {camera.camera_idx} must have left/ and right/ directories under {cam_dir}")
        left_indices = set(_sorted_int_stems(list(left_dir.glob("*.png"))))
        right_indices = set(_sorted_int_stems(list(right_dir.glob("*.png"))))
        camera_indices = left_indices & right_indices
        common_indices = set(camera_indices) if common_indices is None else common_indices & camera_indices

    kit_indices = sorted(common_indices or set())
    declared_count = int(metadata.get("frame_count", len(kit_indices)))
    if declared_count > 0:
        kit_indices = [idx for idx in kit_indices if idx < declared_count]
    if not kit_indices:
        raise ValueError(f"no complete cam0/cam1/cam2 left/right IR kits found in {src_dir}")
    return SourceReplay(
        source_format="demo_v0_2_metadata_cam_left_right_png",
        source_dir=src_dir,
        cameras=cameras,
        kit_indices=kit_indices,
        metadata=metadata,
    )


def discover_source_replay(src_dir: Path, *, camera_count: int) -> SourceReplay:
    src_dir = src_dir.resolve()
    if not src_dir.is_dir():
        raise ValueError(f"source replay directory does not exist: {src_dir}")
    return _detect_v02_replay(src_dir, camera_count=camera_count)


def _validate_ir_image(path: Path, *, width: int, height: int) -> None:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{path} must be uint8, got {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"{path} must be single-channel IR, got shape={array.shape}")
    if tuple(array.shape) != (height, width):
        raise ValueError(f"{path} expected shape {(height, width)}, got {tuple(array.shape)}")


def _copy_ir_pair(
    *,
    source: SourceReplay,
    camera: SourceCamera,
    source_kit_idx: int,
    out_dir: Path,
    out_kit_idx: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    rel_left = Path(f"cam{camera.camera_idx}") / "left" / f"{out_kit_idx:06d}.png"
    rel_right = Path(f"cam{camera.camera_idx}") / "right" / f"{out_kit_idx:06d}.png"
    src_left = source.source_dir / f"cam{camera.camera_idx}" / "left" / f"{source_kit_idx:06d}.png"
    src_right = source.source_dir / f"cam{camera.camera_idx}" / "right" / f"{source_kit_idx:06d}.png"
    if not src_left.is_file() or not src_right.is_file():
        raise ValueError(f"missing source IR pair for cam{camera.camera_idx} kit {source_kit_idx:06d}")
    _validate_ir_image(src_left, width=width, height=height)
    _validate_ir_image(src_right, width=width, height=height)
    dst_left = out_dir / rel_left
    dst_right = out_dir / rel_right
    dst_left.parent.mkdir(parents=True, exist_ok=True)
    dst_right.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_left, dst_left)
    shutil.copy2(src_right, dst_right)
    return {
        "camera_idx": int(camera.camera_idx),
        "left_ir_path": rel_left.as_posix(),
        "right_ir_path": rel_right.as_posix(),
        "source_left_ir_path": src_left.relative_to(source.source_dir).as_posix(),
        "source_right_ir_path": src_right.relative_to(source.source_dir).as_posix(),
        "width": int(width),
        "height": int(height),
        "dtype": "uint8",
    }


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"output directory already exists and is not empty: {out_dir}; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def prepare_ir_triplet_100kits(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_kits) <= 0:
        raise ValueError("--num-kits must be positive")
    if int(args.camera_count) != 3:
        raise ValueError("Demo v0.3 currently requires --camera-count 3")
    if float(args.capture_kit_fps) <= 0:
        raise ValueError("--capture-kit-fps must be positive")

    source = discover_source_replay(Path(args.src_replay_dir), camera_count=int(args.camera_count))
    source_count = len(source.kit_indices)
    output_count = int(args.num_kits)
    if source_count < output_count and not bool(args.allow_cycle_if_needed):
        raise ValueError(
            f"source has only {source_count} complete kits but {output_count} requested; "
            "pass --allow-cycle-if-needed to repeat source kits"
        )

    out_dir = Path(args.out_replay_dir).resolve()
    _prepare_output_dir(out_dir, overwrite=bool(args.overwrite))

    kit_period_ms = 1000.0 / float(args.capture_kit_fps)
    selected_source_indices = [
        source.kit_indices[out_idx % source_count]
        for out_idx in range(output_count)
    ]
    cycled = output_count > source_count
    kits: list[dict[str, Any]] = []
    for out_idx, source_kit_idx in enumerate(selected_source_indices):
        kit_cameras = [
            _copy_ir_pair(
                source=source,
                camera=camera,
                source_kit_idx=source_kit_idx,
                out_dir=out_dir,
                out_kit_idx=out_idx,
                width=int(args.width),
                height=int(args.height),
            )
            for camera in source.cameras
        ]
        kits.append(
            {
                "kit_idx": int(out_idx),
                "source_kit_idx": int(source_kit_idx),
                "capture_time_s": float(out_idx / float(args.capture_kit_fps)),
                "capture_period_ms": float(kit_period_ms),
                "cameras": kit_cameras,
            }
        )

    metadata = dict(source.metadata)
    metadata.update(
        {
            "mode": "demo_v0_3_ir_triplet_100kits",
            "source_replay_dir": str(source.source_dir),
            "source_format": source.source_format,
            "frame_count": int(output_count),
            "kit_count": int(output_count),
            "camera_count": int(args.camera_count),
            "width": int(args.width),
            "height": int(args.height),
            "capture_kit_fps": float(args.capture_kit_fps),
            "kit_period_ms": float(kit_period_ms),
            "manifest": MANIFEST_NAME,
            "kits_jsonl": KITS_JSONL_NAME,
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    kits_jsonl_path = out_dir / KITS_JSONL_NAME
    with kits_jsonl_path.open("w", encoding="utf-8") as handle:
        for kit in kits:
            handle.write(json.dumps(kit, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "qqtt_demo_v0_3_ir_triplet_100kits",
        "schema_version": SCHEMA_VERSION,
        "out_replay_dir": str(out_dir),
        "source_replay_dir": str(source.source_dir),
        "source_format": source.source_format,
        "source_kit_count": int(source_count),
        "output_kit_count": int(output_count),
        "camera_count": int(args.camera_count),
        "width": int(args.width),
        "height": int(args.height),
        "capture_kit_fps": float(args.capture_kit_fps),
        "kit_period_ms": float(kit_period_ms),
        "unique_source_kit_count": int(len(set(selected_source_indices))),
        "cycled": bool(cycled),
        "manifest_path": str(out_dir / MANIFEST_NAME),
        "kits_jsonl_path": str(kits_jsonl_path),
        "source_kit_indices": [int(item) for item in selected_source_indices],
        "cameras": [camera.metadata for camera in source.cameras],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if bool(args.debug):
        print(f"[demo-v0.3-prepare] source_format={source.source_format}")
        print(f"[demo-v0.3-prepare] source_kit_count={source_count} output_kit_count={output_count}")
        print(f"[demo-v0.3-prepare] cycled={cycled} unique_source_kit_count={manifest['unique_source_kit_count']}")

    return manifest


def _print_summary(manifest: dict[str, Any]) -> None:
    keys = (
        "out_replay_dir",
        "source_replay_dir",
        "source_kit_count",
        "output_kit_count",
        "camera_count",
        "width",
        "height",
        "capture_kit_fps",
        "kit_period_ms",
        "unique_source_kit_count",
        "cycled",
        "manifest_path",
    )
    print(
        "[demo-v0.3-prepare-summary] "
        + " ".join(f"{key}={manifest[key]}" for key in keys),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_ir_triplet_100kits(args)
    except (OSError, ValueError) as exc:
        build_parser().exit(2, f"prepare_ir_triplet_100kits.py: error: {exc}\n")
    _print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

