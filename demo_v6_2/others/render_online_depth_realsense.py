#!/usr/bin/env python3
"""Render Demo v6.1 online depth frames with RealSense Dynamic Jet coloring."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_ONLINE_DATA_DIR = Path("outputs_v6_1/online_data")
DEFAULT_OUTPUT_DIR = Path("demo_v6_1/others/obj_shape_asap_outputs")
DEFAULT_VIDEO_PATH = (
    DEFAULT_OUTPUT_DIR / "online_depth_realsense_dynamic_jet_5fps.mp4"
)
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "online_depth_realsense_dynamic_jet_5fps.json"
)
DEFAULT_GRID_PATH = (
    Path("outputs_v6_1/diagnostics")
    / "default_fake_live_first_window_raw_depth_full.png"
)
DEFAULT_GRID_FRAME_COUNT = 36
DEFAULT_GRID_COLUMNS = 6
DEFAULT_TILE_SIZE = (260, 175)

JET_RGB_ANCHORS = np.asarray(
    [
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
        [50, 0, 0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DepthFrameRecord:
    """One online depth frame and its source-recording label data."""

    online_frame_index: int
    depth_path: Path
    source_frame_index: int | None
    seq: int | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Render Demo v6.1 online depth frames as a RealSense Dynamic Jet "
            "MP4 and first-window diagnostic grid."
        )
    )
    parser.add_argument(
        "--online-data-dir",
        type=Path,
        default=DEFAULT_ONLINE_DATA_DIR,
        help="Demo v6.1 online_data directory containing depth/ and metadata.",
    )
    parser.add_argument(
        "--output-video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--first-window-grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Output PNG grid path for the first online depth window.",
    )
    parser.add_argument(
        "--grid-frame-count",
        type=int,
        default=DEFAULT_GRID_FRAME_COUNT,
        help="Number of leading frames to draw in the first-window grid.",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=DEFAULT_GRID_COLUMNS,
        help="Number of columns in the first-window grid.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to online_data/metadata.json fps.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _metadata_fps(online_data_dir: Path) -> float:
    """Return the online-data FPS."""
    metadata = _load_json(online_data_dir / "metadata.json")
    fps = float(metadata["fps"])
    if fps <= 0.0:
        raise ValueError(f"online metadata fps must be positive, got {fps}")
    return fps


def _optional_int(value: Any) -> int | None:
    """Return an int or None when the source value is absent."""
    if value is None:
        return None
    return int(value)


def load_frame_records(online_data_dir: Path) -> list[DepthFrameRecord]:
    """Load and validate the online frame mapping."""
    online_data_dir = Path(online_data_dir)
    enhance_metadata = _load_json(online_data_dir / "enhance_metadata.json")
    frame_mapping = enhance_metadata.get("frame_mapping")
    if not isinstance(frame_mapping, list) or not frame_mapping:
        raise ValueError("enhance_metadata.json frame_mapping must be non-empty")

    records: list[DepthFrameRecord] = []
    for expected_index, item in enumerate(frame_mapping):
        if not isinstance(item, dict):
            raise ValueError(f"frame_mapping[{expected_index}] must be an object")
        online_frame_index = int(item["online_frame_index"])
        if online_frame_index != expected_index:
            raise ValueError(
                "online_frame_index must be contiguous from zero; "
                f"got {online_frame_index} at row {expected_index}"
            )
        depth_path = online_data_dir / str(item["depth_path"])
        if not depth_path.is_file():
            raise FileNotFoundError(f"mapped depth frame not found: {depth_path}")
        records.append(
            DepthFrameRecord(
                online_frame_index=online_frame_index,
                depth_path=depth_path,
                source_frame_index=_optional_int(item.get("source_frame_index")),
                seq=_optional_int(item.get("seq")),
            )
        )
    return records


def load_depth_u16(path: Path) -> np.ndarray:
    """Load one uint16 millimeter depth frame."""
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"depth frame must be 2D, got {depth.shape}: {path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"depth frame must be uint16 millimeters: {path}")
    return np.ascontiguousarray(depth)


def _interpolate_rgb_colormap(
    normalized: np.ndarray,
    anchors_rgb: np.ndarray,
) -> np.ndarray:
    """Return RGB colors sampled from a piecewise-linear color map."""
    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    scaled = values * float(len(anchors_rgb) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors_rgb) - 1)
    fraction = (scaled - lower).reshape(-1, 1)
    colors = (
        anchors_rgb[lower.reshape(-1)] * (1.0 - fraction)
        + anchors_rgb[upper.reshape(-1)] * fraction
    )
    return np.clip(colors, 0, 255).astype(np.uint8).reshape(values.shape + (3,))


def colorize_realsense_dynamic_jet(depth_u16: np.ndarray) -> np.ndarray:
    """Colorize uint16 depth like RealSense Dynamic Jet."""
    depth = np.asarray(depth_u16, dtype=np.uint16)
    valid = depth > 0
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not bool(np.any(valid)):
        return rgb

    histogram = np.bincount(depth.reshape(-1), minlength=65536).astype(np.int64)
    cumulative = histogram.copy()
    cumulative[2:] = np.cumsum(histogram[2:]) + int(histogram[1])
    valid_pixel_count = int(cumulative[-1])
    if valid_pixel_count <= 0:
        return rgb

    normalized = cumulative[depth[valid]].astype(np.float32)
    normalized /= float(valid_pixel_count)
    rgb[valid] = _interpolate_rgb_colormap(normalized, JET_RGB_ANCHORS)
    return rgb


def _frame_label(record: DepthFrameRecord) -> str:
    """Return the per-frame display label."""
    parts = [f"online {record.online_frame_index:04d}"]
    if record.source_frame_index is not None:
        parts.append(f"source {record.source_frame_index}")
    if record.seq is not None:
        parts.append(f"seq {record.seq}")
    return " | ".join(parts)


def draw_label_bgr(
    frame_bgr: np.ndarray,
    label: str,
    *,
    font_scale: float,
    thickness: int,
    origin: tuple[int, int],
) -> None:
    """Draw a readable label onto one BGR frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame_bgr,
        label,
        origin,
        font,
        float(font_scale),
        (0, 0, 0),
        int(thickness) + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        label,
        origin,
        font,
        float(font_scale),
        (255, 255, 255),
        int(thickness),
        cv2.LINE_AA,
    )


def render_video(
    records: list[DepthFrameRecord],
    *,
    output_video_path: Path,
    fps: float,
) -> tuple[int, int]:
    """Render all mapped depth frames to an MP4."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    first_depth = load_depth_u16(records[0].depth_path)
    height, width = first_depth.shape
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if output_video_path.exists():
        output_video_path.unlink()

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_video_path}")

    try:
        for frame_number, record in enumerate(records):
            depth = first_depth if frame_number == 0 else load_depth_u16(
                record.depth_path
            )
            if depth.shape != (height, width):
                raise ValueError(
                    f"depth shape changed at {record.depth_path}: {depth.shape}"
                )
            frame_bgr = cv2.cvtColor(
                colorize_realsense_dynamic_jet(depth),
                cv2.COLOR_RGB2BGR,
            )
            draw_label_bgr(
                frame_bgr,
                _frame_label(record),
                font_scale=0.68,
                thickness=1,
                origin=(12, 28),
            )
            writer.write(frame_bgr)
            if (frame_number + 1) % 100 == 0 or frame_number + 1 == len(records):
                print(
                    f"rendered depth video {frame_number + 1}/{len(records)}",
                    flush=True,
                )
    finally:
        writer.release()
    return int(width), int(height)


def _make_grid_tile(record: DepthFrameRecord, depth: np.ndarray) -> np.ndarray:
    """Render one first-window grid tile."""
    tile_width, tile_height = DEFAULT_TILE_SIZE
    label_height = 28
    image_height = tile_height - label_height
    rgb = colorize_realsense_dynamic_jet(depth)
    resized = cv2.resize(
        rgb,
        (tile_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    tile[label_height:, :] = resized
    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    draw_label_bgr(
        tile_bgr,
        _frame_label(record),
        font_scale=0.36,
        thickness=1,
        origin=(4, 16),
    )
    return cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)


def render_first_window_grid(
    records: list[DepthFrameRecord],
    *,
    output_path: Path,
    frame_count: int,
    columns: int,
) -> None:
    """Render a first-window PNG grid using full-frame Dynamic Jet."""
    if frame_count <= 0:
        raise ValueError(f"grid frame count must be positive, got {frame_count}")
    if columns <= 0:
        raise ValueError(f"grid columns must be positive, got {columns}")

    selected_records = records[: min(int(frame_count), len(records))]
    rows = int(np.ceil(len(selected_records) / float(columns)))
    tile_width, tile_height = DEFAULT_TILE_SIZE
    canvas = np.zeros(
        (rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )
    for index, record in enumerate(selected_records):
        depth = load_depth_u16(record.depth_path)
        tile = _make_grid_tile(record, depth)
        row = index // columns
        column = index % columns
        y0 = row * tile_height
        x0 = column * tile_width
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write first-window grid: {output_path}")


def write_summary(
    records: list[DepthFrameRecord],
    *,
    output_path: Path,
    video_path: Path,
    grid_path: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Write the render summary."""
    summary = {
        "colorizer": "realsense_dynamic_jet",
        "depth_encoding": "uint16_millimeters_invalid_zero",
        "first_window_grid_path": str(grid_path),
        "fps": float(fps),
        "frame_count": int(len(records)),
        "height": int(height),
        "invalid_depth_rgb": [0, 0, 0],
        "output_video_path": str(video_path),
        "source_frame_first": records[0].source_frame_index,
        "source_frame_last": records[-1].source_frame_index,
        "style_reference": "librealsense rs2::colorizer Dynamic Jet",
        "width": int(width),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    online_data_dir = Path(args.online_data_dir)
    records = load_frame_records(online_data_dir)
    fps = float(args.fps) if args.fps is not None else _metadata_fps(
        online_data_dir
    )

    width, height = render_video(
        records,
        output_video_path=Path(args.output_video_path),
        fps=fps,
    )
    render_first_window_grid(
        records,
        output_path=Path(args.first_window_grid_path),
        frame_count=int(args.grid_frame_count),
        columns=int(args.grid_columns),
    )
    write_summary(
        records,
        output_path=Path(args.summary_path),
        video_path=Path(args.output_video_path),
        grid_path=Path(args.first_window_grid_path),
        fps=fps,
        width=width,
        height=height,
    )
    print(f"wrote video: {args.output_video_path}")
    print(f"wrote first-window grid: {args.first_window_grid_path}")
    print(f"wrote summary: {args.summary_path}")


if __name__ == "__main__":
    main()
