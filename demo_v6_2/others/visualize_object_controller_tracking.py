#!/usr/bin/env python3
"""Render Demo v6.2 object/controller tracking chunks as one MP4."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
from matplotlib import colormaps
import numpy as np
import open3d as o3d


DEFAULT_CHUNKS_DIR = Path("outputs_v6_1/online_data/chunks")
DEFAULT_OUTPUT_DIR = Path("demo_v6_2/others/obj_shape_asap_outputs")
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "object_controller_tracking_5fps.mp4"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "object_controller_tracking_5fps.json"
DEFAULT_FPS = 5.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 900
DEFAULT_CONTROLLER_RADIUS_M = 0.01


@dataclass(frozen=True)
class TrackingSequence:
    """Validated tracking tensors concatenated across contiguous chunks."""

    chunk_paths: tuple[Path, ...]
    object_points: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    controller_points: np.ndarray
    source_frame_indices: tuple[int, ...]
    query_schema_hash: str
    track_status_counts: dict[str, int]

    @property
    def frame_count(self) -> int:
        """Return the number of published frames."""
        return int(self.object_points.shape[0])

    @property
    def rendered_object_mask(self) -> np.ndarray:
        """Return object points accepted by both tracking validity gates."""
        return self.object_visibilities & self.object_motions_valid


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Render Demo v6.2 object/controller tracking from contiguous online chunks."
        )
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Directory containing chunk_*.pkl files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="MP4 path to write.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON summary path to write.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--controller-radius-m",
        type=float,
        default=DEFAULT_CONTROLLER_RADIUS_M,
        help="Radius of each red controller sphere.",
    )
    return parser


def _required(
    chunk: Mapping[str, Any],
    key: str,
    *,
    chunk_path: Path,
) -> Any:
    """Return one required chunk field."""
    if key not in chunk:
        raise KeyError(f"{chunk_path} is missing required field {key!r}")
    return chunk[key]


def _load_chunk(path: Path) -> Mapping[str, Any]:
    """Load one mapping-shaped chunk."""
    with path.open("rb") as handle:
        chunk = pickle.load(handle)
    if not isinstance(chunk, Mapping):
        raise TypeError(f"{path} must contain a mapping, got {type(chunk).__name__}")
    return chunk


def _required_integer(
    chunk: Mapping[str, Any],
    key: str,
    *,
    chunk_path: Path,
) -> int:
    """Return one required integer field without coercing other types."""
    value = _required(chunk, key, chunk_path=chunk_path)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{chunk_path} field {key!r} must be an integer")
    return int(value)


def _required_points(
    chunk: Mapping[str, Any],
    key: str,
    *,
    chunk_path: Path,
) -> np.ndarray:
    """Return one finite, non-empty ``(frames, points, 3)`` array."""
    points = np.asarray(
        _required(chunk, key, chunk_path=chunk_path),
        dtype=np.float64,
    )
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            f"{chunk_path} field {key!r} must have shape (frames, points, 3), "
            f"got {points.shape}"
        )
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{chunk_path} field {key!r} must be non-empty")
    if not np.isfinite(points).all():
        raise ValueError(f"{chunk_path} field {key!r} contains non-finite values")
    return np.ascontiguousarray(points)


def _required_mask(
    chunk: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, int],
    chunk_path: Path,
) -> np.ndarray:
    """Return one boolean mask with the required frame/point shape."""
    mask = np.asarray(_required(chunk, key, chunk_path=chunk_path))
    if mask.dtype != np.bool_:
        raise TypeError(f"{chunk_path} field {key!r} must be boolean")
    if mask.shape != shape:
        raise ValueError(
            f"{chunk_path} field {key!r} must have shape {shape}, got {mask.shape}"
        )
    return np.ascontiguousarray(mask)


def _required_source_indices(
    chunk: Mapping[str, Any],
    *,
    frame_count: int,
    chunk_path: Path,
) -> tuple[int, ...]:
    """Return one integer source-frame index per published frame."""
    values = np.asarray(_required(chunk, "source_frame_indices", chunk_path=chunk_path))
    if values.ndim != 1 or values.shape[0] != frame_count:
        raise ValueError(
            f"{chunk_path} source_frame_indices must have shape "
            f"({frame_count},), got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{chunk_path} source_frame_indices must be integers")
    return tuple(int(value) for value in values)


def _sorted_chunk_paths(chunks_dir: Path) -> tuple[Path, ...]:
    """Return all chunk files in publication order."""
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"chunk directory not found: {chunks_dir}")
    paths = tuple(sorted(chunks_dir.glob("chunk_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no chunk_*.pkl files found under {chunks_dir}")
    return paths


def load_tracking_sequence(chunks_dir: Path) -> TrackingSequence:
    """Load and concatenate the current online tracking chunk contract."""
    chunk_paths = _sorted_chunk_paths(Path(chunks_dir))
    object_points_parts: list[np.ndarray] = []
    object_visibility_parts: list[np.ndarray] = []
    object_motion_parts: list[np.ndarray] = []
    controller_points_parts: list[np.ndarray] = []
    source_frame_indices: list[int] = []
    status_counts: dict[str, int] = {}
    expected_start = 0
    expected_object_count = 0
    expected_controller_count = 0
    expected_schema_hash = ""

    for chunk_index, chunk_path in enumerate(chunk_paths):
        chunk = _load_chunk(chunk_path)
        start_frame = _required_integer(
            chunk,
            "start_frame",
            chunk_path=chunk_path,
        )
        end_frame = _required_integer(
            chunk,
            "end_frame",
            chunk_path=chunk_path,
        )
        if start_frame != expected_start:
            raise ValueError(
                f"{chunk_path} starts at frame {start_frame}; expected {expected_start}"
            )
        if end_frame <= start_frame:
            raise ValueError(
                f"{chunk_path} has invalid frame range [{start_frame}, {end_frame})"
            )

        object_points = _required_points(
            chunk,
            "object_points",
            chunk_path=chunk_path,
        )
        controller_points = _required_points(
            chunk,
            "controller_points",
            chunk_path=chunk_path,
        )
        frame_count = int(object_points.shape[0])
        if end_frame - start_frame != frame_count:
            raise ValueError(f"{chunk_path} frame range does not match object_points")
        if controller_points.shape[0] != frame_count:
            raise ValueError(
                f"{chunk_path} controller_points frame count does not match"
            )

        object_count = int(object_points.shape[1])
        controller_count = int(controller_points.shape[1])
        schema_hash = _required(chunk, "query_schema_hash", chunk_path=chunk_path)
        if not isinstance(schema_hash, str) or not schema_hash:
            raise TypeError(f"{chunk_path} query_schema_hash must be a string")
        if chunk_index == 0:
            expected_object_count = object_count
            expected_controller_count = controller_count
            expected_schema_hash = schema_hash
        else:
            if object_count != expected_object_count:
                raise ValueError(
                    f"{chunk_path} object point count changed from "
                    f"{expected_object_count} to {object_count}"
                )
            if controller_count != expected_controller_count:
                raise ValueError(
                    f"{chunk_path} controller point count changed from "
                    f"{expected_controller_count} to {controller_count}"
                )
            if schema_hash != expected_schema_hash:
                raise ValueError(f"{chunk_path} query_schema_hash changed")

        object_visibilities = _required_mask(
            chunk,
            "object_visibilities",
            shape=object_points.shape[:2],
            chunk_path=chunk_path,
        )
        object_motions_valid = _required_mask(
            chunk,
            "object_motions_valid",
            shape=object_points.shape[:2],
            chunk_path=chunk_path,
        )
        chunk_source_indices = _required_source_indices(
            chunk,
            frame_count=frame_count,
            chunk_path=chunk_path,
        )
        status = _required(chunk, "track_process_status", chunk_path=chunk_path)
        if not isinstance(status, str) or not status:
            raise TypeError(f"{chunk_path} track_process_status must be a string")

        object_points_parts.append(object_points)
        object_visibility_parts.append(object_visibilities)
        object_motion_parts.append(object_motions_valid)
        controller_points_parts.append(controller_points)
        source_frame_indices.extend(chunk_source_indices)
        status_counts[status] = status_counts.get(status, 0) + 1
        expected_start = end_frame

    return TrackingSequence(
        chunk_paths=chunk_paths,
        object_points=np.concatenate(object_points_parts, axis=0),
        object_visibilities=np.concatenate(object_visibility_parts, axis=0),
        object_motions_valid=np.concatenate(object_motion_parts, axis=0),
        controller_points=np.concatenate(controller_points_parts, axis=0),
        source_frame_indices=tuple(source_frame_indices),
        query_schema_hash=expected_schema_hash,
        track_status_counts=status_counts,
    )


def _rainbow_colors(object_points: np.ndarray) -> np.ndarray:
    """Assign stable rainbow colors from the first frame's y coordinate."""
    first_frame_y = object_points[0, :, 1]
    y_min = float(first_frame_y.min())
    y_range = float(first_frame_y.max()) - y_min
    if y_range == 0.0:
        normalized_y = np.zeros_like(first_frame_y)
    else:
        normalized_y = (first_frame_y - y_min) / y_range
    return np.ascontiguousarray(colormaps["rainbow"](normalized_y)[:, :3])


def _point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud with one color per point."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _controller_sphere(
    center: np.ndarray,
    *,
    radius_m: float,
) -> o3d.geometry.TriangleMesh:
    """Create one red controller marker."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius_m)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((1.0, 0.0, 0.0))
    mesh.translate(center)
    return mesh


def _capture_rgb(
    visualizer: o3d.visualization.Visualizer,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Capture one RGB frame at the requested output size."""
    visualizer.poll_events()
    visualizer.update_renderer()
    frame = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
    frame_u8 = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    if frame_u8.shape[:2] != (height, width):
        frame_u8 = cv2.resize(
            frame_u8,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    return frame_u8


def _draw_frame_label(
    frame_bgr: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    source_frame_index: int,
) -> None:
    """Draw the published and source frame indices."""
    label = f"frame {frame_index}/{frame_count - 1} | source {source_frame_index}"
    origin = (24, 38)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame_bgr,
        label,
        origin,
        font,
        0.82,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        label,
        origin,
        font,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_tracking_video(
    tracking: TrackingSequence,
    *,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    controller_radius_m: float,
) -> None:
    """Render one video from a validated tracking sequence."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if controller_radius_m <= 0.0:
        raise ValueError("controller_radius_m must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    object_colors = _rainbow_colors(tracking.object_points)
    rendered_mask = tracking.rendered_object_mask

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name="Demo v6.2 object/controller tracking",
        width=width,
        height=height,
        visible=False,
    ):
        raise RuntimeError("Open3D failed to create a headless window")

    writer: cv2.VideoWriter | None = None
    try:
        first_mask = rendered_mask[0]
        object_cloud = _point_cloud(
            tracking.object_points[0, first_mask],
            object_colors[first_mask],
        )
        visualizer.add_geometry(object_cloud)

        controller_meshes: list[o3d.geometry.TriangleMesh] = []
        controller_centers: list[np.ndarray] = []
        for point in tracking.controller_points[0]:
            center = np.asarray(point, dtype=np.float64)
            mesh = _controller_sphere(center, radius_m=controller_radius_m)
            visualizer.add_geometry(mesh)
            controller_meshes.append(mesh)
            controller_centers.append(center.copy())

        render_options = visualizer.get_render_option()
        render_options.background_color = np.zeros(3)
        render_options.point_size = 3.0
        view_control = visualizer.get_view_control()
        view_control.set_front((1.0, 0.0, -2.0))
        view_control.set_up((0.0, 0.0, -1.0))
        view_control.set_zoom(1.0)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {output_path}")

        for frame_index in range(tracking.frame_count):
            frame_mask = rendered_mask[frame_index]
            object_cloud.points = o3d.utility.Vector3dVector(
                tracking.object_points[frame_index, frame_mask]
            )
            object_cloud.colors = o3d.utility.Vector3dVector(object_colors[frame_mask])
            visualizer.update_geometry(object_cloud)

            for point_index, point in enumerate(
                tracking.controller_points[frame_index]
            ):
                center = np.asarray(point, dtype=np.float64)
                controller_meshes[point_index].translate(
                    center - controller_centers[point_index]
                )
                visualizer.update_geometry(controller_meshes[point_index])
                controller_centers[point_index] = center.copy()

            frame_rgb = _capture_rgb(
                visualizer,
                width=width,
                height=height,
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _draw_frame_label(
                frame_bgr,
                frame_index=frame_index,
                frame_count=tracking.frame_count,
                source_frame_index=tracking.source_frame_indices[frame_index],
            )
            writer.write(frame_bgr)
            rendered_count = frame_index + 1
            if rendered_count % 100 == 0 or rendered_count == tracking.frame_count:
                print(
                    f"rendered {rendered_count}/{tracking.frame_count}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.release()
        visualizer.destroy_window()


def build_summary(
    tracking: TrackingSequence,
    *,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    controller_radius_m: float,
) -> dict[str, Any]:
    """Build the machine-readable render summary."""
    rendered_mask = tracking.rendered_object_mask
    return {
        "chunk_count": len(tracking.chunk_paths),
        "controller_point_count": int(tracking.controller_points.shape[1]),
        "controller_sphere_radius_m": controller_radius_m,
        "fps": fps,
        "frame_count": tracking.frame_count,
        "height": height,
        "object_motion_valid_ratio": float(tracking.object_motions_valid.mean()),
        "object_point_count": int(tracking.object_points.shape[1]),
        "object_rendered_valid_ratio": float(rendered_mask.mean()),
        "object_visibility_ratio": float(tracking.object_visibilities.mean()),
        "output_path": str(output_path),
        "query_schema_hash": tracking.query_schema_hash,
        "render_policy": "visibility_and_motion_valid",
        "source_frame_first": tracking.source_frame_indices[0],
        "source_frame_last": tracking.source_frame_indices[-1],
        "style_reference": (
            "data_process_origin/data_process_sample.py::visualize_track"
        ),
        "track_status_counts": tracking.track_status_counts,
        "video_size_bytes": output_path.stat().st_size,
        "width": width,
    }


def write_summary(summary: Mapping[str, Any], path: Path) -> None:
    """Write the render summary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Render the configured tracking video and summary."""
    args = build_parser().parse_args(argv)
    tracking = load_tracking_sequence(args.chunks_dir)
    render_tracking_video(
        tracking,
        output_path=args.output_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        controller_radius_m=args.controller_radius_m,
    )
    summary = build_summary(
        tracking,
        output_path=args.output_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        controller_radius_m=args.controller_radius_m,
    )
    write_summary(summary, args.summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote video: {args.output_path}")
    print(f"wrote summary: {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
