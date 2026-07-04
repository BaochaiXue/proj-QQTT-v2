#!/usr/bin/env python3
"""Render Demo v5.1 object/controller tracking chunks as an MP4.

The rendering style follows ``data_process_origin/data_process_sample.py``:
object points use fixed rainbow colors from the first-frame y coordinate, and
controller points are red spheres moving through the same 3D view.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CHUNKS_DIR = Path("outputs/online_data/chunks")
DEFAULT_OUTPUT_DIR = Path("demo_v5_1/others/obj_shape_asap_outputs")
DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "object_controller_tracking_all_chunks_5fps.mp4"
)
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "object_controller_tracking_all_chunks_5fps.json"
)
DEFAULT_FPS = 5.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 900
DEFAULT_CONTROLLER_RADIUS_M = 0.01


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Demo v5.1 object/controller tracking from all online "
            "final_data chunks."
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
        help="Open3D sphere radius used for each controller point.",
    )
    return parser


def _load_pickle(path: Path) -> Any:
    """Load pickle."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def _require_track_points(value: Any, *, name: str) -> np.ndarray:
    """Return validated track points."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (frames, points, 3), got {points.shape}"
        )
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(points)


def _require_mask(value: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return a mask array or raise when its shape is invalid."""
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return np.ascontiguousarray(mask)


def _sorted_chunk_paths(chunks_dir: Path) -> list[Path]:
    """Return sorted chunk paths."""
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"chunk directory not found: {chunks_dir}")
    chunk_paths = sorted(chunks_dir.glob("chunk_*.pkl"))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunk_*.pkl files found under {chunks_dir}")
    return chunk_paths


def _require_same_count(
    *,
    current_count: int,
    expected_count: int | None,
    chunk_path: Path,
    label: str,
) -> int:
    """Validate that all named arrays share the same frame count."""
    if expected_count is None:
        return int(current_count)
    if int(current_count) != int(expected_count):
        raise ValueError(
            f"{chunk_path} {label} count changed to {current_count}; "
            f"expected {expected_count}"
        )
    return int(expected_count)


def load_tracking_chunks(chunks_dir: Path) -> dict[str, Any]:
    """Load tracking chunks."""
    chunk_paths = _sorted_chunk_paths(Path(chunks_dir))
    object_points_parts: list[np.ndarray] = []
    object_visibilities_parts: list[np.ndarray] = []
    object_motions_valid_parts: list[np.ndarray] = []
    controller_points_parts: list[np.ndarray] = []
    source_frame_indices: list[int] = []
    source_timestamps_s: list[float] = []
    status_counts: dict[str, int] = {}
    expected_start = 0
    expected_schema_hash: str | None = None
    object_count: int | None = None
    controller_count: int | None = None

    for chunk_path in chunk_paths:
        chunk = dict(_load_pickle(chunk_path))
        start_frame = int(chunk["start_frame"])
        end_frame = int(chunk["end_frame"])
        if start_frame != expected_start:
            raise ValueError(
                f"{chunk_path} starts at {start_frame}, expected {expected_start}"
            )
        if end_frame <= start_frame:
            raise ValueError(
                f"{chunk_path} has invalid frame range [{start_frame}, {end_frame})"
            )

        object_points = _require_track_points(
            chunk["object_points"],
            name=f"{chunk_path} object_points",
        )
        controller_points = _require_track_points(
            chunk["controller_points"],
            name=f"{chunk_path} controller_points",
        )
        chunk_frame_count = int(object_points.shape[0])
        if chunk_frame_count != end_frame - start_frame:
            raise ValueError(f"{chunk_path} frame range does not match tensors")
        if int(controller_points.shape[0]) != chunk_frame_count:
            raise ValueError(f"{chunk_path} controller frame count mismatch")

        object_count = _require_same_count(
            current_count=int(object_points.shape[1]),
            expected_count=object_count,
            chunk_path=chunk_path,
            label="object",
        )
        controller_count = _require_same_count(
            current_count=int(controller_points.shape[1]),
            expected_count=controller_count,
            chunk_path=chunk_path,
            label="controller",
        )

        schema_hash = str(chunk.get("query_schema_hash", ""))
        if expected_schema_hash is None:
            expected_schema_hash = schema_hash
        elif schema_hash != expected_schema_hash:
            raise ValueError(
                f"{chunk_path} query_schema_hash changed to {schema_hash}; "
                f"expected {expected_schema_hash}"
            )

        object_visibilities = _require_mask(
            chunk["object_visibilities"],
            name=f"{chunk_path} object_visibilities",
            shape=object_points.shape[:2],
        )
        object_motions_valid = _require_mask(
            chunk["object_motions_valid"],
            name=f"{chunk_path} object_motions_valid",
            shape=object_points.shape[:2],
        )
        chunk_source_frames = [int(value) for value in chunk["source_frame_indices"]]
        if len(chunk_source_frames) != chunk_frame_count:
            raise ValueError(f"{chunk_path} source_frame_indices length mismatch")
        chunk_timestamps = [
            float(value) for value in chunk.get("source_timestamps_s", [])
        ]
        if chunk_timestamps and len(chunk_timestamps) != chunk_frame_count:
            raise ValueError(f"{chunk_path} source_timestamps_s length mismatch")

        object_points_parts.append(object_points)
        object_visibilities_parts.append(object_visibilities)
        object_motions_valid_parts.append(object_motions_valid)
        controller_points_parts.append(controller_points)
        source_frame_indices.extend(chunk_source_frames)
        source_timestamps_s.extend(chunk_timestamps)
        status = str(chunk.get("track_process_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        expected_start = end_frame

    return {
        "chunk_paths": chunk_paths,
        "object_points": np.concatenate(object_points_parts, axis=0),
        "object_visibilities": np.concatenate(object_visibilities_parts, axis=0),
        "object_motions_valid": np.concatenate(object_motions_valid_parts, axis=0),
        "controller_points": np.concatenate(controller_points_parts, axis=0),
        "source_frame_indices": source_frame_indices,
        "source_timestamps_s": source_timestamps_s,
        "query_schema_hash": ""
        if expected_schema_hash is None
        else expected_schema_hash,
        "track_status_counts": status_counts,
    }


def _rainbow_colors_from_first_frame_y(object_points: np.ndarray) -> np.ndarray:
    """Return the rainbow colors from first frame y."""
    first_frame_y = np.asarray(object_points[0, :, 1], dtype=np.float64)
    y_min = float(np.min(first_frame_y))
    y_max = float(np.max(first_frame_y))
    y_range = y_max - y_min
    if abs(y_range) <= 1e-12:
        normalized = np.zeros_like(first_frame_y, dtype=np.float64)
    else:
        normalized = (first_frame_y - y_min) / y_range
    return np.ascontiguousarray(plt.cm.rainbow(normalized)[:, :3])


def _sphere_mesh(
    center: np.ndarray,
    *,
    color: tuple[float, float, float],
    radius: float,
) -> o3d.geometry.TriangleMesh:
    """Return the sphere mesh."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate(np.asarray(center, dtype=np.float64))
    return mesh


def _point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Return an Open3D point cloud."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if color_array.shape != points.shape:
        raise ValueError(
            f"point cloud colors must have shape {points.shape}, got "
            f"{color_array.shape}"
        )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(color_array)
    return cloud


def _capture_rgb(
    visualizer: o3d.visualization.Visualizer,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Capture one RGB frame from an Open3D visualizer."""
    visualizer.poll_events()
    visualizer.update_renderer()
    frame = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
    frame_u8 = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    if frame_u8.shape[:2] != (int(height), int(width)):
        frame_u8 = cv2.resize(
            frame_u8,
            (int(width), int(height)),
            interpolation=cv2.INTER_AREA,
        )
    return frame_u8


def _draw_frame_label(
    frame_bgr: np.ndarray,
    *,
    frame_idx: int,
    frame_count: int,
    source_frame: int,
) -> None:
    """Draw the output frame label in the upper-left corner."""
    label = (
        f"frame {int(frame_idx)}/{int(frame_count) - 1} | source {int(source_frame)}"
    )
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


def _object_render_mask_for_frame(
    object_visibilities: np.ndarray,
    object_motions_valid: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    """Return object points allowed to appear in one rendered frame."""
    return object_visibilities[frame_idx] & object_motions_valid[frame_idx]


def render_tracking_video(
    tracking: dict[str, Any],
    *,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    controller_radius_m: float,
) -> None:
    """Render tracking video."""
    if float(fps) <= 0.0:
        raise ValueError("fps must be positive")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    if float(controller_radius_m) <= 0.0:
        raise ValueError("controller_radius_m must be positive")

    object_points = np.asarray(tracking["object_points"], dtype=np.float64)
    object_visibilities = np.asarray(tracking["object_visibilities"], dtype=bool)
    object_motions_valid = np.asarray(tracking["object_motions_valid"], dtype=bool)
    controller_points = np.asarray(tracking["controller_points"], dtype=np.float64)
    frame_count = int(object_points.shape[0])
    if int(controller_points.shape[0]) != frame_count:
        raise ValueError("object/controller frame count mismatch")
    if object_visibilities.shape != object_points.shape[:2]:
        raise ValueError("object_visibilities shape mismatch")
    if object_motions_valid.shape != object_points.shape[:2]:
        raise ValueError("object_motions_valid shape mismatch")
    source_frame_indices = [int(value) for value in tracking["source_frame_indices"]]
    if len(source_frame_indices) != frame_count:
        raise ValueError("source_frame_indices length mismatch")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rainbow_colors = _rainbow_colors_from_first_frame_y(object_points)
    visualizer = o3d.visualization.Visualizer()
    window_created = visualizer.create_window(
        window_name="Demo v5.1 object/controller tracking",
        width=int(width),
        height=int(height),
        visible=False,
    )
    if not window_created:
        raise RuntimeError("Open3D failed to create a headless visualization window")

    writer: cv2.VideoWriter | None = None
    try:
        first_mask = _object_render_mask_for_frame(
            object_visibilities,
            object_motions_valid,
            0,
        )
        object_cloud = _point_cloud(
            object_points[0, first_mask],
            rainbow_colors[first_mask],
        )
        visualizer.add_geometry(object_cloud)

        controller_meshes: list[o3d.geometry.TriangleMesh] = []
        previous_centers: list[np.ndarray] = []
        for point in controller_points[0]:
            mesh = _sphere_mesh(
                point,
                color=(1.0, 0.0, 0.0),
                radius=float(controller_radius_m),
            )
            controller_meshes.append(mesh)
            previous_centers.append(np.asarray(point, dtype=np.float64).copy())
            visualizer.add_geometry(mesh)

        options = visualizer.get_render_option()
        options.background_color = np.asarray([0.0, 0.0, 0.0])
        options.point_size = 3.0

        view_control = visualizer.get_view_control()
        view_control.set_front([1.0, 0.0, -2.0])
        view_control.set_up([0.0, 0.0, -1.0])
        view_control.set_zoom(1.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {output_path}")

        for frame_idx in range(frame_count):
            mask = _object_render_mask_for_frame(
                object_visibilities,
                object_motions_valid,
                frame_idx,
            )
            object_cloud.points = o3d.utility.Vector3dVector(
                object_points[frame_idx, mask]
            )
            object_cloud.colors = o3d.utility.Vector3dVector(rainbow_colors[mask])
            visualizer.update_geometry(object_cloud)

            for point_idx, point in enumerate(controller_points[frame_idx]):
                current = np.asarray(point, dtype=np.float64)
                controller_meshes[point_idx].translate(
                    current - previous_centers[point_idx]
                )
                visualizer.update_geometry(controller_meshes[point_idx])
                previous_centers[point_idx] = current.copy()

            base_rgb = _capture_rgb(
                visualizer,
                width=int(width),
                height=int(height),
            )
            frame_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
            _draw_frame_label(
                frame_bgr,
                frame_idx=frame_idx,
                frame_count=frame_count,
                source_frame=source_frame_indices[frame_idx],
            )
            writer.write(frame_bgr)
            if (frame_idx + 1) % 100 == 0 or frame_idx + 1 == frame_count:
                print(f"rendered {frame_idx + 1}/{frame_count}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        visualizer.destroy_window()


def build_summary(
    tracking: dict[str, Any],
    *,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    controller_radius_m: float,
) -> dict[str, Any]:
    """Build summary."""
    object_points = np.asarray(tracking["object_points"])
    object_visibilities = np.asarray(tracking["object_visibilities"], dtype=bool)
    object_motions_valid = np.asarray(tracking["object_motions_valid"], dtype=bool)
    controller_points = np.asarray(tracking["controller_points"])
    source_frames = [int(value) for value in tracking["source_frame_indices"]]
    rendered_object_mask = object_visibilities & object_motions_valid
    return {
        "chunk_count": int(len(tracking["chunk_paths"])),
        "controller_point_count": int(controller_points.shape[1]),
        "controller_sphere_radius_m": float(controller_radius_m),
        "fps": float(fps),
        "frame_count": int(object_points.shape[0]),
        "height": int(height),
        "object_motion_valid_ratio": float(np.mean(object_motions_valid)),
        "object_point_count": int(object_points.shape[1]),
        "object_rendered_valid_ratio": float(np.mean(rendered_object_mask)),
        "object_visibility_ratio": float(np.mean(object_visibilities)),
        "output_path": str(output_path),
        "query_schema_hash": str(tracking["query_schema_hash"]),
        "source_frame_first": source_frames[0] if source_frames else None,
        "source_frame_last": source_frames[-1] if source_frames else None,
        "style_reference": "data_process_origin/data_process_sample.py::visualize_track",
        "track_status_counts": dict(tracking["track_status_counts"]),
        "width": int(width),
    }


def write_summary(summary: dict[str, Any], path: Path) -> Path:
    """Write summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    tracking = load_tracking_chunks(args.chunks_dir)
    render_tracking_video(
        tracking,
        output_path=args.output_path,
        fps=float(args.fps),
        width=int(args.width),
        height=int(args.height),
        controller_radius_m=float(args.controller_radius_m),
    )
    summary = build_summary(
        tracking,
        output_path=args.output_path,
        fps=float(args.fps),
        width=int(args.width),
        height=int(args.height),
        controller_radius_m=float(args.controller_radius_m),
    )
    summary_path = write_summary(summary, args.summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote video: {args.output_path}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
