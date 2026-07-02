from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_CASE_NAME = "shape_prior_frame0"
DEFAULT_ARTIFACT_DIR = Path("demo_v5_1/others/obj_shape_asap_outputs")
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACT_DIR / "shape_prior_lbs_from_chunks.pkl"
DEFAULT_REPORT_PATH = DEFAULT_ARTIFACT_DIR / "shape_prior_lbs_from_chunks.md"
DEFAULT_PREVIEW_VIDEO_PATH = DEFAULT_ARTIFACT_DIR / "shape_prior_lbs_preview.mp4"
DEFAULT_CONTACT_SHEET_PATH = Path(
    "demo_v5_1/others/obj_shape_asap_outputs/shape_prior_lbs_preview_sheet.png"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a Demo v5.1 shape-prior LBS diagnostic from online chunk "
            "tracking object points."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Demo v5.1 outputs root. Defaults to ./outputs.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=DEFAULT_CASE_NAME,
        help="Shape-prior warmup case under outputs/shape_prior_case/.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Pickle path for the derived LBS diagnostic.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown summary path for the derived LBS diagnostic.",
    )
    parser.add_argument(
        "--control-k",
        type=int,
        default=8,
        help="Nearest chunk object tracking points blended per query point.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth chunk frame when building the diagnostic.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open an Open3D animation after writing the diagnostic.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Open3D animation FPS when --view is used.",
    )
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write a headless MP4/contact-sheet LBS preview.",
    )
    parser.add_argument(
        "--preview-video-path",
        type=Path,
        default=DEFAULT_PREVIEW_VIDEO_PATH,
        help="MP4 path for --write-preview.",
    )
    parser.add_argument(
        "--contact-sheet-path",
        type=Path,
        default=DEFAULT_CONTACT_SHEET_PATH,
        help="PNG contact-sheet path for --write-preview.",
    )
    parser.add_argument(
        "--preview-frame-count",
        type=int,
        default=90,
        help="Maximum uniformly sampled frames rendered into the MP4 preview.",
    )
    return parser


def _require_file(path: Path) -> Path:
    """Return validated file."""
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


def _load_pickle(path: Path) -> Any:
    """Load pickle."""
    with _require_file(path).open("rb") as handle:
        return pickle.load(handle)


def _require_points(value: Any, *, name: str) -> np.ndarray:
    """Return validated points."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (points, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite points")
    return np.ascontiguousarray(points)


def _require_track_points(value: Any, *, name: str) -> np.ndarray:
    """Return validated track points."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"{name} must have shape (frames, points, 3)")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one frame and one point")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite points")
    return np.ascontiguousarray(points)


def _require_mask(value: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return a mask array or raise when its shape is invalid."""
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return np.ascontiguousarray(mask)


def _mesh_vertices_faces(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the mesh vertices faces."""
    mesh = o3d.io.read_triangle_mesh(str(_require_file(mesh_path)))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"shape-prior mesh is empty: {mesh_path}")
    mesh.compute_vertex_normals()
    vertices = _require_points(np.asarray(mesh.vertices), name="mesh vertices")
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"mesh triangles must have shape (triangles, 3): {mesh_path}")
    return vertices, np.ascontiguousarray(faces)


def _sorted_chunk_paths(chunks_dir: Path) -> list[Path]:
    """Return sorted chunk paths."""
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"required chunk directory not found: {chunks_dir}")
    chunk_paths = sorted(chunks_dir.glob("chunk_*.pkl"))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunk_*.pkl files found under {chunks_dir}")
    return chunk_paths


def load_shape_prior(outputs_root: Path, case_name: str) -> dict[str, Any]:
    """Load shape prior."""
    case_dir = Path(outputs_root) / "shape_prior_case" / str(case_name)
    final_data_path = _require_file(case_dir / "final_data.pkl")
    mesh_path = _require_file(case_dir / "shape" / "matching" / "final_mesh.glb")
    final_data = dict(_load_pickle(final_data_path))
    mesh_vertices, mesh_faces = _mesh_vertices_faces(mesh_path)
    surface_points = _require_points(
        final_data["surface_points"],
        name="shape-prior surface_points",
    )
    interior_points = _require_points(
        final_data["interior_points"],
        name="shape-prior interior_points",
    )
    return {
        "case_dir": case_dir,
        "final_data_path": final_data_path,
        "mesh_path": mesh_path,
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "surface_points": surface_points,
        "interior_points": interior_points,
    }


def load_chunk_tracking(outputs_root: Path, *, frame_stride: int = 1) -> dict[str, Any]:
    """Load chunk tracking."""
    if int(frame_stride) <= 0:
        raise ValueError("frame_stride must be positive")

    chunks_dir = Path(outputs_root) / "online_data" / "chunks"
    chunk_paths = _sorted_chunk_paths(chunks_dir)
    object_points: list[np.ndarray] = []
    object_colors: list[np.ndarray] = []
    object_visibilities: list[np.ndarray] = []
    object_motions_valid: list[np.ndarray] = []
    source_frame_indices: list[int] = []
    source_timestamps_s: list[float] = []
    expected_start = 0
    expected_point_count: int | None = None
    expected_schema_hash: str | None = None
    status_counts: dict[str, int] = {}

    for chunk_path in chunk_paths:
        chunk = dict(_load_pickle(chunk_path))
        start_frame = int(chunk["start_frame"])
        end_frame = int(chunk["end_frame"])
        if start_frame != expected_start:
            raise ValueError(
                f"{chunk_path} starts at {start_frame}, expected {expected_start}"
            )
        if end_frame <= start_frame:
            raise ValueError(f"{chunk_path} has invalid frame range")

        points = _require_track_points(
            chunk["object_points"],
            name=f"{chunk_path} object_points",
        )
        frame_count, point_count = points.shape[:2]
        if frame_count != end_frame - start_frame:
            raise ValueError(
                f"{chunk_path} frame range has {end_frame - start_frame} frames, "
                f"but object_points has {frame_count}"
            )
        if expected_point_count is None:
            expected_point_count = int(point_count)
        elif point_count != expected_point_count:
            raise ValueError(
                f"{chunk_path} object point count changed to {point_count}; "
                f"expected {expected_point_count}"
            )

        schema_hash = str(chunk.get("query_schema_hash", ""))
        if expected_schema_hash is None:
            expected_schema_hash = schema_hash
        elif schema_hash != expected_schema_hash:
            raise ValueError(
                f"{chunk_path} query_schema_hash changed to {schema_hash}; "
                f"expected {expected_schema_hash}"
            )

        colors = np.asarray(chunk["object_colors"], dtype=np.float64)
        if colors.shape != points.shape:
            raise ValueError(f"{chunk_path} object_colors must match object_points")
        colors = np.clip(colors, 0.0, 1.0)
        visibility = _require_mask(
            chunk["object_visibilities"],
            name=f"{chunk_path} object_visibilities",
            shape=points.shape[:2],
        )
        motion_valid = _require_mask(
            chunk["object_motions_valid"],
            name=f"{chunk_path} object_motions_valid",
            shape=points.shape[:2],
        )

        indices = [int(value) for value in chunk["source_frame_indices"]]
        if len(indices) != frame_count:
            raise ValueError(f"{chunk_path} source_frame_indices length mismatch")
        timestamps = [float(value) for value in chunk.get("source_timestamps_s", [])]
        if timestamps and len(timestamps) != frame_count:
            raise ValueError(f"{chunk_path} source_timestamps_s length mismatch")

        object_points.append(points)
        object_colors.append(np.ascontiguousarray(colors))
        object_visibilities.append(visibility)
        object_motions_valid.append(motion_valid)
        source_frame_indices.extend(indices)
        if timestamps:
            source_timestamps_s.extend(timestamps)

        status = str(chunk.get("track_process_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        expected_start = end_frame

    points = np.concatenate(object_points, axis=0)
    colors = np.concatenate(object_colors, axis=0)
    visibility = np.concatenate(object_visibilities, axis=0)
    motion_valid = np.concatenate(object_motions_valid, axis=0)

    frame_selector = np.arange(0, points.shape[0], int(frame_stride), dtype=np.int64)
    if frame_selector.size == 0:
        raise ValueError("frame_stride removed all frames")

    return {
        "chunk_paths": chunk_paths,
        "object_points": np.ascontiguousarray(points[frame_selector]),
        "object_colors": np.ascontiguousarray(colors[frame_selector]),
        "object_visibilities": np.ascontiguousarray(visibility[frame_selector]),
        "object_motions_valid": np.ascontiguousarray(motion_valid[frame_selector]),
        "source_frame_indices": [
            int(source_frame_indices[int(index)]) for index in frame_selector
        ],
        "source_timestamps_s": (
            [float(source_timestamps_s[int(index)]) for index in frame_selector]
            if source_timestamps_s
            else []
        ),
        "query_schema_hash": "" if expected_schema_hash is None else expected_schema_hash,
        "status_counts": status_counts,
        "frame_stride": int(frame_stride),
    }


def _knn_weights(
    query_points: np.ndarray,
    control_points: np.ndarray,
    *,
    control_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse-distance KNN weights from source to target points."""
    query_points = _require_points(query_points, name="query points")
    control_points = _require_points(control_points, name="control points")
    if int(control_k) <= 0:
        raise ValueError("control_k must be positive")
    k = min(int(control_k), int(control_points.shape[0]))
    tree = cKDTree(control_points)
    distances, indices = tree.query(query_points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    weights = np.zeros_like(distances, dtype=np.float64)
    exact = distances <= 1e-12
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        row_exact = exact[exact_rows]
        weights[exact_rows] = row_exact / row_exact.sum(axis=1, keepdims=True)
    if np.any(~exact_rows):
        nonzero_distances = distances[~exact_rows]
        inverse = 1.0 / np.maximum(nonzero_distances, 1e-12)
        weights[~exact_rows] = inverse / inverse.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(indices), np.ascontiguousarray(weights)


def _blend_displacement_trajectory(
    query_points: np.ndarray,
    control_trajectory: np.ndarray,
    *,
    control_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend displacement trajectory."""
    query_points = _require_points(query_points, name="query points")
    control_trajectory = _require_track_points(
        control_trajectory,
        name="control_trajectory",
    )
    control0 = control_trajectory[0]
    control_indices, control_weights = _knn_weights(
        query_points,
        control0,
        control_k=control_k,
    )
    out = np.empty(
        (control_trajectory.shape[0], query_points.shape[0], 3),
        dtype=np.float32,
    )
    out[0] = query_points.astype(np.float32)
    for frame_idx in range(1, control_trajectory.shape[0]):
        displacement = control_trajectory[frame_idx] - control0
        query_displacement = np.einsum(
            "pk,pkd->pd",
            control_weights,
            displacement[control_indices],
            optimize=True,
        )
        out[frame_idx] = (query_points + query_displacement).astype(np.float32)
    return out, control_indices, control_weights


def build_lbs_diagnostic(
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    case_name: str = DEFAULT_CASE_NAME,
    control_k: int = 8,
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Build lbs diagnostic."""
    outputs_root = Path(outputs_root)
    shape_prior = load_shape_prior(outputs_root, case_name)
    tracking = load_chunk_tracking(outputs_root, frame_stride=int(frame_stride))
    object_points = _require_track_points(
        tracking["object_points"],
        name="chunk object_points",
    )

    mesh_trajectory, mesh_control_indices, mesh_control_weights = (
        _blend_displacement_trajectory(
            shape_prior["mesh_vertices"],
            object_points,
            control_k=int(control_k),
        )
    )
    surface_trajectory, surface_control_indices, surface_control_weights = (
        _blend_displacement_trajectory(
            shape_prior["surface_points"],
            object_points,
            control_k=int(control_k),
        )
    )
    interior_trajectory, interior_control_indices, interior_control_weights = (
        _blend_displacement_trajectory(
            shape_prior["interior_points"],
            object_points,
            control_k=int(control_k),
        )
    )

    object_visibility_ratio = float(np.mean(tracking["object_visibilities"]))
    object_motion_valid_ratio = float(np.mean(tracking["object_motions_valid"]))
    summary = {
        "frame_count": int(object_points.shape[0]),
        "object_point_count": int(object_points.shape[1]),
        "mesh_vertex_count": int(shape_prior["mesh_vertices"].shape[0]),
        "mesh_triangle_count": int(shape_prior["mesh_faces"].shape[0]),
        "surface_point_count": int(shape_prior["surface_points"].shape[0]),
        "interior_point_count": int(shape_prior["interior_points"].shape[0]),
        "control_k": int(control_k),
        "frame_stride": int(frame_stride),
        "chunk_count": int(len(tracking["chunk_paths"])),
        "object_visibility_ratio": object_visibility_ratio,
        "object_motion_valid_ratio": object_motion_valid_ratio,
        "track_status_counts": dict(tracking["status_counts"]),
    }
    return {
        "summary": summary,
        "outputs_root": str(outputs_root),
        "case_name": str(case_name),
        "case_dir": str(shape_prior["case_dir"]),
        "shape_prior_final_data_path": str(shape_prior["final_data_path"]),
        "shape_prior_mesh_path": str(shape_prior["mesh_path"]),
        "chunk_paths": [str(path) for path in tracking["chunk_paths"]],
        "query_schema_hash": str(tracking["query_schema_hash"]),
        "source_frame_indices": tracking["source_frame_indices"],
        "source_timestamps_s": tracking["source_timestamps_s"],
        "mesh_faces": np.ascontiguousarray(shape_prior["mesh_faces"]),
        "mesh_vertices_frame0": shape_prior["mesh_vertices"].astype(np.float32),
        "mesh_vertex_trajectories": mesh_trajectory,
        "mesh_control_indices": mesh_control_indices.astype(np.int64),
        "mesh_control_weights": mesh_control_weights.astype(np.float32),
        "object_points": object_points.astype(np.float32),
        "object_colors": tracking["object_colors"].astype(np.float32),
        "object_visibilities": tracking["object_visibilities"],
        "object_motions_valid": tracking["object_motions_valid"],
        "surface_points_frame0": shape_prior["surface_points"].astype(np.float32),
        "surface_trajectories": surface_trajectory,
        "surface_control_indices": surface_control_indices.astype(np.int64),
        "surface_control_weights": surface_control_weights.astype(np.float32),
        "interior_points_frame0": shape_prior["interior_points"].astype(np.float32),
        "interior_trajectories": interior_trajectory,
        "interior_control_indices": interior_control_indices.astype(np.int64),
        "interior_control_weights": interior_control_weights.astype(np.float32),
    }


def write_lbs_diagnostic(result: dict[str, Any], output_path: Path) -> Path:
    """Write lbs diagnostic."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def _fmt_pct(value: float) -> str:
    """Format pct for display."""
    return f"{100.0 * float(value):.2f}%"


def build_report(result: dict[str, Any]) -> str:
    """Build report."""
    summary = dict(result["summary"])
    lines = [
        "# Demo v5.1 Shape-Prior LBS From Chunks",
        "",
        "## Inputs",
        "",
        f"- outputs root: `{result['outputs_root']}`",
        f"- case: `{result['case_dir']}`",
        f"- shape mesh: `{result['shape_prior_mesh_path']}`",
        f"- chunk count: {summary['chunk_count']:,}",
        f"- query schema hash: `{result['query_schema_hash']}`",
        "",
        "## LBS Diagnostic",
        "",
        f"- frames: {summary['frame_count']:,}",
        f"- chunk object tracking points: {summary['object_point_count']:,}",
        f"- mesh vertices: {summary['mesh_vertex_count']:,}",
        f"- mesh triangles: {summary['mesh_triangle_count']:,}",
        f"- surface points: {summary['surface_point_count']:,}",
        f"- interior points: {summary['interior_point_count']:,}",
        f"- control K: {summary['control_k']}",
        f"- frame stride: {summary['frame_stride']}",
        "",
        "## Tracking Quality",
        "",
        f"- object visibility ratio: {_fmt_pct(summary['object_visibility_ratio'])}",
        (
            "- object motion-valid ratio: "
            f"{_fmt_pct(summary['object_motion_valid_ratio'])}"
        ),
        f"- chunk track status counts: `{summary['track_status_counts']}`",
        "",
        "The published chunk object points are used as the LBS controls. The",
        "derived mesh/surface/interior trajectories are diagnostics only and do",
        "not overwrite `outputs/data/final_data.pkl`.",
    ]
    return "\n".join(lines) + "\n"


def write_report(result: dict[str, Any], report_path: Path) -> Path:
    """Write report."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(result), encoding="utf-8")
    return report_path


def _even_frame_indices(frame_count: int, requested_count: int) -> np.ndarray:
    """Return the even frame indices."""
    if int(frame_count) <= 0:
        raise ValueError("frame_count must be positive")
    if int(requested_count) <= 0:
        raise ValueError("requested_count must be positive")
    count = min(int(frame_count), int(requested_count))
    return np.unique(
        np.linspace(0, int(frame_count) - 1, count, dtype=np.int64)
    ).astype(np.int64)


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Return the mesh edges."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must have shape (triangles, 3)")
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _bbox_for_frames(result: dict[str, Any], frame_indices: np.ndarray):
    """Compute a padded bounding box for selected diagnostic frames."""
    arrays = [
        np.asarray(result["object_points"], dtype=np.float64)[frame_indices],
        np.asarray(result["mesh_vertex_trajectories"], dtype=np.float64)[
            frame_indices
        ],
        np.asarray(result["surface_trajectories"], dtype=np.float64)[frame_indices],
        np.asarray(result["interior_trajectories"], dtype=np.float64)[frame_indices],
    ]
    mins = []
    maxes = []
    for array in arrays:
        points = array.reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("preview arrays contain non-finite points")
        mins.append(np.percentile(points, 1.0, axis=0))
        maxes.append(np.percentile(points, 99.0, axis=0))
    bbox_min = np.min(np.stack(mins, axis=0), axis=0)
    bbox_max = np.max(np.stack(maxes, axis=0), axis=0)
    return bbox_min, bbox_max


def _set_equal_3d_limits(ax: Any, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
    """Set equal 3d limits."""
    center = 0.5 * (np.asarray(bbox_min) + np.asarray(bbox_max))
    extent = np.asarray(bbox_max) - np.asarray(bbox_min)
    radius = 0.58 * float(np.max(extent))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 0.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _render_lbs_frame(
    result: dict[str, Any],
    frame_idx: int,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    mesh_edges: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render lbs frame."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    mesh_vertices = np.asarray(
        result["mesh_vertex_trajectories"][int(frame_idx)],
        dtype=np.float64,
    )
    object_points = np.asarray(result["object_points"][int(frame_idx)])
    object_colors = np.asarray(result["object_colors"][int(frame_idx)])
    surface_points = np.asarray(result["surface_trajectories"][int(frame_idx)])
    interior_points = np.asarray(result["interior_trajectories"][int(frame_idx)])

    fig = plt.figure(figsize=(float(width) / 100.0, float(height) / 100.0), dpi=100)
    fig.patch.set_facecolor((0.035, 0.035, 0.04))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0.035, 0.035, 0.04))
    ax.view_init(elev=17.0, azim=-58.0)
    _set_equal_3d_limits(ax, bbox_min, bbox_max)
    ax.set_axis_off()

    edge_segments = mesh_vertices[mesh_edges]
    ax.add_collection3d(
        Line3DCollection(
            edge_segments,
            colors=(0.74, 0.74, 0.74, 0.34),
            linewidths=0.32,
        )
    )
    ax.scatter(
        object_points[:, 0],
        object_points[:, 1],
        object_points[:, 2],
        c=np.clip(object_colors, 0.0, 1.0),
        s=5.0,
        alpha=0.88,
        depthshade=False,
    )
    ax.scatter(
        surface_points[:, 0],
        surface_points[:, 1],
        surface_points[:, 2],
        c=[(0.0, 0.92, 1.0)],
        s=12.0,
        alpha=0.92,
        depthshade=False,
    )
    ax.scatter(
        interior_points[:, 0],
        interior_points[:, 1],
        interior_points[:, 2],
        c=[(0.13, 0.28, 1.0)],
        s=7.0,
        alpha=0.42,
        depthshade=False,
    )
    source_frames = result.get("source_frame_indices", [])
    source_frame = (
        source_frames[int(frame_idx)]
        if int(frame_idx) < len(source_frames)
        else int(frame_idx)
    )
    ax.text2D(
        0.03,
        0.95,
        f"LBS frame {int(frame_idx):04d} | source {int(source_frame)}",
        color=(0.92, 0.92, 0.92),
        transform=ax.transAxes,
        fontsize=11,
    )
    plt.tight_layout(pad=0.0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def write_lbs_preview(
    result: dict[str, Any],
    *,
    video_path: Path = DEFAULT_PREVIEW_VIDEO_PATH,
    contact_sheet_path: Path = DEFAULT_CONTACT_SHEET_PATH,
    max_video_frames: int = 90,
    sheet_frames: int = 12,
    fps: float = 5.0,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    """Write lbs preview."""
    import cv2
    from PIL import Image

    object_points = _require_track_points(
        result["object_points"],
        name="preview object_points",
    )
    frame_indices = _even_frame_indices(object_points.shape[0], max_video_frames)
    sheet_indices = _even_frame_indices(object_points.shape[0], sheet_frames)
    all_indices = np.unique(np.concatenate([frame_indices, sheet_indices]))
    bbox_min, bbox_max = _bbox_for_frames(result, all_indices)
    edges = _mesh_edges(np.asarray(result["mesh_faces"], dtype=np.int32))

    video_path = Path(video_path)
    contact_sheet_path = Path(contact_sheet_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")

    sheet_images: dict[int, Image.Image] = {}
    sheet_index_set = {int(value) for value in sheet_indices}
    try:
        for frame_idx in frame_indices:
            frame = _render_lbs_frame(
                result,
                int(frame_idx),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                mesh_edges=edges,
                width=int(width),
                height=int(height),
            )
            if int(frame_idx) in sheet_index_set:
                sheet_images[int(frame_idx)] = Image.fromarray(frame)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    for frame_idx in sheet_indices:
        if int(frame_idx) in sheet_images:
            continue
        frame = _render_lbs_frame(
            result,
            int(frame_idx),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            mesh_edges=edges,
            width=int(width),
            height=int(height),
        )
        sheet_images[int(frame_idx)] = Image.fromarray(frame)

    ordered_sheet = [sheet_images[int(index)] for index in sheet_indices]
    columns = min(4, len(ordered_sheet))
    rows = int(np.ceil(len(ordered_sheet) / float(columns)))
    thumb_w = max(1, int(width) // 2)
    thumb_h = max(1, int(height) // 2)
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (9, 9, 10))
    for idx, image in enumerate(ordered_sheet):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % columns) * thumb_w
        y = (idx // columns) * thumb_h
        sheet.paste(thumb, (x, y))
    sheet.save(contact_sheet_path)

    return {
        "video_path": str(video_path),
        "contact_sheet_path": str(contact_sheet_path),
        "video_frame_count": int(frame_indices.shape[0]),
        "contact_sheet_frame_count": int(sheet_indices.shape[0]),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }


def _point_cloud(
    points: np.ndarray,
    colors: np.ndarray | tuple[float, float, float],
) -> o3d.geometry.PointCloud:
    """Return the point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if isinstance(colors, tuple):
        cloud.paint_uniform_color(colors)
    else:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def open_lbs_animation(
    result: dict[str, Any],
    *,
    fps: float = 5.0,
    point_size: float = 4.0,
    pingpong: bool = True,
) -> None:
    """Open lbs animation."""
    import time

    mesh_trajectory = np.asarray(result["mesh_vertex_trajectories"], dtype=np.float64)
    mesh_faces = np.asarray(result["mesh_faces"], dtype=np.int32)
    object_points = np.asarray(result["object_points"], dtype=np.float64)
    object_colors = np.asarray(result["object_colors"], dtype=np.float64)
    surface_trajectory = np.asarray(result["surface_trajectories"], dtype=np.float64)
    interior_trajectory = np.asarray(result["interior_trajectories"], dtype=np.float64)
    frame_count = int(mesh_trajectory.shape[0])
    frames = list(range(frame_count))
    if pingpong and frame_count > 2:
        frames.extend(range(frame_count - 2, 0, -1))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_trajectory[0])
    mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
    mesh.compute_vertex_normals()
    mesh_wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_wire.paint_uniform_color((0.55, 0.55, 0.55))

    object_cloud = _point_cloud(object_points[0], object_colors[0])
    surface_cloud = _point_cloud(surface_trajectory[0], (0.0, 0.85, 0.95))
    interior_cloud = _point_cloud(interior_trajectory[0], (0.1, 0.28, 1.0))
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        window_name="Demo v5.1 shape-prior LBS from chunk tracking",
        width=1280,
        height=900,
    )
    for geometry in (mesh_wire, object_cloud, surface_cloud, interior_cloud, coord):
        viewer.add_geometry(geometry)
    options = viewer.get_render_option()
    options.point_size = float(point_size)
    options.background_color = np.asarray([0.04, 0.04, 0.04])
    viewer.poll_events()
    viewer.update_renderer()
    viewer.reset_view_point(True)

    sleep_s = 1.0 / max(float(fps), 1e-6)
    try:
        keep_running = True
        while keep_running:
            for frame_idx in frames:
                mesh.vertices = o3d.utility.Vector3dVector(
                    mesh_trajectory[frame_idx]
                )
                new_wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                mesh_wire.points = new_wire.points
                mesh_wire.lines = new_wire.lines
                mesh_wire.paint_uniform_color((0.55, 0.55, 0.55))
                viewer.update_geometry(mesh_wire)

                object_cloud.points = o3d.utility.Vector3dVector(
                    object_points[frame_idx]
                )
                object_cloud.colors = o3d.utility.Vector3dVector(
                    object_colors[frame_idx]
                )
                viewer.update_geometry(object_cloud)

                surface_cloud.points = o3d.utility.Vector3dVector(
                    surface_trajectory[frame_idx]
                )
                viewer.update_geometry(surface_cloud)

                interior_cloud.points = o3d.utility.Vector3dVector(
                    interior_trajectory[frame_idx]
                )
                viewer.update_geometry(interior_cloud)

                keep_running = viewer.poll_events()
                viewer.update_renderer()
                if not keep_running:
                    break
                time.sleep(sleep_s)
    finally:
        viewer.destroy_window()


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    result = build_lbs_diagnostic(
        outputs_root=args.outputs_root,
        case_name=args.case_name,
        control_k=int(args.control_k),
        frame_stride=int(args.frame_stride),
    )
    output_path = write_lbs_diagnostic(result, args.output_path)
    report_path = write_report(result, args.report_path)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote diagnostic: {output_path}")
    print(f"wrote report: {report_path}")
    if bool(args.write_preview):
        preview = write_lbs_preview(
            result,
            video_path=args.preview_video_path,
            contact_sheet_path=args.contact_sheet_path,
            max_video_frames=int(args.preview_frame_count),
            fps=float(args.fps),
        )
        print(json.dumps(preview, indent=2, sort_keys=True))
    if bool(args.view):
        open_lbs_animation(result, fps=float(args.fps))


if __name__ == "__main__":
    main()
