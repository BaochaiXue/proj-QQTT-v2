from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


DEFAULT_OUTPUTS_ROOT = Path("outputs_v6_1")
DEFAULT_CASE_NAME = "shape_prior_frame0"
DEFAULT_ARTIFACT_DIR = Path("demo_v6_1/others/obj_shape_asap_outputs")
DEFAULT_PREVIEW_VIDEO_PATH = DEFAULT_ARTIFACT_DIR / "shape_prior_lbs_preview.mp4"
DEFAULT_CONTACT_SHEET_PATH = Path(
    "demo_v6_1/others/obj_shape_asap_outputs/shape_prior_lbs_preview_sheet.png"
)

RAW_PCD_RGB = np.asarray([150, 154, 162], dtype=np.uint8)
SURFACE_RGB = np.asarray([0, 224, 255], dtype=np.uint8)
INTERIOR_RGB = np.asarray([255, 104, 168], dtype=np.uint8)
BACKGROUND_RGB = np.asarray([8, 9, 12], dtype=np.uint8)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Render a static Demo v6.1 shape-prior orbit preview from the "
            "raw PCD, surface points, and interior points."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Demo v6.1 outputs root. Defaults to ./outputs_v6_1.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=DEFAULT_CASE_NAME,
        help="Shape-prior warmup case under outputs_v6_1/shape_prior_case/.",
    )
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write the headless MP4/contact-sheet shape-prior orbit preview.",
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
        help="Number of frames in the one-circle orbit MP4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Preview MP4 FPS.",
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
    """Return validated 3D points."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (points, 3)")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite points")
    return np.ascontiguousarray(points)


def _mesh_vertices_faces(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh vertices and faces for summary and framing."""
    mesh = o3d.io.read_triangle_mesh(str(_require_file(mesh_path)))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"shape-prior mesh is empty: {mesh_path}")
    vertices = _require_points(np.asarray(mesh.vertices), name="mesh vertices")
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"mesh triangles must have shape (triangles, 3): {mesh_path}")
    return vertices, np.ascontiguousarray(faces)


def _load_processed_object_mask(mask_path: Path) -> np.ndarray:
    """Return the processed object mask used by the warmup shape-prior case."""
    processed_masks = _load_pickle(_require_file(mask_path))
    object_mask = np.asarray(processed_masks[0][0]["object"], dtype=bool)
    if object_mask.ndim != 2:
        raise ValueError(f"processed object mask must be 2D: {mask_path}")
    if not np.any(object_mask):
        raise ValueError(f"processed object mask is empty: {mask_path}")
    return np.ascontiguousarray(object_mask)


def _masked_raw_object_pcd(pcd_path: Path, mask_path: Path) -> np.ndarray:
    """Return raw object PCD points from the warmup RGB-D point grid."""
    pcd_data = np.load(_require_file(pcd_path))
    points = np.asarray(pcd_data["points"], dtype=np.float64)
    valid_mask = np.asarray(pcd_data["masks"], dtype=bool)
    if points.ndim != 4 or points.shape[0] != 1 or points.shape[-1] != 3:
        raise ValueError(f"expected one-camera point grid at {pcd_path}")
    if valid_mask.shape != points.shape[:3]:
        raise ValueError(f"pcd masks must match point grid shape at {pcd_path}")

    object_mask = _load_processed_object_mask(mask_path)
    points = points[0]
    valid_mask = valid_mask[0]
    if object_mask.shape != points.shape[:2]:
        raise ValueError(
            f"processed object mask shape {object_mask.shape} does not match "
            f"pcd image shape {points.shape[:2]}"
        )
    raw_object_points = points[object_mask & valid_mask]
    return _require_points(raw_object_points, name="raw masked object PCD")


def load_shape_prior(outputs_root: Path, case_name: str) -> dict[str, Any]:
    """Load the static shape-prior assets used by the orbit preview."""
    case_dir = Path(outputs_root) / "shape_prior_case" / str(case_name)
    final_data_path = _require_file(case_dir / "final_data.pkl")
    raw_pcd_path = _require_file(case_dir / "pcd" / "0.npz")
    processed_masks_path = _require_file(case_dir / "mask" / "processed_masks.pkl")
    mesh_path = _require_file(case_dir / "shape" / "matching" / "final_mesh.glb")

    final_data = dict(_load_pickle(final_data_path))
    mesh_vertices, mesh_faces = _mesh_vertices_faces(mesh_path)
    raw_pcd_points = _masked_raw_object_pcd(raw_pcd_path, processed_masks_path)
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
        "raw_pcd_path": raw_pcd_path,
        "processed_masks_path": processed_masks_path,
        "mesh_path": mesh_path,
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "raw_pcd_points": raw_pcd_points,
        "surface_points": surface_points,
        "interior_points": interior_points,
    }


def build_shape_prior_orbit_diagnostic(
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    case_name: str = DEFAULT_CASE_NAME,
) -> dict[str, Any]:
    """Build the static shape-prior diagnostic rendered by the preview."""
    outputs_root = Path(outputs_root)
    shape_prior = load_shape_prior(outputs_root, case_name)
    summary = {
        "raw_pcd_point_count": int(shape_prior["raw_pcd_points"].shape[0]),
        "mesh_vertex_count": int(shape_prior["mesh_vertices"].shape[0]),
        "mesh_triangle_count": int(shape_prior["mesh_faces"].shape[0]),
        "surface_point_count": int(shape_prior["surface_points"].shape[0]),
        "interior_point_count": int(shape_prior["interior_points"].shape[0]),
    }
    return {
        "summary": summary,
        "outputs_root": str(outputs_root),
        "case_name": str(case_name),
        "case_dir": str(shape_prior["case_dir"]),
        "shape_prior_final_data_path": str(shape_prior["final_data_path"]),
        "shape_prior_raw_pcd_path": str(shape_prior["raw_pcd_path"]),
        "shape_prior_processed_masks_path": str(
            shape_prior["processed_masks_path"]
        ),
        "shape_prior_mesh_path": str(shape_prior["mesh_path"]),
        "mesh_vertices": shape_prior["mesh_vertices"].astype(np.float32),
        "mesh_faces": np.ascontiguousarray(shape_prior["mesh_faces"]),
        "raw_pcd_points": shape_prior["raw_pcd_points"].astype(np.float32),
        "surface_points": shape_prior["surface_points"].astype(np.float32),
        "interior_points": shape_prior["interior_points"].astype(np.float32),
    }


def _positive_int(value: int, *, name: str) -> int:
    """Return a positive integer."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _even_frame_indices(frame_count: int, requested_count: int) -> np.ndarray:
    """Return evenly spaced frame indices."""
    frame_count = _positive_int(frame_count, name="frame_count")
    requested_count = _positive_int(requested_count, name="requested_count")
    count = min(frame_count, requested_count)
    return np.unique(np.linspace(0, frame_count - 1, count, dtype=np.int64))


def _bbox_for_shape_prior(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Compute a bounding box for the shape-prior point sets."""
    point_sets = [
        np.asarray(result["raw_pcd_points"], dtype=np.float64),
        np.asarray(result["surface_points"], dtype=np.float64),
        np.asarray(result["interior_points"], dtype=np.float64),
        np.asarray(result["mesh_vertices"], dtype=np.float64),
    ]
    points = np.concatenate(point_sets, axis=0)
    if not np.isfinite(points).all():
        raise ValueError("shape-prior preview points contain non-finite values")
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    if float(np.max(extent)) <= 0.0:
        raise ValueError("shape-prior preview points have zero spatial extent")
    return bbox_min, bbox_max


def _camera_basis(azimuth_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, and forward axes for an orbit camera."""
    azimuth = np.deg2rad(float(azimuth_degrees))
    elevation = np.deg2rad(22.0)
    eye_direction = np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )
    forward = -eye_direction / np.linalg.norm(eye_direction)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise ValueError("orbit camera basis is degenerate")
    right /= right_norm
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def _project_points(
    points: np.ndarray,
    *,
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    half_width_m: float,
    half_height_m: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points into the orbit preview frame."""
    relative = np.asarray(points, dtype=np.float64) - center[None, :]
    projected_x = relative @ right
    projected_y = relative @ up
    depth = relative @ forward
    pixel_x = np.rint(
        (0.5 + 0.5 * projected_x / float(half_width_m)) * (int(width) - 1)
    ).astype(np.int32)
    pixel_y = np.rint(
        (0.5 - 0.5 * projected_y / float(half_height_m)) * (int(height) - 1)
    ).astype(np.int32)
    visible = (
        (pixel_x >= 0)
        & (pixel_x < int(width))
        & (pixel_y >= 0)
        & (pixel_y < int(height))
    )
    return pixel_x[visible], pixel_y[visible], depth[visible]


def _depth_shade(depth: np.ndarray) -> np.ndarray:
    """Return near-brighter depth shading weights."""
    if depth.size == 0:
        return depth.astype(np.float64)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.ones_like(depth, dtype=np.float64)
    far_to_near = 1.0 - (depth - depth_min) / (depth_max - depth_min)
    return 0.62 + 0.38 * far_to_near


def _draw_pixel_points(
    canvas: np.ndarray,
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    color: np.ndarray,
    alpha: float,
) -> None:
    """Draw a dense one-pixel point layer."""
    if pixel_x.size == 0:
        return
    order = np.argsort(depth)[::-1]
    x = pixel_x[order]
    y = pixel_y[order]
    shade = _depth_shade(depth[order])[:, None]
    shaded_color = np.clip(color[None, :] * shade, 0.0, 255.0)
    existing = canvas[y, x].astype(np.float64)
    blended = existing * (1.0 - float(alpha)) + shaded_color * float(alpha)
    canvas[y, x] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _draw_circle_points(
    canvas: np.ndarray,
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    color: np.ndarray,
    radius: int,
    alpha: float,
) -> None:
    """Draw a sparse point layer as anti-aliased circles."""
    if pixel_x.size == 0:
        return
    import cv2

    overlay = canvas.copy()
    order = np.argsort(depth)[::-1]
    shade = _depth_shade(depth[order])
    for draw_index, point_index in enumerate(order):
        shaded = np.clip(color.astype(np.float64) * shade[draw_index], 0.0, 255.0)
        cv2.circle(
            overlay,
            (int(pixel_x[point_index]), int(pixel_y[point_index])),
            int(radius),
            tuple(int(value) for value in shaded),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, float(alpha), canvas, 1.0 - float(alpha), 0.0, canvas)


def _draw_legend(
    canvas: np.ndarray,
    *,
    frame_idx: int,
    frame_count: int,
) -> None:
    """Draw the preview legend and orbit frame counter."""
    import cv2

    overlay = canvas.copy()
    cv2.rectangle(overlay, (18, 18), (268, 112), (14, 15, 18), thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)

    entries = [
        ("raw pcd", RAW_PCD_RGB),
        ("surface points", SURFACE_RGB),
        ("interior points", INTERIOR_RGB),
    ]
    for row, (label, color) in enumerate(entries):
        y = 42 + row * 26
        cv2.circle(
            canvas,
            (36, y - 5),
            6,
            tuple(int(value) for value in color),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (54, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 237, 240),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"shape prior orbit {int(frame_idx) + 1:03d}/{int(frame_count):03d}",
        (18, canvas.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (228, 231, 235),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _render_shape_prior_orbit_frame(
    result: dict[str, Any],
    *,
    frame_idx: int,
    frame_count: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one frame of the static shape-prior orbit preview."""
    width = _positive_int(width, name="width")
    height = _positive_int(height, name="height")
    frame_count = _positive_int(frame_count, name="frame_count")

    center = 0.5 * (np.asarray(bbox_min) + np.asarray(bbox_max))
    extent = np.asarray(bbox_max) - np.asarray(bbox_min)
    radius = 0.62 * float(np.max(extent))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("shape-prior orbit radius must be positive")

    azimuth = 360.0 * float(frame_idx) / float(frame_count)
    right, up, forward = _camera_basis(azimuth)
    half_height_m = 1.18 * radius
    half_width_m = half_height_m * float(width) / float(height)

    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = BACKGROUND_RGB

    for key, color, alpha, draw_radius in [
        ("raw_pcd_points", RAW_PCD_RGB, 0.58, 1),
        ("interior_points", INTERIOR_RGB, 0.82, 3),
        ("surface_points", SURFACE_RGB, 0.96, 4),
    ]:
        pixel_x, pixel_y, depth = _project_points(
            np.asarray(result[key], dtype=np.float64),
            center=center,
            right=right,
            up=up,
            forward=forward,
            half_width_m=half_width_m,
            half_height_m=half_height_m,
            width=width,
            height=height,
        )
        if draw_radius == 1:
            _draw_pixel_points(
                canvas,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                depth=depth,
                color=color,
                alpha=alpha,
            )
        else:
            _draw_circle_points(
                canvas,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                depth=depth,
                color=color,
                radius=draw_radius,
                alpha=alpha,
            )

    _draw_legend(canvas, frame_idx=frame_idx, frame_count=frame_count)
    return canvas


def write_shape_prior_orbit_preview(
    result: dict[str, Any],
    *,
    video_path: Path = DEFAULT_PREVIEW_VIDEO_PATH,
    contact_sheet_path: Path = DEFAULT_CONTACT_SHEET_PATH,
    frame_count: int = 90,
    sheet_frames: int = 12,
    fps: float = 5.0,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    """Write the static shape-prior one-circle orbit preview."""
    import cv2
    from PIL import Image

    frame_count = _positive_int(frame_count, name="frame_count")
    sheet_frames = _positive_int(sheet_frames, name="sheet_frames")
    if float(fps) <= 0.0:
        raise ValueError("fps must be positive")

    bbox_min, bbox_max = _bbox_for_shape_prior(result)
    sheet_indices = _even_frame_indices(frame_count, sheet_frames)

    video_path = Path(video_path)
    contact_sheet_path = Path(contact_sheet_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")

    sheet_images: dict[int, Image.Image] = {}
    sheet_index_set = {int(value) for value in sheet_indices}
    try:
        for frame_idx in range(frame_count):
            frame = _render_shape_prior_orbit_frame(
                result,
                frame_idx=int(frame_idx),
                frame_count=frame_count,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                width=int(width),
                height=int(height),
            )
            if int(frame_idx) in sheet_index_set:
                sheet_images[int(frame_idx)] = Image.fromarray(frame)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ordered_sheet = [sheet_images[int(index)] for index in sheet_indices]
    columns = min(4, len(ordered_sheet))
    rows = int(np.ceil(len(ordered_sheet) / float(columns)))
    thumb_w = max(1, int(width) // 2)
    thumb_h = max(1, int(height) // 2)
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (8, 9, 12))
    for idx, image in enumerate(ordered_sheet):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % columns) * thumb_w
        y = (idx // columns) * thumb_h
        sheet.paste(thumb, (x, y))
    sheet.save(contact_sheet_path)

    return {
        "video_path": str(video_path),
        "contact_sheet_path": str(contact_sheet_path),
        "video_frame_count": int(frame_count),
        "contact_sheet_frame_count": int(sheet_indices.shape[0]),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "raw_pcd_color_rgb": RAW_PCD_RGB.astype(int).tolist(),
        "surface_color_rgb": SURFACE_RGB.astype(int).tolist(),
        "interior_color_rgb": INTERIOR_RGB.astype(int).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    result = build_shape_prior_orbit_diagnostic(
        outputs_root=args.outputs_root,
        case_name=args.case_name,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if bool(args.write_preview):
        preview = write_shape_prior_orbit_preview(
            result,
            video_path=args.preview_video_path,
            contact_sheet_path=args.contact_sheet_path,
            frame_count=int(args.preview_frame_count),
            fps=float(args.fps),
        )
        print(json.dumps(preview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
