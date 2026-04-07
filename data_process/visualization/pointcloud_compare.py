from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration_io import load_calibration_transforms


RENDER_MODES = (
    "color_by_rgb",
    "color_by_depth",
    "color_by_height",
    "color_by_normals",
    "neutral_gray_shaded",
)
VIEW_NAMES = ("oblique", "top", "side")


def load_case_metadata(case_dir: Path) -> dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def decode_depth_to_meters(depth: np.ndarray, depth_scale_m_per_unit: float | None) -> np.ndarray:
    depth = np.asarray(depth)
    if np.issubdtype(depth.dtype, np.floating):
        depth_m = depth.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0] = 0.0
        return depth_m
    if depth_scale_m_per_unit is None:
        raise ValueError("depth_scale_m_per_unit is required to decode integer depth.")
    depth_m = depth.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_m[depth == 0] = 0.0
    return depth_m


def depth_to_camera_points(
    depth_m: np.ndarray,
    K_color: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
    color_image: np.ndarray,
    max_points_per_camera: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    color = np.asarray(color_image)
    if color.ndim == 2:
        color = np.tile(color[..., None], (1, 1, 3))

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    valid &= depth >= float(depth_min_m)
    valid &= depth <= float(depth_max_m)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"valid_depth_pixels": 0, "points_after_sampling": 0},
        )

    z = depth[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = color[valid].astype(np.uint8)

    if max_points_per_camera is not None and len(points) > int(max_points_per_camera):
        idx = np.linspace(0, len(points) - 1, int(max_points_per_camera), dtype=np.int32)
        points = points[idx]
        colors = colors[idx]

    return points, colors, {
        "valid_depth_pixels": valid_count,
        "points_after_sampling": int(len(points)),
    }


def transform_points(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float | None) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices], colors[unique_indices]


def write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[2])} {int(color[1])} {int(color[0])}\n"
            )


def resolve_case_dirs(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
) -> tuple[Path, Path, bool]:
    if case_name:
        case_dir = aligned_root / case_name
        return case_dir, case_dir, True
    if realsense_case and ffs_case:
        return aligned_root / realsense_case, aligned_root / ffs_case, False
    raise ValueError("Use either --case_name or both --realsense_case and --ffs_case.")


def get_frame_count(metadata: dict[str, Any]) -> int:
    return int(metadata["frame_num"])


def select_frame_indices(
    *,
    native_count: int,
    ffs_count: int,
    frame_start: int | None,
    frame_end: int | None,
    frame_stride: int,
) -> list[tuple[int, int]]:
    max_count = min(native_count, ffs_count)
    start = 0 if frame_start is None else max(0, int(frame_start))
    end = max_count - 1 if frame_end is None else min(int(frame_end), max_count - 1)
    if start > end:
        return []
    return [(idx, idx) for idx in range(start, end + 1, max(1, int(frame_stride)))]


def get_case_intrinsics(metadata: dict[str, Any]) -> list[np.ndarray]:
    matrices = metadata.get("K_color", metadata.get("intrinsics"))
    return [np.asarray(matrix, dtype=np.float32) for matrix in matrices]


def get_depth_scale_list(metadata: dict[str, Any], num_cameras: int) -> list[float | None]:
    scales = metadata.get("depth_scale_m_per_unit", [None] * num_cameras)
    if not isinstance(scales, list):
        scales = [scales for _ in range(num_cameras)]
    return scales


def choose_depth_stream(case_dir: Path, metadata: dict[str, Any], source: str, use_float_ffs_depth_when_available: bool) -> tuple[str, bool]:
    if source == "realsense":
        return "depth", False

    if use_float_ffs_depth_when_available and (case_dir / "depth_ffs_float_m").is_dir():
        return "depth_ffs_float_m", True
    if (case_dir / "depth_ffs").is_dir():
        return "depth_ffs", False
    return "depth", False


def load_case_frame_cloud(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    serials = metadata["serial_numbers"]
    intrinsics = get_case_intrinsics(metadata)
    depth_scales = get_depth_scale_list(metadata, len(serials))
    calibration_reference_serials = metadata.get("calibration_reference_serials", metadata["serial_numbers"])
    c2w_list = load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=serials,
        calibration_reference_serials=calibration_reference_serials,
    )
    depth_dir_name, use_float = choose_depth_stream(case_dir, metadata, depth_source, use_float_ffs_depth_when_available)

    fused_points = []
    fused_colors = []
    per_camera_stats = []
    for camera_idx, serial in enumerate(serials):
        color_path = case_dir / "color" / str(camera_idx) / f"{frame_idx}.png"
        depth_path = case_dir / depth_dir_name / str(camera_idx) / f"{frame_idx}.npy"
        color_image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_image is None:
            raise FileNotFoundError(f"Missing color frame: {color_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")
        depth_raw = np.load(depth_path)
        depth_m = decode_depth_to_meters(depth_raw, None if use_float else depth_scales[camera_idx])
        camera_points, camera_colors, stats = depth_to_camera_points(
            depth_m,
            intrinsics[camera_idx],
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            color_image=color_image,
            max_points_per_camera=max_points_per_camera,
        )
        world_points = transform_points(camera_points, c2w_list[camera_idx])
        fused_points.append(world_points)
        fused_colors.append(camera_colors)
        per_camera_stats.append(
            {
                "camera_idx": camera_idx,
                "serial": serial,
                "depth_dir_used": depth_dir_name,
                "used_float_depth": bool(use_float),
                **stats,
            }
        )

    if fused_points:
        points = np.concatenate(fused_points, axis=0) if len(fused_points) > 1 else fused_points[0]
        colors = np.concatenate(fused_colors, axis=0) if len(fused_colors) > 1 else fused_colors[0]
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    points, colors = voxel_downsample(points, colors, voxel_size)
    stats = {
        "per_camera": per_camera_stats,
        "fused_point_count": int(len(points)),
    }
    return points, colors, stats


def compute_view_config(bounds_min: np.ndarray, bounds_max: np.ndarray, view_name: str = "oblique") -> dict[str, Any]:
    center = (bounds_min + bounds_max) * 0.5
    extents = np.maximum(bounds_max - bounds_min, 1e-6)
    radius = float(np.linalg.norm(extents))
    if view_name == "top":
        camera_position = center + radius * np.array([0.0, 0.0, 1.8], dtype=np.float32)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    elif view_name == "side":
        camera_position = center + radius * np.array([0.0, -1.8, 0.35], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        azimuth = np.deg2rad(35.0)
        elevation = np.deg2rad(25.0)
        camera_position = center + radius * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float32,
        )
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "view_name": view_name,
        "center": center.astype(np.float32),
        "camera_position": camera_position.astype(np.float32),
        "up": up,
        "radius": radius,
    }


def _look_at(camera_position: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rotation = np.stack([right, true_up, -forward], axis=0)
    translation = -rotation @ camera_position
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = translation
    return view


def _rasterize_view(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    canvas = {
        "rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "depth": np.zeros((height, width), dtype=np.float32),
        "xyz_view": np.zeros((height, width, 3), dtype=np.float32),
        "world_z": np.zeros((height, width), dtype=np.float32),
        "valid": np.zeros((height, width), dtype=bool),
    }
    if len(points) == 0:
        return canvas

    view = _look_at(view_config["camera_position"], view_config["center"], view_config["up"])
    homogeneous = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera_points = homogeneous @ view.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] < -1e-6
    xyz = xyz[valid]
    world_points = points[valid]
    color_values = colors[valid]
    if len(xyz) == 0:
        return canvas

    z = -xyz[:, 2]
    focal = width / (2.0 * np.tan(np.deg2rad(35.0) / 2.0))
    u = (xyz[:, 0] * focal / z) + width * 0.5
    v = (xyz[:, 1] * focal / z) + height * 0.5
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return canvas
    u = np.rint(u[inside]).astype(np.int32)
    v = np.rint(v[inside]).astype(np.int32)
    z = z[inside]
    xyz = xyz[inside]
    world_points = world_points[inside]
    color_values = color_values[inside]

    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    z = z[order]
    xyz = xyz[order]
    world_points = world_points[order]
    color_values = color_values[order]

    canvas["rgb"][v, u] = color_values
    canvas["depth"][v, u] = z
    canvas["xyz_view"][v, u] = xyz
    canvas["world_z"][v, u] = world_points[:, 2]
    canvas["valid"][v, u] = True
    return canvas


def _colorize_scalar_map(
    scalar_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    canvas = np.zeros(scalar_map.shape + (3,), dtype=np.uint8)
    if np.any(valid_mask):
        normalized = np.clip((scalar_map - float(min_value)) / max(1e-6, float(max_value) - float(min_value)), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[valid_mask] = colored[valid_mask]
    return canvas


def _compute_normals_from_xyz_map(xyz_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros_like(xyz_map, dtype=np.float32)
    if xyz_map.shape[0] < 3 or xyz_map.shape[1] < 3:
        return normals, np.zeros(valid_mask.shape, dtype=bool)
    left = xyz_map[1:-1, :-2]
    right = xyz_map[1:-1, 2:]
    up = xyz_map[:-2, 1:-1]
    down = xyz_map[2:, 1:-1]
    valid_inner = (
        valid_mask[1:-1, 1:-1]
        & valid_mask[1:-1, :-2]
        & valid_mask[1:-1, 2:]
        & valid_mask[:-2, 1:-1]
        & valid_mask[2:, 1:-1]
    )
    dx = right - left
    dy = down - up
    inner_normals = np.cross(dx, dy)
    norm = np.linalg.norm(inner_normals, axis=2, keepdims=True)
    good = valid_inner & (norm[..., 0] > 1e-6)
    inner_normals[good] = inner_normals[good] / norm[good]
    flip = inner_normals[..., 2] > 0
    inner_normals[flip] *= -1.0
    normals[1:-1, 1:-1] = inner_normals
    valid_normals = np.zeros(valid_mask.shape, dtype=bool)
    valid_normals[1:-1, 1:-1] = good
    return normals, valid_normals


def _render_normals(normals: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid_mask):
        rgb = ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
        canvas[valid_mask] = rgb[valid_mask][:, ::-1]
    return canvas


def _render_gray_shaded(normals: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid_mask):
        light_dir = np.asarray([0.35, -0.25, -1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        intensity = np.clip((normals @ light_dir) * 0.5 + 0.5, 0.0, 1.0)
        shaded = np.clip(35.0 + intensity * 220.0, 0.0, 255.0).astype(np.uint8)
        gray = np.stack([shaded, shaded, shaded], axis=2)
        canvas[valid_mask] = gray[valid_mask]
    return canvas


def _apply_zoom(image: np.ndarray, zoom_scale: float) -> np.ndarray:
    if zoom_scale <= 1.0:
        return image
    height, width = image.shape[:2]
    crop_w = max(8, int(round(width / float(zoom_scale))))
    crop_h = max(8, int(round(height / float(zoom_scale))))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def render_point_cloud_fallback(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    render_mode: str,
    scalar_bounds: dict[str, tuple[float, float]],
    width: int = 960,
    height: int = 720,
    zoom_scale: float = 1.0,
) -> np.ndarray:
    raster = _rasterize_view(points, colors, view_config=view_config, width=width, height=height)
    valid = raster["valid"]
    if render_mode == "color_by_rgb":
        canvas = raster["rgb"]
    elif render_mode == "color_by_height":
        min_height, max_height = scalar_bounds["height"]
        canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
    elif render_mode == "color_by_depth":
        min_depth, max_depth = scalar_bounds["depth"]
        canvas = _colorize_scalar_map(raster["depth"], valid, min_value=min_depth, max_value=max_depth)
    else:
        normals, valid_normals = _compute_normals_from_xyz_map(raster["xyz_view"], valid)
        if render_mode == "color_by_normals":
            canvas = _render_normals(normals, valid_normals)
        else:
            canvas = _render_gray_shaded(normals, valid_normals)
    return _apply_zoom(canvas, zoom_scale)


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    renderer: str,
    view_config: dict[str, Any],
    render_mode: str,
    scalar_bounds: dict[str, tuple[float, float]],
    width: int = 960,
    height: int = 720,
    zoom_scale: float = 1.0,
) -> tuple[np.ndarray, str]:
    if renderer == "fallback" or render_mode != "color_by_rgb":
        return render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode=render_mode,
            scalar_bounds=scalar_bounds,
            width=width,
            height=height,
            zoom_scale=zoom_scale,
        ), "fallback"

    if renderer == "open3d" or renderer == "auto":
        try:
            import open3d as o3d
            from open3d.visualization import rendering

            if not hasattr(rendering, "OffscreenRenderer"):
                raise RuntimeError("Open3D offscreen renderer unavailable.")
            renderer_o3d = rendering.OffscreenRenderer(width, height)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0)[:, ::-1])
            renderer_o3d.scene.add_geometry("pcd", pcd, material)
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            center = view_config["center"]
            eye = view_config["camera_position"]
            up = view_config["up"]
            renderer_o3d.scene.camera.look_at(center, eye, up)
            renderer_o3d.scene.set_background([0.0, 0.0, 0.0, 1.0])
            image = np.asarray(renderer_o3d.render_to_image())
            renderer_o3d.release()
            return _apply_zoom(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), zoom_scale), "open3d"
        except Exception:
            if renderer == "open3d":
                raise
    return render_point_cloud_fallback(
        points,
        colors,
        view_config=view_config,
        render_mode=render_mode,
        scalar_bounds=scalar_bounds,
        width=width,
        height=height,
        zoom_scale=zoom_scale,
    ), "fallback"


def compose_panel(native_image: np.ndarray, ffs_image: np.ndarray, *, layout: str) -> np.ndarray:
    if layout == "stacked":
        return np.vstack([native_image, ffs_image])
    return np.hstack([native_image, ffs_image])


def write_video(video_path: Path, frame_paths: list[Path], fps: int) -> None:
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (first.shape[1], first.shape[0]),
    )
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        writer.write(image)
    writer.release()


def run_depth_comparison_workflow(
    *,
    aligned_root: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    output_dir: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = None,
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
    renderer: str = "auto",
    write_ply: bool = False,
    write_mp4: bool = False,
    fps: int = 30,
    panel_layout: str = "side_by_side",
    use_float_ffs_depth_when_available: bool = False,
    render_mode: str = "neutral_gray_shaded",
    views: list[str] | None = None,
    zoom_scale: float = 1.0,
) -> dict[str, Any]:
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported render_mode: {render_mode}")
    aligned_root = Path(aligned_root).resolve()
    output_dir = Path(output_dir).resolve()
    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=aligned_root,
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)

    if same_case_mode and not (native_case_dir / "depth_ffs").exists():
        raise ValueError("Same-case comparison requires an aligned case containing depth_ffs/.")

    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras for comparison.")

    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for comparison.")

    selected_views = views or ["oblique"]
    for view_name in selected_views:
        if view_name not in VIEW_NAMES:
            raise ValueError(f"Unsupported view: {view_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for native_frame_idx, ffs_frame_idx in frame_pairs:
        for source, case_dir, metadata, frame_idx in (
            ("native", native_case_dir, native_metadata, native_frame_idx),
            ("ffs", ffs_case_dir, ffs_metadata, ffs_frame_idx),
        ):
            points, colors, stats = load_case_frame_cloud(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=frame_idx,
                depth_source="realsense" if source == "native" else "ffs",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            cache[(source, frame_idx)] = (points, colors, stats)
            if len(points) > 0:
                bounds_min = np.minimum(bounds_min, points.min(axis=0))
                bounds_max = np.maximum(bounds_max, points.max(axis=0))

    if not np.isfinite(bounds_min).all():
        bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(float(np.linalg.norm(bounds_max - bounds_min)) * 2.0, 1.0)),
    }
    metrics = []
    renderer_used_by_view: dict[str, str] = {}

    for view_name in selected_views:
        view_output_dir = output_dir if len(selected_views) == 1 else output_dir / f"view_{view_name}"
        view_output_dir.mkdir(parents=True, exist_ok=True)
        native_frames_dir = view_output_dir / "native_frames"
        ffs_frames_dir = view_output_dir / "ffs_frames"
        side_frames_dir = view_output_dir / "side_by_side_frames"
        for directory in (native_frames_dir, ffs_frames_dir, side_frames_dir):
            directory.mkdir(parents=True, exist_ok=True)
        native_clouds_dir = view_output_dir / "native_clouds"
        ffs_clouds_dir = view_output_dir / "ffs_clouds"
        if write_ply:
            native_clouds_dir.mkdir(parents=True, exist_ok=True)
            ffs_clouds_dir.mkdir(parents=True, exist_ok=True)

        view_config = compute_view_config(bounds_min, bounds_max, view_name=view_name)
        renderer_used = None
        native_frame_paths = []
        ffs_frame_paths = []
        side_frame_paths = []

        for panel_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
            native_points, native_colors, native_stats = cache[("native", native_frame_idx)]
            ffs_points, ffs_colors, ffs_stats = cache[("ffs", ffs_frame_idx)]
            native_render, renderer_used = render_point_cloud(
                native_points,
                native_colors,
                renderer=renderer,
                view_config=view_config,
                render_mode=render_mode,
                scalar_bounds=scalar_bounds,
                zoom_scale=zoom_scale,
            )
            ffs_render, renderer_used = render_point_cloud(
                ffs_points,
                ffs_colors,
                renderer=renderer if renderer != "auto" else renderer_used or "auto",
                view_config=view_config,
                render_mode=render_mode,
                scalar_bounds=scalar_bounds,
                zoom_scale=zoom_scale,
            )
            side_render = compose_panel(native_render, ffs_render, layout=panel_layout)

            native_frame_path = native_frames_dir / f"{panel_idx:06d}.png"
            ffs_frame_path = ffs_frames_dir / f"{panel_idx:06d}.png"
            side_frame_path = side_frames_dir / f"{panel_idx:06d}.png"
            cv2.imwrite(str(native_frame_path), native_render)
            cv2.imwrite(str(ffs_frame_path), ffs_render)
            cv2.imwrite(str(side_frame_path), side_render)
            native_frame_paths.append(native_frame_path)
            ffs_frame_paths.append(ffs_frame_path)
            side_frame_paths.append(side_frame_path)

            if write_ply:
                write_ply_ascii(native_clouds_dir / f"{panel_idx:06d}.ply", native_points, native_colors)
                write_ply_ascii(ffs_clouds_dir / f"{panel_idx:06d}.ply", ffs_points, ffs_colors)

            metrics.append(
                {
                    "view_name": view_name,
                    "panel_frame_idx": panel_idx,
                    "native_frame_idx": native_frame_idx,
                    "ffs_frame_idx": ffs_frame_idx,
                    "native": native_stats,
                    "ffs": ffs_stats,
                }
            )

        renderer_used_by_view[view_name] = renderer_used or "fallback"
        videos_dir = view_output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if write_mp4:
            write_video(videos_dir / "native.mp4", native_frame_paths, fps)
            write_video(videos_dir / "ffs.mp4", ffs_frame_paths, fps)
            write_video(videos_dir / "side_by_side.mp4", side_frame_paths, fps)

    comparison_metadata = {
        "same_case_mode": same_case_mode,
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": frame_pairs,
        "renderer_requested": renderer,
        "renderer_used": renderer_used_by_view,
        "views": list(selected_views),
        "render_mode": render_mode,
        "panel_layout": panel_layout,
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "voxel_size": voxel_size,
        "max_points_per_camera": max_points_per_camera,
        "use_float_ffs_depth_when_available": use_float_ffs_depth_when_available,
        "zoom_scale": float(zoom_scale),
        "scalar_bounds": scalar_bounds,
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(comparison_metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "comparison_metadata": comparison_metadata,
        "metrics": metrics,
    }
