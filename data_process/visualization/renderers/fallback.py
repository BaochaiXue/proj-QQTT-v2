from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_image_flip(image: np.ndarray, image_flip: str) -> np.ndarray:
    if image_flip == "none":
        return image
    if image_flip == "vertical":
        return cv2.flip(image, 0)
    if image_flip == "horizontal":
        return cv2.flip(image, 1)
    if image_flip == "both":
        return cv2.flip(image, -1)
    raise ValueError(f"Unsupported image_flip: {image_flip}")


def look_at_view_matrix(camera_position: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
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


def _look_at(camera_position: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    return look_at_view_matrix(camera_position, center, up)


def _project_view_coordinates(
    xyz: np.ndarray,
    *,
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    z = -xyz[:, 2]
    if projection_mode == "perspective":
        focal = width / (2.0 * np.tan(np.deg2rad(35.0) / 2.0))
        u = (xyz[:, 0] * focal / z) + width * 0.5
        v = height * 0.5 - (xyz[:, 1] * focal / z)
        return u, v
    if projection_mode == "orthographic":
        scale = float(ortho_scale) if ortho_scale is not None else max(1e-6, float(np.max(np.abs(xyz[:, :2]))) * 1.2)
        u = (xyz[:, 0] / scale) * (width * 0.5) + width * 0.5
        v = height * 0.5 - (xyz[:, 1] / scale) * (height * 0.5)
        return u, v
    raise ValueError(f"Unsupported projection_mode: {projection_mode}")


def project_world_points_to_image(
    points: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    xyz_view = np.zeros((len(points), 3), dtype=np.float32)
    valid = np.zeros((len(points),), dtype=bool)
    if len(points) == 0:
        return {
            "uv": uv,
            "xyz_view": xyz_view,
            "valid": valid,
        }

    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    view = look_at_view_matrix(view_config["camera_position"], view_config["center"], view_config["up"])
    camera_points = homogeneous @ view.T
    xyz_view = camera_points[:, :3]
    valid = xyz_view[:, 2] < -1e-6
    if np.any(valid):
        u, v = _project_view_coordinates(
            xyz_view[valid],
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        uv[valid, 0] = u
        uv[valid, 1] = v

    return {
        "uv": uv,
        "xyz_view": xyz_view,
        "valid": valid,
    }


def estimate_ortho_scale(
    point_sets: list[np.ndarray],
    *,
    view_config: dict[str, Any],
    margin: float = 1.15,
) -> float:
    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        return 1.0
    stacked = np.concatenate(points, axis=0)
    homogeneous = np.concatenate(
        [stacked.astype(np.float32), np.ones((len(stacked), 1), dtype=np.float32)],
        axis=1,
    )
    view = _look_at(view_config["camera_position"], view_config["center"], view_config["up"])
    camera_points = homogeneous @ view.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] < -1e-6
    xyz = xyz[valid]
    if len(xyz) == 0:
        return 1.0
    scale = max(
        float(np.max(np.abs(xyz[:, 0]))),
        float(np.max(np.abs(xyz[:, 1]))),
        1e-3,
    )
    return float(scale * margin)


def _rasterize_view(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
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
    u, v = _project_view_coordinates(
        xyz,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return canvas
    u = np.rint(u[inside]).astype(np.int32)
    v = np.rint(v[inside]).astype(np.int32)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
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


def rasterize_point_cloud_view(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> dict[str, np.ndarray]:
    return _rasterize_view(
        points,
        colors,
        view_config=view_config,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )


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


def _densify_xyz_map(
    xyz_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    blur_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = max(3, blur_radius * 2 + 1)
    mask = valid_mask.astype(np.float32)
    blurred_mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=max(1.0, blur_radius * 0.75))
    densified = np.zeros_like(xyz_map, dtype=np.float32)
    for channel_idx in range(3):
        channel = np.asarray(xyz_map[..., channel_idx], dtype=np.float32) * mask
        blurred_channel = cv2.GaussianBlur(channel, (kernel, kernel), sigmaX=max(1.0, blur_radius * 0.75))
        valid = blurred_mask > 1e-4
        densified[..., channel_idx][valid] = blurred_channel[valid] / np.maximum(blurred_mask[valid], 1e-4)
    valid_dense = blurred_mask > 0.05
    return densified, valid_dense


def _render_normals(normals: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid_mask):
        rgb = ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
        canvas[valid_mask] = rgb[valid_mask][:, ::-1]
    return canvas


def _render_gray_shaded(normals: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid_mask):
        light_dir = np.asarray([0.30, -0.18, -1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        intensity = np.clip((normals @ light_dir) * 0.5 + 0.5, 0.0, 1.0)
        shaded = np.clip(58.0 + intensity * 178.0, 0.0, 255.0).astype(np.uint8)
        gray = np.stack([shaded, shaded, shaded], axis=2)
        canvas[valid_mask] = gray[valid_mask]
    return canvas


def _background_bgr_for_render_mode(render_mode: str) -> tuple[int, int, int]:
    if render_mode == "color_by_rgb":
        return (28, 28, 30)
    if render_mode == "neutral_gray_shaded":
        return (30, 34, 40)
    return (24, 26, 30)


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
    point_radius_px: int = 2,
    supersample_scale: int = 2,
    projection_mode: str = "perspective",
    ortho_scale: float | None = None,
) -> np.ndarray:
    ss = max(1, int(supersample_scale))
    ss_width = width * ss
    ss_height = height * ss
    raster = _rasterize_view(
        points,
        colors,
        view_config=view_config,
        width=ss_width,
        height=ss_height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    valid = raster["valid"]
    valid_render = valid.astype(np.float32)
    radius_ss = max(1, int(point_radius_px) * ss)
    if render_mode == "color_by_rgb":
        canvas = raster["rgb"]
    elif render_mode == "color_by_height":
        min_height, max_height = scalar_bounds["height"]
        canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
    elif render_mode == "color_by_depth":
        min_depth, max_depth = scalar_bounds["depth"]
        canvas = _colorize_scalar_map(raster["depth"], valid, min_value=min_depth, max_value=max_depth)
    else:
        densified_xyz, densified_valid = _densify_xyz_map(
            raster["xyz_view"],
            valid,
            blur_radius=radius_ss,
        )
        normals, valid_normals = _compute_normals_from_xyz_map(densified_xyz, densified_valid)
        if render_mode == "color_by_normals":
            if np.count_nonzero(valid_normals) < 512:
                min_height, max_height = scalar_bounds["height"]
                canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
            else:
                canvas = _render_normals(normals, valid_normals)
        else:
            if np.count_nonzero(valid_normals) < 512:
                min_height, max_height = scalar_bounds["height"]
                canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
            else:
                canvas = _render_gray_shaded(normals, valid_normals)

    if radius_ss > 1:
        kernel = radius_ss * 2 + 1
        mask = valid.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=max(1.0, radius_ss * 0.65))
        valid_render = blurred_mask
        if canvas.ndim == 3:
            blurred_canvas = cv2.GaussianBlur(canvas.astype(np.float32), (kernel, kernel), sigmaX=max(1.0, radius_ss * 0.65))
            canvas = np.zeros_like(canvas)
            valid_blur = blurred_mask > 1e-4
            if np.any(valid_blur):
                normalized = blurred_canvas / np.maximum(blurred_mask[..., None], 1e-4)
                canvas[valid_blur] = np.clip(normalized[valid_blur], 0.0, 255.0).astype(np.uint8)

    if ss > 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        valid_render = cv2.resize(valid_render.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)

    background = np.full(canvas.shape, _background_bgr_for_render_mode(render_mode), dtype=np.uint8)
    final_valid = valid_render > (1e-4 if radius_ss > 1 else 0.0)
    if np.any(final_valid):
        background[final_valid] = canvas[final_valid]
    return _apply_zoom(background, zoom_scale)


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
    point_radius_px: int = 2,
    supersample_scale: int = 2,
    projection_mode: str = "perspective",
    ortho_scale: float | None = None,
) -> tuple[np.ndarray, str]:
    if (
        renderer == "fallback"
        or render_mode != "color_by_rgb"
        or projection_mode != "perspective"
        or point_radius_px != 1
        or supersample_scale != 1
    ):
        return render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode=render_mode,
            scalar_bounds=scalar_bounds,
            width=width,
            height=height,
            zoom_scale=zoom_scale,
            point_radius_px=point_radius_px,
            supersample_scale=supersample_scale,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
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
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    ), "fallback"
