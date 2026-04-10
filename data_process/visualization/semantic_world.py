from __future__ import annotations

from typing import Any

import numpy as np


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32).reshape(3)
    return (vec / norm).astype(np.float32)


def _plane_residuals(
    points: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0:
        return np.empty((0,), dtype=np.float32)
    point = np.asarray(plane_point, dtype=np.float32).reshape(3)
    normal = _normalize(plane_normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return ((cloud - point[None, :]) @ normal).astype(np.float32)


def fit_dominant_plane(
    points: np.ndarray,
    *,
    max_iterations: int = 3,
    trim_quantile: float = 0.35,
) -> dict[str, Any]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) < 3:
        point = np.zeros((3,), dtype=np.float32)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return {
            "plane_point": point,
            "plane_normal": normal,
            "plane_mask": np.zeros((len(cloud),), dtype=bool),
            "plane_points": np.empty((0, 3), dtype=np.float32),
        }

    working = cloud
    for _ in range(max(1, int(max_iterations))):
        centroid = working.mean(axis=0).astype(np.float32)
        centered = working - centroid[None, :]
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = _normalize(vh[-1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        residuals = np.abs(_plane_residuals(cloud, plane_point=centroid, plane_normal=normal))
        threshold = float(np.quantile(residuals, float(np.clip(trim_quantile, 0.05, 0.95))))
        threshold = max(0.004, threshold)
        mask = residuals <= threshold
        if int(np.count_nonzero(mask)) < 3:
            break
        working = cloud[mask]
    centroid = working.mean(axis=0).astype(np.float32)
    centered = working - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize(vh[-1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    residuals = np.abs(_plane_residuals(cloud, plane_point=centroid, plane_normal=normal))
    mask = residuals <= max(0.006, float(np.quantile(residuals, 0.30)))
    return {
        "plane_point": centroid,
        "plane_normal": normal,
        "plane_mask": mask,
        "plane_points": cloud[mask].astype(np.float32),
    }


def infer_semantic_world_transform(
    *,
    scene_points: np.ndarray,
    camera_centers: np.ndarray,
    plane_point: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
) -> dict[str, Any]:
    cloud = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    camera_positions = np.asarray(camera_centers, dtype=np.float32).reshape(-1, 3)
    if len(camera_positions) == 0:
        raise ValueError("infer_semantic_world_transform requires at least one camera center.")

    if plane_point is None or plane_normal is None:
        plane_fit = fit_dominant_plane(cloud)
        plane_point_raw = np.asarray(plane_fit["plane_point"], dtype=np.float32)
        plane_normal_raw = np.asarray(plane_fit["plane_normal"], dtype=np.float32)
        plane_points = np.asarray(plane_fit["plane_points"], dtype=np.float32)
    else:
        plane_point_raw = np.asarray(plane_point, dtype=np.float32).reshape(3)
        plane_normal_raw = _normalize(plane_normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        residuals = np.abs(_plane_residuals(cloud, plane_point=plane_point_raw, plane_normal=plane_normal_raw))
        mask = residuals <= max(0.008, float(np.quantile(residuals, 0.20))) if len(residuals) > 0 else np.zeros((0,), dtype=bool)
        plane_points = cloud[mask] if np.any(mask) else np.empty((0, 3), dtype=np.float32)

    if len(plane_points) > 0:
        table_point = plane_points.mean(axis=0).astype(np.float32)
    else:
        table_point = plane_point_raw.astype(np.float32)
    mean_camera_center = camera_positions.mean(axis=0).astype(np.float32)

    plane_normal_flipped = plane_normal_raw.copy()
    if float((mean_camera_center - table_point) @ plane_normal_flipped) < 0.0:
        plane_normal_flipped = -plane_normal_flipped
    semantic_z = _normalize(plane_normal_flipped, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    calibration_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_candidate = calibration_x - semantic_z * float(calibration_x @ semantic_z)
    if float(np.linalg.norm(x_candidate)) <= 1e-6:
        source_points = plane_points if len(plane_points) >= 3 else cloud
        if len(source_points) >= 3:
            centered = source_points - source_points.mean(axis=0, keepdims=True)
            projected = centered - np.outer(centered @ semantic_z, semantic_z)
            _, _, vh = np.linalg.svd(projected, full_matrices=False)
            x_candidate = vh[0]
        else:
            x_candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    semantic_x = _normalize(x_candidate, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    semantic_y = _normalize(np.cross(semantic_z, semantic_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    semantic_x = _normalize(np.cross(semantic_y, semantic_z), np.array([1.0, 0.0, 0.0], dtype=np.float32))

    rotation = np.stack([semantic_x, semantic_y, semantic_z], axis=0).astype(np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ table_point)
    inverse_transform = np.eye(4, dtype=np.float32)
    inverse_transform[:3, :3] = rotation.T
    inverse_transform[:3, 3] = table_point

    return {
        "plane_point": table_point,
        "plane_normal_raw": plane_normal_raw.astype(np.float32),
        "plane_normal_flipped": plane_normal_flipped.astype(np.float32),
        "semantic_axes": {
            "x": semantic_x,
            "y": semantic_y,
            "z": semantic_z,
        },
        "mean_camera_center": mean_camera_center,
        "transform": transform,
        "inverse_transform": inverse_transform,
    }


def transform_points_to_semantic(points: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0:
        return np.empty((0, 3), dtype=np.float32)
    transform = np.asarray(frame["transform"], dtype=np.float32).reshape(4, 4)
    homogeneous = np.concatenate([cloud, np.ones((len(cloud), 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3].astype(np.float32)


def transform_vectors_to_semantic(vectors: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotation = np.asarray(frame["transform"], dtype=np.float32)[:3, :3]
    return (values @ rotation.T).astype(np.float32)


def transform_c2w_to_semantic(c2w: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(frame["transform"], dtype=np.float32).reshape(4, 4)
    return (transform @ np.asarray(c2w, dtype=np.float32).reshape(4, 4)).astype(np.float32)


def transform_c2w_list_to_semantic(c2w_list: list[np.ndarray], frame: dict[str, Any]) -> list[np.ndarray]:
    return [transform_c2w_to_semantic(item, frame) for item in c2w_list]


def transform_camera_clouds_to_semantic(
    camera_clouds: list[dict[str, Any]],
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    transformed_clouds: list[dict[str, Any]] = []
    for camera_cloud in camera_clouds:
        transformed_cloud = {
            **camera_cloud,
            "points": transform_points_to_semantic(np.asarray(camera_cloud["points"], dtype=np.float32), frame),
        }
        if "c2w" in camera_cloud and camera_cloud["c2w"] is not None:
            transformed_cloud["c2w"] = transform_c2w_to_semantic(np.asarray(camera_cloud["c2w"], dtype=np.float32), frame)
        transformed_clouds.append(transformed_cloud)
    return transformed_clouds


def transform_bounds_to_semantic(bounds: dict[str, np.ndarray], frame: dict[str, Any]) -> dict[str, np.ndarray]:
    minimum = np.asarray(bounds["min"], dtype=np.float32).reshape(3)
    maximum = np.asarray(bounds["max"], dtype=np.float32).reshape(3)
    corners = []
    for x_value in (minimum[0], maximum[0]):
        for y_value in (minimum[1], maximum[1]):
            for z_value in (minimum[2], maximum[2]):
                corners.append(np.array([x_value, y_value, z_value], dtype=np.float32))
    transformed = transform_points_to_semantic(np.stack(corners, axis=0), frame)
    return {
        "min": transformed.min(axis=0).astype(np.float32),
        "max": transformed.max(axis=0).astype(np.float32),
    }


def transform_scene_to_semantic(scene: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    transformed = dict(scene)
    point_keys = (
        "native_points",
        "native_render_points",
        "native_object_points",
        "native_context_points",
        "ffs_points",
        "ffs_render_points",
        "ffs_object_points",
        "ffs_context_points",
    )
    cloud_keys = (
        "native_camera_clouds",
        "native_render_camera_clouds",
        "native_object_camera_clouds",
        "native_context_camera_clouds",
        "ffs_camera_clouds",
        "ffs_render_camera_clouds",
        "ffs_object_camera_clouds",
        "ffs_context_camera_clouds",
    )
    for key in point_keys:
        if key in scene:
            transformed[key] = transform_points_to_semantic(np.asarray(scene[key], dtype=np.float32), frame)
    for key in cloud_keys:
        if key in scene:
            transformed[key] = transform_camera_clouds_to_semantic(list(scene[key]), frame)
    for key in ("focus_point", "plane_point"):
        if key in scene and scene[key] is not None:
            transformed[key] = transform_points_to_semantic(np.asarray(scene[key], dtype=np.float32).reshape(1, 3), frame)[0]
    if "plane_normal" in scene and scene["plane_normal"] is not None:
        transformed["plane_normal"] = transform_vectors_to_semantic(np.asarray(scene["plane_normal"], dtype=np.float32).reshape(1, 3), frame)[0]
    if "bounds_min" in scene and "bounds_max" in scene:
        transformed_bounds = transform_bounds_to_semantic(
            {"min": np.asarray(scene["bounds_min"], dtype=np.float32), "max": np.asarray(scene["bounds_max"], dtype=np.float32)},
            frame,
        )
        transformed["bounds_min"] = transformed_bounds["min"]
        transformed["bounds_max"] = transformed_bounds["max"]
    if "render_bounds_min" in scene and "render_bounds_max" in scene:
        transformed_render_bounds = transform_bounds_to_semantic(
            {"min": np.asarray(scene["render_bounds_min"], dtype=np.float32), "max": np.asarray(scene["render_bounds_max"], dtype=np.float32)},
            frame,
        )
        transformed["render_bounds_min"] = transformed_render_bounds["min"]
        transformed["render_bounds_max"] = transformed_render_bounds["max"]
    if "crop_bounds" in scene and scene["crop_bounds"] is not None:
        transformed["crop_bounds"] = transform_bounds_to_semantic(scene["crop_bounds"], frame)
    if "object_roi_bounds" in scene and scene["object_roi_bounds"] is not None:
        transformed["object_roi_bounds"] = transform_bounds_to_semantic(scene["object_roi_bounds"], frame)
    render_min = np.asarray(
        transformed.get("render_bounds_min", transformed.get("bounds_min", np.array([-0.1, -0.1, -0.1], dtype=np.float32))),
        dtype=np.float32,
    )
    render_max = np.asarray(
        transformed.get("render_bounds_max", transformed.get("bounds_max", np.array([0.1, 0.1, 0.1], dtype=np.float32))),
        dtype=np.float32,
    )
    depth_bounds = scene.get("scalar_bounds", {}).get("depth", (0.0, 1.0))
    transformed["scalar_bounds"] = {
        "height": (
            float(render_min[2]),
            float(render_max[2]),
        ),
        "depth": depth_bounds,
    }
    return transformed
