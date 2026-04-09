from __future__ import annotations

import numpy as np


def estimate_focus_point(
    point_sets: list[np.ndarray],
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    focus_mode: str,
) -> np.ndarray:
    default_center = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    if focus_mode == "none":
        return default_center

    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        return default_center

    stacked = np.concatenate(points, axis=0)
    if len(stacked) == 0:
        return default_center

    if focus_mode == "table":
        z_low = float(np.percentile(stacked[:, 2], 20))
        z_high = float(np.percentile(stacked[:, 2], 65))
        band = stacked[(stacked[:, 2] >= z_low) & (stacked[:, 2] <= z_high)]
        if len(band) < 128:
            band = stacked
        return np.asarray(
            [
                float(np.median(band[:, 0])),
                float(np.median(band[:, 1])),
                float(np.median(band[:, 2])),
            ],
            dtype=np.float32,
        )

    raise ValueError(f"Unsupported focus_mode: {focus_mode}")


def compute_scene_crop_bounds(
    point_sets: list[np.ndarray],
    *,
    focus_point: np.ndarray,
    scene_crop_mode: str,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
    manual_xyz_roi: dict[str, float] | None = None,
    object_seed_point_sets: list[np.ndarray] | None = None,
    object_height_min: float = 0.02,
    object_height_max: float = 0.30,
    object_component_mode: str = "graph_union",
    object_component_topk: int = 2,
) -> dict[str, np.ndarray]:
    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        focus = np.asarray(focus_point, dtype=np.float32)
        return {
            "mode": scene_crop_mode,
            "min": focus - np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "max": focus + np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }

    stacked = np.concatenate(points, axis=0)
    full_min = stacked.min(axis=0)
    full_max = stacked.max(axis=0)

    if scene_crop_mode == "none":
        return {"mode": scene_crop_mode, "min": full_min.astype(np.float32), "max": full_max.astype(np.float32)}

    if scene_crop_mode == "manual_xyz_roi":
        if manual_xyz_roi is None:
            raise ValueError("manual_xyz_roi crop mode requires explicit roi bounds.")
        crop_min = np.array(
            [manual_xyz_roi["x_min"], manual_xyz_roi["y_min"], manual_xyz_roi["z_min"]],
            dtype=np.float32,
        )
        crop_max = np.array(
            [manual_xyz_roi["x_max"], manual_xyz_roi["y_max"], manual_xyz_roi["z_max"]],
            dtype=np.float32,
        )
        if np.any(crop_min >= crop_max):
            raise ValueError(f"Invalid manual_xyz_roi bounds: {manual_xyz_roi}")
        return {"mode": scene_crop_mode, "min": crop_min, "max": crop_max}

    if scene_crop_mode == "auto_object_bbox":
        from .object_roi import estimate_object_roi_bounds

        table_bounds = compute_scene_crop_bounds(
            point_sets,
            focus_point=focus_point,
            scene_crop_mode="auto_table_bbox",
            crop_margin_xy=crop_margin_xy,
            crop_min_z=crop_min_z,
            crop_max_z=crop_max_z,
            manual_xyz_roi=None,
        )
        table_valid = (
            np.all(stacked >= table_bounds["min"][None, :], axis=1)
            & np.all(stacked <= table_bounds["max"][None, :], axis=1)
        )
        table_points = stacked[table_valid]
        seed_points = None
        if object_seed_point_sets:
            seed_sets = [np.asarray(item, dtype=np.float32) for item in object_seed_point_sets if len(item) > 0]
            if seed_sets:
                seed_stacked = np.concatenate(seed_sets, axis=0)
                seed_valid = (
                    np.all(seed_stacked >= table_bounds["min"][None, :], axis=1)
                    & np.all(seed_stacked <= table_bounds["max"][None, :], axis=1)
                )
                seed_points = seed_stacked[seed_valid]
        if seed_points is not None and len(seed_points) >= 32:
            seed_object_roi = estimate_object_roi_bounds(
                seed_points,
                fallback_bounds=table_bounds,
                full_bounds={"min": full_min.astype(np.float32), "max": full_max.astype(np.float32)},
                plane_reference_points=table_points if len(table_points) > 0 else stacked,
                object_height_min=float(object_height_min),
                object_height_max=max(0.40, float(object_height_max)),
                object_component_mode=object_component_mode,
                object_component_topk=int(object_component_topk),
                roi_margin_xy=max(0.02, float(crop_margin_xy) * 0.45),
                roi_margin_z=max(0.015, abs(float(crop_max_z) - float(crop_min_z)) * 0.08),
            )
            seed_object_roi["seed_bbox_used"] = True
            seed_object_roi["seed_source_point_count"] = int(len(seed_points))
            return seed_object_roi
        object_roi = estimate_object_roi_bounds(
            seed_points if seed_points is not None and len(seed_points) > 0 else (table_points if len(table_points) > 0 else stacked),
            fallback_bounds=table_bounds,
            full_bounds={"min": full_min.astype(np.float32), "max": full_max.astype(np.float32)},
            plane_reference_points=table_points if len(table_points) > 0 else stacked,
            object_height_min=float(object_height_min),
            object_height_max=float(object_height_max),
            object_component_mode=object_component_mode,
            object_component_topk=int(object_component_topk),
            roi_margin_xy=max(0.02, float(crop_margin_xy) * 0.45),
            roi_margin_z=max(0.015, abs(float(crop_max_z) - float(crop_min_z)) * 0.08),
        )
        return object_roi

    if scene_crop_mode != "auto_table_bbox":
        raise ValueError(f"Unsupported scene_crop_mode: {scene_crop_mode}")

    focus = np.asarray(focus_point, dtype=np.float32)
    z_min = float(focus[2] + crop_min_z)
    z_max = float(focus[2] + crop_max_z)
    band = stacked[(stacked[:, 2] >= z_min) & (stacked[:, 2] <= z_max)]
    if len(band) < 256:
        band = stacked[np.abs(stacked[:, 2] - focus[2]) <= max(abs(crop_min_z), abs(crop_max_z), 0.35)]
    if len(band) < 64:
        band = stacked

    x_min, x_max = np.quantile(band[:, 0], [0.05, 0.95])
    y_min, y_max = np.quantile(band[:, 1], [0.05, 0.95])
    crop_min = np.array(
        [
            float(x_min - crop_margin_xy),
            float(y_min - crop_margin_xy),
            z_min,
        ],
        dtype=np.float32,
    )
    crop_max = np.array(
        [
            float(x_max + crop_margin_xy),
            float(y_max + crop_margin_xy),
            z_max,
        ],
        dtype=np.float32,
    )
    crop_min = np.maximum(crop_min, full_min)
    crop_max = np.minimum(crop_max, full_max)
    return {"mode": scene_crop_mode, "min": crop_min, "max": crop_max}


def crop_points_to_bounds(
    points: np.ndarray,
    colors: np.ndarray,
    crop_bounds: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_bounds is None or len(points) == 0:
        return points, colors
    crop_min = np.asarray(crop_bounds["min"], dtype=np.float32)
    crop_max = np.asarray(crop_bounds["max"], dtype=np.float32)
    valid = np.all(points >= crop_min[None, :], axis=1) & np.all(points <= crop_max[None, :], axis=1)
    return points[valid], colors[valid]
