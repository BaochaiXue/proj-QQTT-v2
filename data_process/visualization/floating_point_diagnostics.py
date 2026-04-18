from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from .depth_diagnostics import compose_grid, get_case_camera_transform, label_tile, load_color_frame, resolve_camera_ids
from .io_artifacts import write_json
from .io_case import (
    get_case_intrinsics,
    get_frame_count,
    load_case_metadata,
    load_depth_frame,
    resolve_case_dirs,
    select_frame_indices,
)


PRIMARY_CAUSE_PRIORITY = ("occlusion", "edge", "dark", "other")
PRIMARY_CAUSE_COLORS_BGR = {
    "occlusion": (0, 0, 255),
    "edge": (0, 220, 255),
    "dark": (255, 128, 0),
    "other": (220, 220, 220),
}
PRIMARY_CAUSE_LABELS = {
    "occlusion": "OCC",
    "edge": "EDGE",
    "dark": "DARK",
    "other": "OTHER",
}


def assign_primary_cause(
    *,
    occluded_in_other_views: bool,
    near_edge: bool,
    dark_region: bool,
) -> str:
    if bool(occluded_in_other_views):
        return "occlusion"
    if bool(near_edge):
        return "edge"
    if bool(dark_region):
        return "dark"
    return "other"


def detect_radius_outlier_indices(
    points_world: np.ndarray,
    *,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    cloud = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    point_count = int(len(cloud))
    if point_count == 0:
        empty = np.empty((0,), dtype=np.int32)
        return {"inlier_indices": empty, "outlier_indices": empty}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    _, inlier_indices = pcd.remove_radius_outlier(
        nb_points=int(nb_points),
        radius=float(radius_m),
    )
    inliers = np.asarray(inlier_indices, dtype=np.int32).reshape(-1)
    if len(inliers) == 0:
        return {
            "inlier_indices": inliers,
            "outlier_indices": np.arange(point_count, dtype=np.int32),
        }
    keep_mask = np.zeros((point_count,), dtype=bool)
    keep_mask[inliers] = True
    outliers = np.flatnonzero(~keep_mask).astype(np.int32)
    return {"inlier_indices": inliers, "outlier_indices": outliers}


def _backproject_depth_with_metadata(
    *,
    depth_m: np.ndarray,
    K_color: np.ndarray,
    color_image: np.ndarray,
    c2w: np.ndarray,
) -> dict[str, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    camera_points = np.stack([x, y, z], axis=1).astype(np.float32)

    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    homogeneous = np.concatenate(
        [camera_points, np.ones((len(camera_points), 1), dtype=np.float32)],
        axis=1,
    )
    world_points = (homogeneous @ transform.T)[:, :3].astype(np.float32)
    source_pixel_uv = np.stack([xx[valid], yy[valid]], axis=1).astype(np.int32)
    colors = np.asarray(color_image, dtype=np.uint8)[valid].astype(np.uint8)
    return {
        "points": world_points,
        "colors": colors,
        "source_pixel_uv": source_pixel_uv,
        "source_depth_m": z.astype(np.float32),
    }


def load_frame_camera_clouds_with_metadata(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    ffs_native_like_postprocess: bool = False,
    camera_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    serial_numbers = metadata["serial_numbers"]
    intrinsics = get_case_intrinsics(metadata)
    c2w_list = get_case_camera_transform(case_dir=case_dir, metadata=metadata)
    selected_camera_ids = resolve_camera_ids(metadata, camera_ids)
    camera_clouds: list[dict[str, Any]] = []
    for camera_idx in selected_camera_ids:
        color_image = load_color_frame(case_dir, camera_idx, frame_idx)
        _, depth_m, depth_info = load_depth_frame(
            case_dir=case_dir,
            metadata=metadata,
            camera_idx=camera_idx,
            frame_idx=frame_idx,
            depth_source=depth_source,
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            ffs_native_like_postprocess=ffs_native_like_postprocess,
        )
        point_payload = _backproject_depth_with_metadata(
            depth_m=depth_m,
            K_color=intrinsics[camera_idx],
            color_image=color_image,
            c2w=c2w_list[camera_idx],
        )
        point_count = int(len(point_payload["points"]))
        camera_clouds.append(
            {
                "camera_idx": int(camera_idx),
                "serial": str(serial_numbers[camera_idx]),
                "K_color": np.asarray(intrinsics[camera_idx], dtype=np.float32),
                "c2w": np.asarray(c2w_list[camera_idx], dtype=np.float32),
                "color_image": np.asarray(color_image, dtype=np.uint8),
                "depth_m": np.asarray(depth_m, dtype=np.float32),
                "points": point_payload["points"],
                "colors": point_payload["colors"],
                "source_pixel_uv": point_payload["source_pixel_uv"],
                "source_depth_m": point_payload["source_depth_m"],
                "source_camera_idx": np.full((point_count,), int(camera_idx), dtype=np.int16),
                "source_serial": np.full((point_count,), str(serial_numbers[camera_idx]), dtype=object),
                **depth_info,
            }
        )
    return camera_clouds


def _concat_array_fields(camera_clouds: list[dict[str, Any]], key: str, *, dtype: Any) -> np.ndarray:
    arrays = [np.asarray(item[key], dtype=dtype) for item in camera_clouds if len(item["points"]) > 0]
    if not arrays:
        return np.empty((0,), dtype=dtype)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def _concat_point_fields(camera_clouds: list[dict[str, Any]], key: str, *, dtype: Any, width: int) -> np.ndarray:
    arrays = [np.asarray(item[key], dtype=dtype).reshape(-1, width) for item in camera_clouds if len(item["points"]) > 0]
    if not arrays:
        return np.empty((0, width), dtype=dtype)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def _fuse_camera_clouds(camera_clouds: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "points": _concat_point_fields(camera_clouds, "points", dtype=np.float32, width=3),
        "colors": _concat_point_fields(camera_clouds, "colors", dtype=np.uint8, width=3),
        "source_pixel_uv": _concat_point_fields(camera_clouds, "source_pixel_uv", dtype=np.int32, width=2),
        "source_depth_m": _concat_array_fields(camera_clouds, "source_depth_m", dtype=np.float32),
        "source_camera_idx": _concat_array_fields(camera_clouds, "source_camera_idx", dtype=np.int16),
        "source_serial": _concat_array_fields(camera_clouds, "source_serial", dtype=object),
    }


def _build_edge_band_mask(
    color_image: np.ndarray,
    depth_m: np.ndarray,
    *,
    edge_band_px: int,
) -> dict[str, np.ndarray]:
    rgb = np.asarray(color_image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(gray, 60, 160) > 0

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth_vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        min_depth = float(np.percentile(depth[valid], 5))
        max_depth = float(np.percentile(depth[valid], 95))
        normalized = np.clip(
            (depth - min_depth) / max(1e-6, max_depth - min_depth),
            0.0,
            1.0,
        )
        depth_vis = np.rint(normalized * 255.0).astype(np.uint8)
    depth_edges = cv2.Canny(depth_vis, 40, 120) > 0
    combined_edges = rgb_edges | depth_edges
    if int(edge_band_px) > 0:
        kernel_size = int(edge_band_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edge_band = cv2.dilate(combined_edges.astype(np.uint8), kernel) > 0
    else:
        edge_band = combined_edges
    return {
        "gray": gray,
        "rgb_edges": rgb_edges,
        "depth_edges": depth_edges,
        "edge_band": edge_band,
    }


def _patch_mean(gray_image: np.ndarray, *, u: int, v: int, radius: int) -> float:
    height, width = gray_image.shape[:2]
    x0 = max(0, int(u) - int(radius))
    x1 = min(width, int(u) + int(radius) + 1)
    y0 = max(0, int(v) - int(radius))
    y1 = min(height, int(v) + int(radius) + 1)
    patch = gray_image[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def _project_world_point_to_camera(
    world_point: np.ndarray,
    *,
    c2w: np.ndarray,
    K_color: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    point = np.asarray(world_point, dtype=np.float32).reshape(3)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32).reshape(4, 4))
    homogeneous = np.concatenate([point, np.ones((1,), dtype=np.float32)], axis=0)
    camera_point = (homogeneous @ w2c.T)[:3]
    z = float(camera_point[2])
    if z <= 1e-6:
        return {"inside": False}
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    uvw = camera_point @ K.T
    u = int(np.rint(float(uvw[0] / max(uvw[2], 1e-6))))
    v = int(np.rint(float(uvw[1] / max(uvw[2], 1e-6))))
    height, width = image_shape
    inside = 0 <= u < int(width) and 0 <= v < int(height)
    return {
        "inside": bool(inside),
        "u": int(u),
        "v": int(v),
        "projected_z": float(z),
    }


def classify_cross_view_relation(
    world_point: np.ndarray,
    *,
    source_camera_idx: int,
    camera_clouds: list[dict[str, Any]],
    occlusion_depth_tol_m: float,
    occlusion_depth_tol_ratio: float,
) -> dict[str, Any]:
    support_count = 0
    occluded_count = 0
    projected_count = 0
    observed_count = 0
    per_camera: list[dict[str, Any]] = []
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        if camera_idx == int(source_camera_idx):
            continue
        projection = _project_world_point_to_camera(
            world_point,
            c2w=camera_cloud["c2w"],
            K_color=camera_cloud["K_color"],
            image_shape=np.asarray(camera_cloud["depth_m"]).shape,
        )
        if not bool(projection.get("inside", False)):
            per_camera.append({"camera_idx": camera_idx, "relation": "outside"})
            continue
        projected_count += 1
        u = int(projection["u"])
        v = int(projection["v"])
        target_depth = float(np.asarray(camera_cloud["depth_m"], dtype=np.float32)[v, u])
        if not np.isfinite(target_depth) or target_depth <= 0.0:
            per_camera.append({"camera_idx": camera_idx, "relation": "invalid_depth"})
            continue
        observed_count += 1
        projected_z = float(projection["projected_z"])
        tolerance = max(float(occlusion_depth_tol_m), float(occlusion_depth_tol_ratio) * projected_z)
        delta = target_depth - projected_z
        relation = "behind"
        if abs(delta) <= tolerance:
            support_count += 1
            relation = "supported"
        elif target_depth < projected_z - tolerance:
            occluded_count += 1
            relation = "occluded"
        per_camera.append(
            {
                "camera_idx": camera_idx,
                "relation": relation,
                "target_depth_m": float(target_depth),
                "projected_z_m": float(projected_z),
            }
        )
    return {
        "support_count_other_views": int(support_count),
        "cross_view_supported": bool(support_count > 0),
        "occluded_in_other_views": bool(occluded_count > 0 and support_count == 0),
        "projected_other_view_count": int(projected_count),
        "observed_other_view_count": int(observed_count),
        "per_camera": per_camera,
    }


def _histogram(records: list[dict[str, Any]], key: str, *, values: list[str] | None = None) -> dict[str, int]:
    histogram = {str(value): 0 for value in (values or [])}
    for record in records:
        item = str(record[key])
        histogram[item] = histogram.get(item, 0) + 1
    return histogram


def _ratio_histogram(counts: dict[str, int], *, total: int) -> dict[str, float]:
    if int(total) <= 0:
        return {str(key): 0.0 for key in counts}
    return {str(key): float(value) / float(total) for key, value in counts.items()}


def _summarize_outlier_records(
    *,
    source_name: str,
    frame_idx: int,
    panel_frame_idx: int,
    total_point_count: int,
    records: list[dict[str, Any]],
    selected_camera_ids: list[int],
) -> dict[str, Any]:
    outlier_point_count = int(len(records))
    primary_cause_histogram = _histogram(records, "primary_cause", values=list(PRIMARY_CAUSE_PRIORITY))
    source_camera_histogram = _histogram(records, "source_camera_idx", values=[str(camera_idx) for camera_idx in selected_camera_ids])
    support_histogram = _histogram(records, "support_count_other_views", values=["0", "1", "2"])
    near_edge_count = int(sum(1 for record in records if bool(record["near_edge"])))
    dark_region_count = int(sum(1 for record in records if bool(record["dark_region"])))
    occlusion_count = int(sum(1 for record in records if bool(record["occluded_in_other_views"])))
    cross_view_supported_count = int(sum(1 for record in records if bool(record["cross_view_supported"])))
    total = max(1, outlier_point_count)
    return {
        "source": source_name,
        "frame_idx": int(frame_idx),
        "panel_frame_idx": int(panel_frame_idx),
        "total_point_count": int(total_point_count),
        "outlier_point_count": int(outlier_point_count),
        "outlier_ratio": float(outlier_point_count / max(1, int(total_point_count))),
        "primary_cause_histogram": primary_cause_histogram,
        "primary_cause_ratio": _ratio_histogram(primary_cause_histogram, total=outlier_point_count),
        "source_camera_histogram": source_camera_histogram,
        "source_camera_ratio": _ratio_histogram(source_camera_histogram, total=outlier_point_count),
        "support_count_other_views_histogram": support_histogram,
        "support_count_other_views_ratio": _ratio_histogram(support_histogram, total=outlier_point_count),
        "near_edge_count": near_edge_count,
        "near_edge_ratio": float(near_edge_count / total),
        "dark_region_count": dark_region_count,
        "dark_region_ratio": float(dark_region_count / total),
        "occluded_in_other_views_count": occlusion_count,
        "occluded_in_other_views_ratio": float(occlusion_count / total),
        "cross_view_supported_count": cross_view_supported_count,
        "cross_view_supported_ratio": float(cross_view_supported_count / total),
    }


def _render_color_overlay(
    color_image: np.ndarray,
    records: list[dict[str, Any]],
) -> np.ndarray:
    base = np.asarray(color_image, dtype=np.uint8).copy()
    if not records:
        return base
    overlay = base.copy()
    for record in records:
        u, v = [int(item) for item in record["source_pixel_uv"]]
        color = PRIMARY_CAUSE_COLORS_BGR[str(record["primary_cause"])]
        cv2.circle(overlay, (u, v), 4, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (u, v), 7, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.70, base, 0.30, 0.0)


def _render_cause_heatmap(
    color_image: np.ndarray,
    records: list[dict[str, Any]],
) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(color_image, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = np.clip(canvas.astype(np.float32) * 0.35, 0.0, 255.0).astype(np.uint8)
    if not records:
        return canvas
    height, width = gray.shape[:2]
    masks = {
        cause: np.zeros((height, width), dtype=np.uint8)
        for cause in PRIMARY_CAUSE_PRIORITY
    }
    for record in records:
        u, v = [int(item) for item in record["source_pixel_uv"]]
        if 0 <= u < width and 0 <= v < height:
            masks[str(record["primary_cause"])][v, u] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for cause in ("other", "dark", "edge", "occlusion"):
        dilated = cv2.dilate(masks[cause], kernel) > 0
        canvas[dilated] = np.asarray(PRIMARY_CAUSE_COLORS_BGR[cause], dtype=np.uint8)
    return canvas


def _camera_records(records: list[dict[str, Any]], camera_idx: int) -> list[dict[str, Any]]:
    return [record for record in records if int(record["source_camera_idx"]) == int(camera_idx)]


def _short_cause_counts(records: list[dict[str, Any]]) -> str:
    counts = _histogram(records, "primary_cause", values=list(PRIMARY_CAUSE_PRIORITY))
    return " ".join(
        f"{PRIMARY_CAUSE_LABELS[cause]}={counts[cause]}"
        for cause in PRIMARY_CAUSE_PRIORITY
    )


def _annotate_source_board(
    board: np.ndarray,
    *,
    source_name: str,
    frame_idx: int,
    outlier_point_count: int,
    primary_cause_histogram: dict[str, int],
) -> np.ndarray:
    canvas = np.asarray(board, dtype=np.uint8)
    header_h = 34
    framed = np.zeros((canvas.shape[0] + header_h, canvas.shape[1], 3), dtype=np.uint8)
    framed[header_h:] = canvas
    cv2.rectangle(framed, (0, 0), (framed.shape[1] - 1, header_h - 1), (18, 18, 22), -1)
    summary = " ".join(
        f"{PRIMARY_CAUSE_LABELS[cause]}={int(primary_cause_histogram.get(cause, 0))}"
        for cause in PRIMARY_CAUSE_PRIORITY
    )
    text = f"{source_name} | frame={frame_idx} | outliers={outlier_point_count} | {summary}"
    cv2.putText(
        framed,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return framed


def _compose_source_board(
    *,
    source_name: str,
    frame_idx: int,
    camera_clouds: list[dict[str, Any]],
    records: list[dict[str, Any]],
    tile_size: tuple[int, int],
    source_metrics: dict[str, Any],
) -> np.ndarray:
    overlay_tiles: list[np.ndarray] = []
    heatmap_tiles: list[np.ndarray] = []
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        camera_records = _camera_records(records, camera_idx)
        overlay_tiles.append(
            label_tile(
                _render_color_overlay(camera_cloud["color_image"], camera_records),
                f"Cam{camera_idx} Overlay | n={len(camera_records)}",
                tile_size,
            )
        )
        heatmap_tiles.append(
            label_tile(
                _render_cause_heatmap(camera_cloud["color_image"], camera_records),
                f"Cam{camera_idx} Causes | {_short_cause_counts(camera_records)}",
                tile_size,
            )
        )
    board = compose_grid(overlay_tiles + heatmap_tiles, columns=max(1, len(camera_clouds)))
    return _annotate_source_board(
        board,
        source_name=source_name,
        frame_idx=frame_idx,
        outlier_point_count=int(source_metrics["outlier_point_count"]),
        primary_cause_histogram=source_metrics["primary_cause_histogram"],
    )


def analyze_floating_point_source_frame(
    *,
    source_name: str,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    camera_ids: list[int] | None,
    use_float_ffs_depth_when_available: bool,
    ffs_native_like_postprocess: bool,
    radius_m: float,
    nb_points: int,
    edge_band_px: int,
    dark_threshold: float,
    occlusion_depth_tol_m: float,
    occlusion_depth_tol_ratio: float,
    panel_frame_idx: int,
) -> dict[str, Any]:
    camera_clouds = load_frame_camera_clouds_with_metadata(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=frame_idx,
        depth_source=depth_source,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        ffs_native_like_postprocess=ffs_native_like_postprocess,
        camera_ids=camera_ids,
    )
    fused_cloud = _fuse_camera_clouds(camera_clouds)
    outlier_indices = detect_radius_outlier_indices(
        fused_cloud["points"],
        radius_m=radius_m,
        nb_points=nb_points,
    )["outlier_indices"]
    camera_context = {
        int(cloud["camera_idx"]): {
            **cloud,
            **_build_edge_band_mask(
                cloud["color_image"],
                cloud["depth_m"],
                edge_band_px=edge_band_px,
            ),
        }
        for cloud in camera_clouds
    }
    records: list[dict[str, Any]] = []
    for outlier_idx in outlier_indices:
        point_idx = int(outlier_idx)
        source_camera_idx = int(fused_cloud["source_camera_idx"][point_idx])
        source_camera = camera_context[source_camera_idx]
        u, v = [int(item) for item in fused_cloud["source_pixel_uv"][point_idx]]
        near_edge = bool(source_camera["edge_band"][v, u])
        dark_region = bool(
            _patch_mean(
                source_camera["gray"],
                u=u,
                v=v,
                radius=2,
            ) < float(dark_threshold)
        )
        cross_view = classify_cross_view_relation(
            fused_cloud["points"][point_idx],
            source_camera_idx=source_camera_idx,
            camera_clouds=camera_clouds,
            occlusion_depth_tol_m=occlusion_depth_tol_m,
            occlusion_depth_tol_ratio=occlusion_depth_tol_ratio,
        )
        primary_cause = assign_primary_cause(
            occluded_in_other_views=bool(cross_view["occluded_in_other_views"]),
            near_edge=near_edge,
            dark_region=dark_region,
        )
        records.append(
            {
                "source_camera_idx": int(source_camera_idx),
                "source_serial": str(fused_cloud["source_serial"][point_idx]),
                "source_pixel_uv": [u, v],
                "source_depth_m": float(fused_cloud["source_depth_m"][point_idx]),
                "world_point": fused_cloud["points"][point_idx].astype(np.float32).tolist(),
                "near_edge": bool(near_edge),
                "dark_region": bool(dark_region),
                "occluded_in_other_views": bool(cross_view["occluded_in_other_views"]),
                "cross_view_supported": bool(cross_view["cross_view_supported"]),
                "support_count_other_views": int(cross_view["support_count_other_views"]),
                "primary_cause": str(primary_cause),
            }
        )
    selected_camera_ids = [int(camera_cloud["camera_idx"]) for camera_cloud in camera_clouds]
    metrics = _summarize_outlier_records(
        source_name=source_name,
        frame_idx=frame_idx,
        panel_frame_idx=panel_frame_idx,
        total_point_count=int(len(fused_cloud["points"])),
        records=records,
        selected_camera_ids=selected_camera_ids,
    )
    board = _compose_source_board(
        source_name=source_name,
        frame_idx=frame_idx,
        camera_clouds=camera_clouds,
        records=records,
        tile_size=(320, 220),
        source_metrics=metrics,
    )
    return {
        "frame_idx": int(frame_idx),
        "camera_clouds": camera_clouds,
        "metrics": metrics,
        "records": records,
        "board": board,
    }


def _aggregate_source_metrics(per_frame_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_point_count = int(sum(int(item["total_point_count"]) for item in per_frame_metrics))
    outlier_point_count = int(sum(int(item["outlier_point_count"]) for item in per_frame_metrics))
    primary_cause_histogram = {cause: 0 for cause in PRIMARY_CAUSE_PRIORITY}
    source_camera_histogram: dict[str, int] = {}
    support_histogram = {"0": 0, "1": 0, "2": 0}
    near_edge_count = 0
    dark_region_count = 0
    occlusion_count = 0
    cross_view_supported_count = 0
    for item in per_frame_metrics:
        for cause in PRIMARY_CAUSE_PRIORITY:
            primary_cause_histogram[cause] += int(item["primary_cause_histogram"].get(cause, 0))
        for camera_idx, count in item["source_camera_histogram"].items():
            source_camera_histogram[str(camera_idx)] = source_camera_histogram.get(str(camera_idx), 0) + int(count)
        for support_count, count in item["support_count_other_views_histogram"].items():
            support_histogram[str(support_count)] = support_histogram.get(str(support_count), 0) + int(count)
        near_edge_count += int(item["near_edge_count"])
        dark_region_count += int(item["dark_region_count"])
        occlusion_count += int(item["occluded_in_other_views_count"])
        cross_view_supported_count += int(item["cross_view_supported_count"])
    total_outliers = max(1, outlier_point_count)
    return {
        "frame_count": int(len(per_frame_metrics)),
        "total_point_count": int(total_point_count),
        "outlier_point_count": int(outlier_point_count),
        "outlier_ratio": float(outlier_point_count / max(1, total_point_count)),
        "primary_cause_histogram": primary_cause_histogram,
        "primary_cause_ratio": _ratio_histogram(primary_cause_histogram, total=outlier_point_count),
        "source_camera_histogram": source_camera_histogram,
        "source_camera_ratio": _ratio_histogram(source_camera_histogram, total=outlier_point_count),
        "support_count_other_views_histogram": support_histogram,
        "support_count_other_views_ratio": _ratio_histogram(support_histogram, total=outlier_point_count),
        "near_edge_count": int(near_edge_count),
        "near_edge_ratio": float(near_edge_count / total_outliers),
        "dark_region_count": int(dark_region_count),
        "dark_region_ratio": float(dark_region_count / total_outliers),
        "occluded_in_other_views_count": int(occlusion_count),
        "occluded_in_other_views_ratio": float(occlusion_count / total_outliers),
        "cross_view_supported_count": int(cross_view_supported_count),
        "cross_view_supported_ratio": float(cross_view_supported_count / total_outliers),
    }


def run_floating_point_source_diagnostics_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    camera_ids: list[int] | None = None,
    use_float_ffs_depth_when_available: bool = True,
    ffs_native_like_postprocess: bool = False,
    radius_m: float = 0.01,
    nb_points: int = 40,
    edge_band_px: int = 8,
    dark_threshold: float = 40.0,
    occlusion_depth_tol_m: float = 0.02,
    occlusion_depth_tol_ratio: float = 0.03,
    write_mp4: bool = False,
) -> dict[str, Any]:
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
    selected_camera_ids = resolve_camera_ids(native_metadata, camera_ids)
    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for floating-point diagnostics.")

    output_dir.mkdir(parents=True, exist_ok=True)
    native_frames_dir = output_dir / "native" / "frames"
    ffs_frames_dir = output_dir / "ffs" / "frames"
    native_frames_dir.mkdir(parents=True, exist_ok=True)
    ffs_frames_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    comparison_mp4_path = output_dir / "comparison.mp4"
    fps = int(native_metadata.get("fps", 10))
    native_per_frame_metrics: list[dict[str, Any]] = []
    ffs_per_frame_metrics: list[dict[str, Any]] = []
    for panel_frame_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
        native_result = analyze_floating_point_source_frame(
            source_name="Native",
            case_dir=native_case_dir,
            metadata=native_metadata,
            frame_idx=native_frame_idx,
            depth_source="realsense",
            camera_ids=selected_camera_ids,
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            ffs_native_like_postprocess=False,
            radius_m=radius_m,
            nb_points=nb_points,
            edge_band_px=edge_band_px,
            dark_threshold=dark_threshold,
            occlusion_depth_tol_m=occlusion_depth_tol_m,
            occlusion_depth_tol_ratio=occlusion_depth_tol_ratio,
            panel_frame_idx=panel_frame_idx,
        )
        ffs_result = analyze_floating_point_source_frame(
            source_name="FFS",
            case_dir=ffs_case_dir,
            metadata=ffs_metadata,
            frame_idx=ffs_frame_idx,
            depth_source="ffs",
            camera_ids=selected_camera_ids,
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            ffs_native_like_postprocess=ffs_native_like_postprocess,
            radius_m=radius_m,
            nb_points=nb_points,
            edge_band_px=edge_band_px,
            dark_threshold=dark_threshold,
            occlusion_depth_tol_m=occlusion_depth_tol_m,
            occlusion_depth_tol_ratio=occlusion_depth_tol_ratio,
            panel_frame_idx=panel_frame_idx,
        )
        native_frame_path = native_frames_dir / f"{panel_frame_idx:06d}.png"
        ffs_frame_path = ffs_frames_dir / f"{panel_frame_idx:06d}.png"
        cv2.imwrite(str(native_frame_path), native_result["board"])
        cv2.imwrite(str(ffs_frame_path), ffs_result["board"])
        native_per_frame_metrics.append(native_result["metrics"])
        ffs_per_frame_metrics.append(ffs_result["metrics"])

        if write_mp4:
            combined = np.hstack([native_result["board"], ffs_result["board"]])
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(comparison_mp4_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(max(1, fps)),
                    (combined.shape[1], combined.shape[0]),
                )
            video_writer.write(combined)
    if video_writer is not None:
        video_writer.release()

    write_json(output_dir / "native" / "per_frame_metrics.json", native_per_frame_metrics)
    write_json(output_dir / "ffs" / "per_frame_metrics.json", ffs_per_frame_metrics)
    summary = {
        "same_case_mode": bool(same_case_mode),
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": [[int(native_idx), int(ffs_idx)] for native_idx, ffs_idx in frame_pairs],
        "camera_ids": [int(camera_idx) for camera_idx in selected_camera_ids],
        "parameters": {
            "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
            "ffs_native_like_postprocess": bool(ffs_native_like_postprocess),
            "radius_m": float(radius_m),
            "nb_points": int(nb_points),
            "edge_band_px": int(edge_band_px),
            "dark_threshold": float(dark_threshold),
            "occlusion_depth_tol_m": float(occlusion_depth_tol_m),
            "occlusion_depth_tol_ratio": float(occlusion_depth_tol_ratio),
            "write_mp4": bool(write_mp4),
        },
        "native": {
            "per_frame_metrics_path": str((output_dir / "native" / "per_frame_metrics.json").resolve()),
            "frames_dir": str(native_frames_dir.resolve()),
            "aggregate": _aggregate_source_metrics(native_per_frame_metrics),
        },
        "ffs": {
            "per_frame_metrics_path": str((output_dir / "ffs" / "per_frame_metrics.json").resolve()),
            "frames_dir": str(ffs_frames_dir.resolve()),
            "aggregate": _aggregate_source_metrics(ffs_per_frame_metrics),
        },
        "comparison_mp4": None if not write_mp4 else str(comparison_mp4_path.resolve()),
    }
    write_json(output_dir / "summary.json", summary)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
    }
