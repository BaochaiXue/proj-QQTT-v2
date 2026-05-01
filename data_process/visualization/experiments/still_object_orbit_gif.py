from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from ..calibration_io import load_calibration_transforms
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_frame_camera_clouds, load_case_metadata
from ..layouts import compose_single_row_board
from ..renderers import estimate_ortho_scale, render_point_cloud
from ..views import normalize_vector, project_vector_to_plane, rotate_vector_around_axis
from ..workflows.masked_pointcloud_compare import (
    filter_camera_clouds_with_pixel_masks,
    load_union_masks_for_camera_clouds,
)
from .enhanced_phystwin_postprocess_pcd_compare import (
    DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
)
from .ffs_confidence_filter_pcd_compare import (
    _apply_enhanced_phystwin_like_postprocess,
    _apply_enhanced_phystwin_like_postprocess_with_trace,
)
from .native_ffs_fused_pcd_compare import DEFAULT_PHYSTWIN_NB_POINTS, DEFAULT_PHYSTWIN_RADIUS_M


DEFAULT_STILL_OBJECT_CASE = (
    "data/still_object/ffs203048_iter4_trt_level5/"
    "both_30_still_object_round1_20260428"
)
DEFAULT_OUTPUT_DIR = (
    "result/"
    "still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_TEXT_PROMPT = "stuffed animal"
DEFAULT_6X2_OUTPUT_DIR = (
    "result/"
    "still_object_rope_frame0_cam0_orbit_6x2_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_6X2_ERODE_SWEEP_OUTPUT_DIR = (
    "result/"
    "still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_6X2_ERODE_SWEEP_HIGHLIGHT_OUTPUT_DIR = (
    "result/"
    "still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_3X4_ERODE_SWEEP_HIGHLIGHT_OUTPUT_DIR = (
    "result/"
    "still_object_rope_frame0_cam0_orbit_3x4_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_6X2_ENHANCED_OUTPUT_DIR = (
    "result/"
    "still_object_rope_frame0_cam0_orbit_6x2_enhanced_pt_like_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_3X4_REMOVED_HIGHLIGHT_OUTPUT_DIR = (
    "result/"
    "still_object_rope_frame0_cam0_orbit_3x4_removed_highlight_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_6X2_ERODE_SWEEP_PIXELS: tuple[int, ...] = (1, 3, 5, 10)
SOURCE_CAMERA_HIGHLIGHT_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 0, 255),
    (255, 255, 0),
    (0, 191, 255),
)
SOURCE_CAMERA_HIGHLIGHT_LABELS: tuple[str, ...] = ("Cam0 magenta", "Cam1 cyan", "Cam2 amber")


def default_still_object_rope_6x2_case_specs(*, root: Path) -> list[dict[str, Any]]:
    repo_root = Path(root).resolve()
    return [
        {
            "label": "Still Object R1",
            "case_dir": repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round1_20260428",
            "text_prompt": "stuffed animal",
        },
        {
            "label": "Still Object R2",
            "case_dir": repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round2_20260428",
            "text_prompt": "stuffed animal",
        },
        {
            "label": "Still Object R3",
            "case_dir": repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round3_20260428",
            "text_prompt": "stuffed animal",
        },
        {
            "label": "Still Object R4",
            "case_dir": repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round4_20260428",
            "text_prompt": "stuffed animal",
        },
        {
            "label": "Still Rope R1",
            "case_dir": repo_root / "data/still_rope/ffs203048_iter4_trt_level5/both_30_still_rope_round1_20260428",
            "text_prompt": "white twisted rope on the blue box, white thick twisted rope on top of the blue box",
        },
        {
            "label": "Still Rope R2",
            "case_dir": repo_root / "data/still_rope/ffs203048_iter4_trt_level5/both_30_still_rope_round2_20260428",
            "text_prompt": "white twisted rope lying on the wooden table",
        },
    ]


def _source_camera_highlight_color_bgr(camera_idx: int) -> tuple[int, int, int]:
    palette = SOURCE_CAMERA_HIGHLIGHT_COLORS_BGR
    return tuple(int(item) for item in palette[int(camera_idx) % len(palette)])


def _source_camera_highlight_color_summary(camera_ids: list[int] | tuple[int, ...] = (0, 1, 2)) -> dict[str, list[int]]:
    return {str(int(camera_idx)): list(_source_camera_highlight_color_bgr(int(camera_idx))) for camera_idx in camera_ids}


def _concat_clouds(camera_clouds: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    source_sets: list[np.ndarray] = []
    for item in camera_clouds:
        points = np.asarray(item["points"], dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            continue
        colors = np.asarray(item["colors"], dtype=np.uint8).reshape(-1, 3)
        source_camera_idx = np.asarray(
            item.get("source_camera_idx", np.full((len(points),), int(item["camera_idx"]), dtype=np.int16)),
            dtype=np.int16,
        ).reshape(-1)
        if len(colors) != len(points) or len(source_camera_idx) != len(points):
            raise ValueError(f"Cloud arrays must align for camera {item.get('camera_idx')}.")
        point_sets.append(points)
        color_sets.append(colors)
        source_sets.append(source_camera_idx)
    if not point_sets:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int16),
        )
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0], source_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0), np.concatenate(source_sets, axis=0)


def parse_mask_erode_pixels(erode_pixels: str | int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(erode_pixels, str):
        values = [int(item.strip()) for item in erode_pixels.split(",") if item.strip()]
    elif isinstance(erode_pixels, int):
        values = [int(erode_pixels)]
    else:
        values = [int(item) for item in erode_pixels]
    if not values:
        raise ValueError("At least one mask erode value is required.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Mask erode sweep expects positive pixel values, got {values}.")
    if len(set(values)) != len(values):
        raise ValueError(f"Mask erode pixels must be unique, got {values}.")
    return values


def _erode_masks_by_camera(
    mask_by_camera: dict[int, np.ndarray],
    *,
    erode_pixels: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    iterations = int(erode_pixels)
    if iterations < 0:
        raise ValueError(f"mask_erode_pixels must be >= 0, got {erode_pixels}.")
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_by_camera: dict[int, np.ndarray] = {}
    debug_by_camera: dict[str, Any] = {}
    for camera_idx, mask in mask_by_camera.items():
        mask_bool = np.asarray(mask, dtype=bool)
        original_count = int(np.count_nonzero(mask_bool))
        if iterations == 0:
            eroded = mask_bool.copy()
        else:
            eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=iterations) > 0
        eroded_count = int(np.count_nonzero(eroded))
        eroded_by_camera[int(camera_idx)] = eroded
        debug_by_camera[str(int(camera_idx))] = {
            "camera_idx": int(camera_idx),
            "mask_erode_pixels": iterations,
            "mask_pixel_count_before_erode": original_count,
            "mask_pixel_count_after_erode": eroded_count,
            "eroded_removed_pixel_count": int(original_count - eroded_count),
        }
    return eroded_by_camera, debug_by_camera


def _deterministic_point_cap(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points is None or int(max_points) <= 0 or len(point_array) <= int(max_points):
        return point_array, color_array
    indices = np.linspace(0, len(point_array) - 1, int(max_points), dtype=np.int32)
    return point_array[indices], color_array[indices]


def _robust_bounds(point_sets: list[np.ndarray], *, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    pct = float(percentile)
    for points in point_sets:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(point_array) == 0:
            continue
        if pct > 0.0 and pct < 50.0 and len(point_array) >= 16:
            lows.append(np.percentile(point_array, pct, axis=0).astype(np.float32))
            highs.append(np.percentile(point_array, 100.0 - pct, axis=0).astype(np.float32))
        else:
            lows.append(point_array.min(axis=0).astype(np.float32))
            highs.append(point_array.max(axis=0).astype(np.float32))
    if not lows:
        center = np.zeros((3,), dtype=np.float32)
        return center - 0.5, center + 0.5
    return np.min(np.stack(lows, axis=0), axis=0), np.max(np.stack(highs, axis=0), axis=0)


def _crop_to_expanded_bounds(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(point_array) == 0:
        return point_array, color_array, 0
    lo = np.asarray(bounds_min, dtype=np.float32).reshape(3)
    hi = np.asarray(bounds_max, dtype=np.float32).reshape(3)
    extent = np.maximum(hi - lo, 1e-4)
    margin = extent * max(0.0, float(margin_ratio))
    keep = np.all((point_array >= (lo - margin)) & (point_array <= (hi + margin)), axis=1)
    if int(np.count_nonzero(keep)) < 64:
        return point_array, color_array, 0
    return point_array[keep], color_array[keep], int(len(point_array) - np.count_nonzero(keep))


def _serialize_array(values: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(values, dtype=np.float32).reshape(-1)]


def _load_masked_variant_cloud(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    masks_by_camera: dict[int, np.ndarray],
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_camera: int | None,
    use_float_ffs_depth_when_available: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    camera_clouds, cloud_stats = load_case_frame_camera_clouds(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=int(frame_idx),
        depth_source=str(depth_source),
        use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
        pixel_roi_by_camera=None,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
    )
    masked_clouds, mask_metrics = filter_camera_clouds_with_pixel_masks(
        camera_clouds,
        pixel_mask_by_camera=masks_by_camera,
    )
    points, colors, source_camera_idx = _concat_clouds(masked_clouds)
    stats = {
        "depth_source": str(depth_source),
        "point_count": int(len(points)),
        "cloud_stats": cloud_stats,
        "mask_metrics": mask_metrics,
    }
    return points, colors, source_camera_idx, masked_clouds, stats


def _no_pt_like_postprocess_stats(point_count: int) -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "none",
        "input_point_count": int(point_count),
        "output_point_count": int(point_count),
    }


def _no_pt_like_removed_highlight_stats(point_count: int) -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "none",
        "deletes_points": False,
        "input_point_count": int(point_count),
        "render_point_count": int(point_count),
        "would_remove_point_count": 0,
        "would_remove_point_ratio": 0.0,
        "would_remove_point_count_by_source_camera": {"0": 0, "1": 0, "2": 0},
    }


def _enhanced_pt_like_removed_count_by_source(
    *,
    removed_mask: np.ndarray,
    source_camera_idx: np.ndarray,
    camera_ids: list[int],
) -> dict[str, int]:
    removed = np.asarray(removed_mask, dtype=bool).reshape(-1)
    sources = np.asarray(source_camera_idx, dtype=np.int16).reshape(-1)
    return {
        str(int(camera_idx)): int(np.count_nonzero(removed & (sources == int(camera_idx))))
        for camera_idx in camera_ids
    }


def _highlight_enhanced_pt_like_removed_if_enabled(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    source_camera_idx: np.ndarray,
    enabled: bool,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    source_array = np.asarray(source_camera_idx, dtype=np.int16).reshape(-1)
    if len(color_array) != len(point_array) or len(source_array) != len(point_array):
        raise ValueError("points, colors, and source_camera_idx must have the same length.")
    if not bool(enabled):
        return point_array, color_array, _no_pt_like_removed_highlight_stats(len(point_array))

    _filtered_points, _filtered_colors, trace_stats, trace = _apply_enhanced_phystwin_like_postprocess_with_trace(
        points=point_array,
        colors=color_array,
        enabled=True,
        radius_m=float(phystwin_radius_m),
        nb_points=int(phystwin_nb_points),
        component_voxel_size_m=float(enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )
    removed_mask = np.asarray(trace["removed_mask"], dtype=bool).reshape(-1)
    radius_removed_mask = np.asarray(trace["radius_removed_mask"], dtype=bool).reshape(-1)
    component_removed_mask = np.asarray(trace["component_removed_mask"], dtype=bool).reshape(-1)
    if len(removed_mask) != len(point_array):
        raise ValueError("Enhanced PT-like removed trace length does not match point count.")

    camera_ids = sorted(set(range(len(SOURCE_CAMERA_HIGHLIGHT_COLORS_BGR))) | {int(item) for item in source_array.tolist()})
    highlighted_colors = color_array.copy()
    for camera_idx in camera_ids:
        camera_removed_mask = removed_mask & (source_array == int(camera_idx))
        if np.any(camera_removed_mask):
            highlighted_colors[camera_removed_mask] = np.asarray(
                _source_camera_highlight_color_bgr(int(camera_idx)),
                dtype=np.uint8,
            )

    would_remove_count = int(np.count_nonzero(removed_mask))
    stats = dict(trace_stats)
    stats.update(
        {
            "enabled": True,
            "mode": "enhanced_pt_like_removed_highlight_only",
            "trace_postprocess_mode": str(trace_stats.get("mode", "unknown")),
            "deletes_points": False,
            "input_point_count": int(len(point_array)),
            "render_point_count": int(len(point_array)),
            "output_point_count_if_deleted": int(trace_stats.get("output_point_count", len(point_array))),
            "would_remove_point_count": would_remove_count,
            "would_remove_point_ratio": float(would_remove_count / max(1, len(point_array))),
            "would_remove_point_count_by_source_camera": _enhanced_pt_like_removed_count_by_source(
                removed_mask=removed_mask,
                source_camera_idx=source_array,
                camera_ids=camera_ids,
            ),
            "radius_would_remove_point_count_by_source_camera": _enhanced_pt_like_removed_count_by_source(
                removed_mask=radius_removed_mask,
                source_camera_idx=source_array,
                camera_ids=camera_ids,
            ),
            "component_would_remove_point_count_by_source_camera": _enhanced_pt_like_removed_count_by_source(
                removed_mask=component_removed_mask,
                source_camera_idx=source_array,
                camera_ids=camera_ids,
            ),
            "highlight_color_mode": "source_camera",
            "source_camera_highlight_colors_bgr": _source_camera_highlight_color_summary(tuple(camera_ids)),
            "source_camera_highlight_labels": list(SOURCE_CAMERA_HIGHLIGHT_LABELS),
        }
    )
    return point_array, highlighted_colors, stats


def _apply_enhanced_pt_like_postprocess_if_enabled(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if not bool(enabled):
        return point_array, color_array, _no_pt_like_postprocess_stats(len(point_array))
    return _apply_enhanced_phystwin_like_postprocess(
        points=point_array,
        colors=color_array,
        enabled=True,
        radius_m=float(phystwin_radius_m),
        nb_points=int(phystwin_nb_points),
        component_voxel_size_m=float(enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )


def _build_keyed_orbit_views(
    *,
    c2w_list: list[np.ndarray],
    serial_numbers: list[str],
    focus_point: np.ndarray,
    start_camera_idx: int,
    num_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    focus = np.asarray(focus_point, dtype=np.float32).reshape(3)
    camera_positions = [np.asarray(c2w, dtype=np.float32).reshape(4, 4)[:3, 3] for c2w in c2w_list]
    camera_ups = [
        normalize_vector(-np.asarray(c2w, dtype=np.float32).reshape(4, 4)[:3, 1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        for c2w in c2w_list
    ]
    orbit_axis = normalize_vector(np.mean(np.stack(camera_ups, axis=0), axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    start_idx = int(start_camera_idx)
    start_offset = camera_positions[start_idx] - focus
    basis_x = project_vector_to_plane(start_offset, orbit_axis)
    basis_x = normalize_vector(basis_x, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis_y = normalize_vector(np.cross(orbit_axis, basis_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    key_nodes: list[dict[str, Any]] = []
    for camera_idx, (position, up, serial) in enumerate(zip(camera_positions, camera_ups, serial_numbers, strict=False)):
        offset = position - focus
        planar_x = float(offset @ basis_x)
        planar_y = float(offset @ basis_y)
        rel_angle = float(np.rad2deg(np.arctan2(planar_y, planar_x)) % 360.0)
        if camera_idx == start_idx:
            rel_angle = 0.0
        key_nodes.append(
            {
                "camera_idx": int(camera_idx),
                "serial": str(serial),
                "relative_angle_deg": rel_angle,
                "radius_m": float(np.hypot(planar_x, planar_y)),
                "height_m": float(offset @ orbit_axis),
                "position": _serialize_array(position),
                "up": _serialize_array(up),
            }
        )

    key_nodes_sorted = sorted(key_nodes, key=lambda item: float(item["relative_angle_deg"]))
    if key_nodes_sorted[0]["camera_idx"] != start_idx:
        key_nodes_sorted = [item for item in key_nodes_sorted if item["camera_idx"] == start_idx] + [
            item for item in key_nodes_sorted if item["camera_idx"] != start_idx
        ]
    interp_nodes = key_nodes_sorted + [dict(key_nodes_sorted[0], relative_angle_deg=360.0)]
    node_angles = np.asarray([float(item["relative_angle_deg"]) for item in interp_nodes], dtype=np.float32)
    node_radii = np.asarray([float(item["radius_m"]) for item in interp_nodes], dtype=np.float32)
    node_heights = np.asarray([float(item["height_m"]) for item in interp_nodes], dtype=np.float32)

    views: list[dict[str, Any]] = []
    total = max(1, int(num_frames))
    for frame_idx, angle_deg in enumerate(np.linspace(0.0, 360.0, total, endpoint=False)):
        radius = float(np.interp(angle_deg, node_angles, node_radii))
        height = float(np.interp(angle_deg, node_angles, node_heights))
        theta = np.deg2rad(float(angle_deg))
        planar = basis_x * (np.cos(theta) * radius) + basis_y * (np.sin(theta) * radius)
        camera_position = (focus + planar + orbit_axis * height).astype(np.float32)
        up = rotate_vector_around_axis(camera_ups[start_idx], orbit_axis, float(angle_deg))
        views.append(
            {
                "view_name": f"cam{start_idx}_orbit_{frame_idx:03d}",
                "label": f"Cam{start_idx} orbit {angle_deg:06.2f} deg",
                "camera_idx": int(start_idx),
                "serial": str(serial_numbers[start_idx]),
                "center": focus.copy(),
                "camera_position": camera_position,
                "up": normalize_vector(up, orbit_axis),
                "orbit_angle_deg": float(angle_deg),
                "radius": float(np.linalg.norm(camera_position - focus)),
            }
        )

    summary = {
        "focus_point": _serialize_array(focus),
        "orbit_axis": _serialize_array(orbit_axis),
        "orbit_basis_x": _serialize_array(basis_x),
        "orbit_basis_y": _serialize_array(basis_y),
        "start_camera_idx": int(start_idx),
        "key_nodes": key_nodes_sorted,
    }
    return views, summary


def _estimate_global_ortho_scale(
    *,
    point_sets: list[np.ndarray],
    views: list[dict[str, Any]],
    margin: float,
) -> float:
    if not views:
        return 1.0
    sample_count = min(36, len(views))
    sample_indices = np.linspace(0, len(views) - 1, sample_count, dtype=np.int32)
    scales = [
        estimate_ortho_scale(point_sets, view_config=views[int(idx)], margin=float(margin))
        for idx in sample_indices
    ]
    return float(max(scales) if scales else 1.0)


def _label_tile_fit(image: np.ndarray, label: str, tile_size: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(image_array, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 22), (0, 0, 0), -1)
    _draw_text_fit(
        tile,
        text=str(label),
        origin=(8, 16),
        max_width=max(16, tile_w - 16),
        font_scale=0.5,
        color=(255, 255, 255),
        thickness=1,
    )
    return tile


def _render_variant_tile(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    view_config: dict[str, Any],
    render_mode: str,
    scalar_bounds: dict[str, tuple[float, float]],
    tile_width: int,
    tile_height: int,
    point_radius_px: int,
    supersample_scale: int,
    projection_mode: str,
    ortho_scale: float | None,
    label: str,
) -> np.ndarray:
    rendered, _renderer_used = render_point_cloud(
        points,
        colors,
        renderer="fallback",
        view_config=view_config,
        render_mode=str(render_mode),
        scalar_bounds=scalar_bounds,
        width=int(tile_width),
        height=int(tile_height),
        point_radius_px=int(point_radius_px),
        supersample_scale=int(supersample_scale),
        projection_mode=str(projection_mode),
        ortho_scale=ortho_scale,
    )
    return _label_tile_fit(rendered, label, (int(tile_width), int(tile_height)))


def _draw_text_fit(
    image: np.ndarray,
    *,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    scale = float(font_scale)
    while scale > 0.36:
        width = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, int(thickness))[0][0]
        if width <= int(max_width):
            break
        scale -= 0.04
    cv2.putText(
        image,
        str(text),
        (int(origin[0]), int(origin[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _compose_6x2_orbit_board(
    *,
    title_lines: list[str],
    row_headers: list[str],
    column_headers: list[str],
    image_rows: list[list[np.ndarray]],
    row_label_width: int,
) -> np.ndarray:
    if len(column_headers) != 2:
        raise ValueError("6x2 orbit board requires exactly 2 column headers.")
    if len(row_headers) != len(image_rows):
        raise ValueError("row_headers must match image_rows.")
    if any(len(row) != 2 for row in image_rows):
        raise ValueError("Each image row must contain exactly 2 tiles.")
    if not image_rows:
        raise ValueError("No image rows provided.")

    tile_h, tile_w = image_rows[0][0].shape[:2]
    if any(tile.shape[:2] != (tile_h, tile_w) for row in image_rows for tile in row):
        raise ValueError("All tiles must have the same size.")

    title_h = 78
    header_h = 40
    body_h = tile_h * len(image_rows)
    body_w = int(row_label_width) + tile_w * 2
    title = np.full((title_h, body_w, 3), (10, 10, 10), dtype=np.uint8)
    for line_idx, line in enumerate(title_lines[:2]):
        _draw_text_fit(
            title,
            text=line,
            origin=(16, 30 + line_idx * 26),
            max_width=body_w - 32,
            font_scale=0.84 if line_idx == 0 else 0.58,
            color=(255, 255, 255) if line_idx == 0 else (220, 220, 220),
            thickness=2 if line_idx == 0 else 1,
        )

    header = np.full((header_h, body_w, 3), (18, 18, 18), dtype=np.uint8)
    header[:, : int(row_label_width)] = (14, 14, 14)
    for col_idx, column_header in enumerate(column_headers):
        x0 = int(row_label_width) + col_idx * tile_w
        _draw_text_fit(
            header,
            text=column_header,
            origin=(x0 + 12, 28),
            max_width=tile_w - 24,
            font_scale=0.78,
            color=(255, 255, 255),
            thickness=2,
        )

    body = np.full((body_h, body_w, 3), (24, 24, 24), dtype=np.uint8)
    for row_idx, (row_header, row_tiles) in enumerate(zip(row_headers, image_rows, strict=True)):
        y0 = row_idx * tile_h
        body[y0 : y0 + tile_h, : int(row_label_width)] = (12, 12, 12)
        _draw_text_fit(
            body,
            text=row_header,
            origin=(14, y0 + max(26, tile_h // 2)),
            max_width=int(row_label_width) - 24,
            font_scale=0.66,
            color=(255, 255, 255),
            thickness=2,
        )
        for col_idx, tile in enumerate(row_tiles):
            x0 = int(row_label_width) + col_idx * tile_w
            body[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return np.vstack([title, header, body])


def _compose_3x4_orbit_board(
    *,
    title_lines: list[str],
    column_headers: list[str],
    image_rows: list[list[np.ndarray]],
) -> np.ndarray:
    if len(column_headers) != 4:
        raise ValueError("3x4 orbit board requires exactly 4 column headers.")
    if any(len(row) != 4 for row in image_rows):
        raise ValueError("Each 3x4 image row must contain exactly 4 tiles.")
    if len(image_rows) != 3:
        raise ValueError("3x4 orbit board requires exactly 3 image rows.")

    tile_h, tile_w = image_rows[0][0].shape[:2]
    if any(tile.shape[:2] != (tile_h, tile_w) for row in image_rows for tile in row):
        raise ValueError("All tiles must have the same size.")

    title_h = 106
    header_h = 36
    body_h = tile_h * 3
    body_w = tile_w * 4
    title = np.full((title_h, body_w, 3), (10, 10, 10), dtype=np.uint8)
    title_origins = (30, 58, 84)
    for line_idx, line in enumerate(title_lines[:3]):
        _draw_text_fit(
            title,
            text=line,
            origin=(16, title_origins[line_idx]),
            max_width=body_w - 32,
            font_scale=0.84 if line_idx == 0 else 0.56,
            color=(255, 255, 255) if line_idx == 0 else (220, 220, 220),
            thickness=2 if line_idx == 0 else 1,
        )

    header = np.full((header_h, body_w, 3), (18, 18, 18), dtype=np.uint8)
    for col_idx, column_header in enumerate(column_headers):
        x0 = col_idx * tile_w
        _draw_text_fit(
            header,
            text=column_header,
            origin=(x0 + 12, 25),
            max_width=tile_w - 24,
            font_scale=0.68,
            color=(255, 255, 255),
            thickness=2,
        )

    body = np.full((body_h, body_w, 3), (24, 24, 24), dtype=np.uint8)
    for row_idx, row_tiles in enumerate(image_rows):
        y0 = row_idx * tile_h
        for col_idx, tile in enumerate(row_tiles):
            x0 = col_idx * tile_w
            body[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return np.vstack([title, header, body])


def _short_case_label(label: str) -> str:
    text = str(label)
    text = text.replace("Still Object", "Object")
    text = text.replace("Still Rope", "Rope")
    return text


def _format_point_count(point_count: int) -> str:
    count = int(point_count)
    if count >= 1000:
        return f"{count / 1000.0:.1f}k"
    return str(count)


def _format_removed_highlight_label(stats: dict[str, Any]) -> str:
    if not bool(stats.get("enabled", False)):
        return ""
    removed_count = int(stats.get("would_remove_point_count", 0))
    by_camera = stats.get("would_remove_point_count_by_source_camera", {})
    c0 = int(by_camera.get("0", 0))
    c1 = int(by_camera.get("1", 0))
    c2 = int(by_camera.get("2", 0))
    return f"rm={removed_count} C={c0}/{c1}/{c2}"


def _orbit_tile_label(
    *,
    case_label: str,
    point_count: int,
    angle_deg: float,
    removed_highlight_stats: dict[str, Any],
) -> str:
    highlight_label = _format_removed_highlight_label(removed_highlight_stats)
    if highlight_label:
        return f"{_short_case_label(case_label)} | {_format_point_count(point_count)} | {highlight_label} | {angle_deg:05.1f}deg"
    return f"{_short_case_label(case_label)} | {_format_point_count(point_count)} pts | {angle_deg:05.1f}deg"


def _prepare_orbit_case_payload(
    *,
    case_dir: Path,
    label: str,
    text_prompt: str,
    frame_idx: int,
    mask_root: Path | None,
    start_camera_idx: int,
    num_frames: int,
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_camera: int | None,
    max_points_per_variant: int | None,
    robust_bounds_percentile: float,
    crop_to_robust_bounds: bool,
    crop_margin_ratio: float,
    projection_mode: str,
    ortho_margin: float,
    mask_erode_pixels: int = 0,
    enhanced_pt_like_postprocess: bool = False,
    highlight_enhanced_pt_like_removed: bool = False,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
) -> dict[str, Any]:
    if bool(enhanced_pt_like_postprocess) and bool(highlight_enhanced_pt_like_removed):
        raise ValueError("Use either enhanced_pt_like_postprocess deletion or highlight_enhanced_pt_like_removed, not both.")
    resolved_case_dir = Path(case_dir).resolve()
    selected_mask_root = (resolved_case_dir / "sam31_masks") if mask_root is None else Path(mask_root)
    selected_mask_root = selected_mask_root.resolve()
    metadata = load_case_metadata(resolved_case_dir)
    selected_frame = int(frame_idx)
    if selected_frame < 0 or selected_frame >= int(metadata["frame_num"]):
        raise ValueError(f"{label}: frame_idx={selected_frame} is outside frame_num={metadata['frame_num']}.")
    serial_numbers = [str(item) for item in metadata["serial_numbers"]]
    if int(start_camera_idx) < 0 or int(start_camera_idx) >= len(serial_numbers):
        raise ValueError(f"{label}: start_camera_idx={start_camera_idx} is outside camera count {len(serial_numbers)}.")

    minimal_clouds = [
        {
            "camera_idx": int(camera_idx),
            "serial": serial,
            "color_path": str(resolved_case_dir / "color" / str(camera_idx) / f"{selected_frame}.png"),
        }
        for camera_idx, serial in enumerate(serial_numbers)
    ]
    masks_by_camera, mask_debug = load_union_masks_for_camera_clouds(
        mask_root=selected_mask_root,
        camera_clouds=minimal_clouds,
        frame_token=str(selected_frame),
        text_prompt=str(text_prompt),
    )
    masks_by_camera, erode_debug = _erode_masks_by_camera(
        masks_by_camera,
        erode_pixels=int(mask_erode_pixels),
    )
    native_points, native_colors, native_source_camera_idx, _native_clouds, native_stats = _load_masked_variant_cloud(
        case_dir=resolved_case_dir,
        metadata=metadata,
        frame_idx=selected_frame,
        depth_source="realsense",
        masks_by_camera=masks_by_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=False,
    )
    ffs_points, ffs_colors, ffs_source_camera_idx, _ffs_clouds, ffs_stats = _load_masked_variant_cloud(
        case_dir=resolved_case_dir,
        metadata=metadata,
        frame_idx=selected_frame,
        depth_source="ffs_raw",
        masks_by_camera=masks_by_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=True,
    )
    native_pt_like_stats = _no_pt_like_postprocess_stats(len(native_points))
    ffs_pt_like_stats = _no_pt_like_postprocess_stats(len(ffs_points))
    native_pt_like_removed_highlight_stats = _no_pt_like_removed_highlight_stats(len(native_points))
    ffs_pt_like_removed_highlight_stats = _no_pt_like_removed_highlight_stats(len(ffs_points))
    if bool(enhanced_pt_like_postprocess):
        native_points, native_colors, native_pt_like_stats = _apply_enhanced_pt_like_postprocess_if_enabled(
            points=native_points,
            colors=native_colors,
            enabled=True,
            phystwin_radius_m=float(phystwin_radius_m),
            phystwin_nb_points=int(phystwin_nb_points),
            enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        ffs_points, ffs_colors, ffs_pt_like_stats = _apply_enhanced_pt_like_postprocess_if_enabled(
            points=ffs_points,
            colors=ffs_colors,
            enabled=True,
            phystwin_radius_m=float(phystwin_radius_m),
            phystwin_nb_points=int(phystwin_nb_points),
            enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
    elif bool(highlight_enhanced_pt_like_removed):
        native_points, native_colors, native_pt_like_removed_highlight_stats = _highlight_enhanced_pt_like_removed_if_enabled(
            points=native_points,
            colors=native_colors,
            source_camera_idx=native_source_camera_idx,
            enabled=True,
            phystwin_radius_m=float(phystwin_radius_m),
            phystwin_nb_points=int(phystwin_nb_points),
            enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        ffs_points, ffs_colors, ffs_pt_like_removed_highlight_stats = _highlight_enhanced_pt_like_removed_if_enabled(
            points=ffs_points,
            colors=ffs_colors,
            source_camera_idx=ffs_source_camera_idx,
            enabled=True,
            phystwin_radius_m=float(phystwin_radius_m),
            phystwin_nb_points=int(phystwin_nb_points),
            enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
    initial_bounds_min, initial_bounds_max = _robust_bounds(
        [native_points, ffs_points],
        percentile=float(robust_bounds_percentile),
    )
    native_cropped_point_count = 0
    ffs_cropped_point_count = 0
    if bool(crop_to_robust_bounds):
        native_points, native_colors, native_cropped_point_count = _crop_to_expanded_bounds(
            native_points,
            native_colors,
            bounds_min=initial_bounds_min,
            bounds_max=initial_bounds_max,
            margin_ratio=float(crop_margin_ratio),
        )
        ffs_points, ffs_colors, ffs_cropped_point_count = _crop_to_expanded_bounds(
            ffs_points,
            ffs_colors,
            bounds_min=initial_bounds_min,
            bounds_max=initial_bounds_max,
            margin_ratio=float(crop_margin_ratio),
        )

    native_points, native_colors = _deterministic_point_cap(
        native_points,
        native_colors,
        max_points=max_points_per_variant,
    )
    ffs_points, ffs_colors = _deterministic_point_cap(
        ffs_points,
        ffs_colors,
        max_points=max_points_per_variant,
    )
    native_pt_like_removed_highlight_stats["final_render_point_count"] = int(len(native_points))
    ffs_pt_like_removed_highlight_stats["final_render_point_count"] = int(len(ffs_points))

    c2w_list = load_calibration_transforms(
        resolved_case_dir / "calibrate.pkl",
        serial_numbers=serial_numbers,
        calibration_reference_serials=metadata.get("calibration_reference_serials", serial_numbers),
    )
    bounds_min, bounds_max = _robust_bounds(
        [native_points, ffs_points],
        percentile=float(robust_bounds_percentile),
    )
    focus_point = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    views, orbit_summary = _build_keyed_orbit_views(
        c2w_list=c2w_list,
        serial_numbers=serial_numbers,
        focus_point=focus_point,
        start_camera_idx=int(start_camera_idx),
        num_frames=int(num_frames),
    )

    selected_projection_mode = str(projection_mode)
    ortho_scale = None
    if selected_projection_mode == "orthographic":
        ortho_scale = _estimate_global_ortho_scale(
            point_sets=[native_points, ffs_points],
            views=views,
            margin=float(ortho_margin),
        )
    elif selected_projection_mode != "perspective":
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")

    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(0.25, float(np.linalg.norm(bounds_max - bounds_min) * 2.5))),
    }
    return {
        "label": str(label),
        "case_dir": str(resolved_case_dir),
        "mask_root": str(selected_mask_root),
        "text_prompt": str(text_prompt),
        "metadata": metadata,
        "serial_numbers": serial_numbers,
        "native_points": native_points,
        "native_colors": native_colors,
        "ffs_points": ffs_points,
        "ffs_colors": ffs_colors,
        "views": views,
        "scalar_bounds": scalar_bounds,
        "ortho_scale": ortho_scale,
        "initial_bounds_min": initial_bounds_min,
        "initial_bounds_max": initial_bounds_max,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "mask_debug": mask_debug,
        "mask_erode_pixels": int(mask_erode_pixels),
        "mask_erode_debug": erode_debug,
        "native_stats": native_stats,
        "ffs_stats": ffs_stats,
        "native_pt_like_stats": native_pt_like_stats,
        "ffs_pt_like_stats": ffs_pt_like_stats,
        "native_pt_like_removed_highlight_stats": native_pt_like_removed_highlight_stats,
        "ffs_pt_like_removed_highlight_stats": ffs_pt_like_removed_highlight_stats,
        "native_cropped_point_count": int(native_cropped_point_count),
        "ffs_cropped_point_count": int(ffs_cropped_point_count),
        "orbit": orbit_summary,
    }


def run_still_object_orbit_gif_workflow(
    *,
    case_dir: Path,
    output_dir: Path,
    frame_idx: int = 0,
    mask_root: Path | None = None,
    text_prompt: str = DEFAULT_TEXT_PROMPT,
    start_camera_idx: int = 0,
    num_frames: int = 360,
    fps: int = 30,
    tile_width: int = 480,
    tile_height: int = 360,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_variant: int | None = 160_000,
    robust_bounds_percentile: float = 1.0,
    crop_to_robust_bounds: bool = True,
    crop_margin_ratio: float = 0.25,
    render_mode: str = "color_by_rgb",
    projection_mode: str = "orthographic",
    ortho_margin: float = 1.28,
    point_radius_px: int = 1,
    supersample_scale: int = 1,
    highlight_enhanced_pt_like_removed: bool = True,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_mask_root = (case_dir / "sam31_masks") if mask_root is None else Path(mask_root)
    selected_mask_root = selected_mask_root.resolve()

    metadata = load_case_metadata(case_dir)
    frame_count = int(metadata["frame_num"])
    selected_frame = int(frame_idx)
    if selected_frame < 0 or selected_frame >= frame_count:
        raise ValueError(f"frame_idx={selected_frame} is outside frame_num={frame_count}.")
    serial_numbers = [str(item) for item in metadata["serial_numbers"]]
    if int(start_camera_idx) < 0 or int(start_camera_idx) >= len(serial_numbers):
        raise ValueError(f"start_camera_idx={start_camera_idx} is outside camera count {len(serial_numbers)}.")

    minimal_clouds = [
        {
            "camera_idx": int(camera_idx),
            "serial": serial,
            "color_path": str(case_dir / "color" / str(camera_idx) / f"{selected_frame}.png"),
        }
        for camera_idx, serial in enumerate(serial_numbers)
    ]
    masks_by_camera, mask_debug = load_union_masks_for_camera_clouds(
        mask_root=selected_mask_root,
        camera_clouds=minimal_clouds,
        frame_token=str(selected_frame),
        text_prompt=str(text_prompt),
    )

    native_points, native_colors, _native_source_camera_idx, _native_clouds, native_stats = _load_masked_variant_cloud(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=selected_frame,
        depth_source="realsense",
        masks_by_camera=masks_by_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=False,
    )
    ffs_points, ffs_colors, _ffs_source_camera_idx, _ffs_clouds, ffs_stats = _load_masked_variant_cloud(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=selected_frame,
        depth_source="ffs_raw",
        masks_by_camera=masks_by_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=True,
    )
    initial_bounds_min, initial_bounds_max = _robust_bounds(
        [native_points, ffs_points],
        percentile=float(robust_bounds_percentile),
    )
    native_cropped_point_count = 0
    ffs_cropped_point_count = 0
    if bool(crop_to_robust_bounds):
        native_points, native_colors, native_cropped_point_count = _crop_to_expanded_bounds(
            native_points,
            native_colors,
            bounds_min=initial_bounds_min,
            bounds_max=initial_bounds_max,
            margin_ratio=float(crop_margin_ratio),
        )
        ffs_points, ffs_colors, ffs_cropped_point_count = _crop_to_expanded_bounds(
            ffs_points,
            ffs_colors,
            bounds_min=initial_bounds_min,
            bounds_max=initial_bounds_max,
            margin_ratio=float(crop_margin_ratio),
        )

    native_points, native_colors = _deterministic_point_cap(
        native_points,
        native_colors,
        max_points=max_points_per_variant,
    )
    ffs_points, ffs_colors = _deterministic_point_cap(
        ffs_points,
        ffs_colors,
        max_points=max_points_per_variant,
    )

    c2w_list = load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=serial_numbers,
        calibration_reference_serials=metadata.get("calibration_reference_serials", serial_numbers),
    )
    bounds_min, bounds_max = _robust_bounds(
        [native_points, ffs_points],
        percentile=float(robust_bounds_percentile),
    )
    focus_point = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    views, orbit_summary = _build_keyed_orbit_views(
        c2w_list=c2w_list,
        serial_numbers=serial_numbers,
        focus_point=focus_point,
        start_camera_idx=int(start_camera_idx),
        num_frames=int(num_frames),
    )

    selected_projection_mode = str(projection_mode)
    ortho_scale = None
    if selected_projection_mode == "orthographic":
        ortho_scale = _estimate_global_ortho_scale(
            point_sets=[native_points, ffs_points],
            views=views,
            margin=float(ortho_margin),
        )
    elif selected_projection_mode != "perspective":
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")

    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(0.25, float(np.linalg.norm(bounds_max - bounds_min) * 2.5))),
    }
    ffs_config = dict(metadata.get("ffs_config", {}))
    ffs_model_name = str(ffs_config.get("model_name", "20-30-48"))
    ffs_valid_iters = int(ffs_config.get("valid_iters", 4))
    ffs_builder_level = int(ffs_config.get("builder_optimization_level", 5))
    ffs_label = f"FFS {ffs_model_name} iter{ffs_valid_iters} L{ffs_builder_level}"
    title_lines = [
        f"Still object round 1 | frame {selected_frame} | Cam{int(start_camera_idx)} keyed orbit",
        (
            f"FFS={ffs_model_name} iter{ffs_valid_iters} pad864 TRT L{ffs_builder_level} | "
            "key poses Cam0/Cam1/Cam2 | 360deg look-at object"
        ),
    ]

    gif_path = output_dir / f"still_object_round1_frame{selected_frame:04d}_cam{int(start_camera_idx)}_orbit_1x2.gif"
    first_frame_path = output_dir / f"still_object_round1_frame{selected_frame:04d}_cam{int(start_camera_idx)}_orbit_first.png"
    summary_path = output_dir / "summary.json"

    with imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(fps)), loop=0) as writer:
        for idx, view_config in enumerate(views):
            angle = float(view_config["orbit_angle_deg"])
            native_tile = _render_variant_tile(
                points=native_points,
                colors=native_colors,
                view_config=view_config,
                render_mode=str(render_mode),
                scalar_bounds=scalar_bounds,
                tile_width=int(tile_width),
                tile_height=int(tile_height),
                point_radius_px=int(point_radius_px),
                supersample_scale=int(supersample_scale),
                projection_mode=selected_projection_mode,
                ortho_scale=ortho_scale,
                label=f"Native Depth | {len(native_points)} pts | {angle:05.1f} deg",
            )
            ffs_tile = _render_variant_tile(
                points=ffs_points,
                colors=ffs_colors,
                view_config=view_config,
                render_mode=str(render_mode),
                scalar_bounds=scalar_bounds,
                tile_width=int(tile_width),
                tile_height=int(tile_height),
                point_radius_px=int(point_radius_px),
                supersample_scale=int(supersample_scale),
                projection_mode=selected_projection_mode,
                ortho_scale=ortho_scale,
                label=f"FFS pad864 L{ffs_builder_level} | {len(ffs_points)} pts | {angle:05.1f} deg",
            )
            board = compose_single_row_board(
                title_lines=title_lines,
                column_headers=["Native Depth", ffs_label],
                images=[native_tile, ffs_tile],
            )
            if idx == 0:
                write_image(first_frame_path, board)
            writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            if idx == 0 or (idx + 1) % 30 == 0 or idx + 1 == len(views):
                print(f"rendered {idx + 1}/{len(views)} frames", flush=True)

    summary = {
        "case_dir": str(case_dir),
        "mask_root": str(selected_mask_root),
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "frame_idx": selected_frame,
        "text_prompt": str(text_prompt),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "crop_to_robust_bounds": bool(crop_to_robust_bounds),
        "crop_margin_ratio": float(crop_margin_ratio),
        "initial_bounds_min": _serialize_array(initial_bounds_min),
        "initial_bounds_max": _serialize_array(initial_bounds_max),
        "projection_mode": selected_projection_mode,
        "ortho_scale": None if ortho_scale is None else float(ortho_scale),
        "render_mode": str(render_mode),
        "point_radius_px": int(point_radius_px),
        "serial_numbers": serial_numbers,
        "ffs_config": ffs_config,
        "mask_debug": mask_debug,
        "native_stats": native_stats,
        "ffs_stats": ffs_stats,
        "native_cropped_point_count": int(native_cropped_point_count),
        "ffs_cropped_point_count": int(ffs_cropped_point_count),
        "native_render_point_count": int(len(native_points)),
        "ffs_render_point_count": int(len(ffs_points)),
        "bounds_min": _serialize_array(bounds_min),
        "bounds_max": _serialize_array(bounds_max),
        "orbit": orbit_summary,
    }
    write_json(summary_path, summary)
    return summary


def run_still_object_rope_6x2_orbit_gif_workflow(
    *,
    case_specs: list[dict[str, Any]],
    output_dir: Path,
    frame_idx: int = 0,
    start_camera_idx: int = 0,
    num_frames: int = 360,
    fps: int = 30,
    tile_width: int = 360,
    tile_height: int = 220,
    row_label_width: int = 180,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_variant: int | None = 120_000,
    robust_bounds_percentile: float = 1.0,
    crop_to_robust_bounds: bool = True,
    crop_margin_ratio: float = 0.25,
    render_mode: str = "color_by_rgb",
    projection_mode: str = "orthographic",
    ortho_margin: float = 1.28,
    point_radius_px: int = 1,
    supersample_scale: int = 1,
    layout: str = "6x2",
    mask_erode_pixels: int = 0,
    enhanced_pt_like_postprocess: bool = False,
    highlight_enhanced_pt_like_removed: bool = False,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
) -> dict[str, Any]:
    if bool(enhanced_pt_like_postprocess) and bool(highlight_enhanced_pt_like_removed):
        raise ValueError("Use either enhanced_pt_like_postprocess deletion or highlight_enhanced_pt_like_removed, not both.")
    if len(case_specs) != 6:
        raise ValueError(f"6x2 orbit GIF requires exactly 6 case specs, got {len(case_specs)}.")
    selected_layout = str(layout).lower()
    if selected_layout not in {"6x2", "3x4"}:
        raise ValueError(f"layout must be '6x2' or '3x4', got {layout!r}.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    for spec in case_specs:
        payload = _prepare_orbit_case_payload(
            case_dir=Path(spec["case_dir"]),
            label=str(spec["label"]),
            text_prompt=str(spec["text_prompt"]),
            frame_idx=int(frame_idx),
            mask_root=Path(spec["mask_root"]) if spec.get("mask_root") is not None else None,
            start_camera_idx=int(start_camera_idx),
            num_frames=int(num_frames),
            depth_min_m=float(depth_min_m),
            depth_max_m=float(depth_max_m),
            max_points_per_camera=max_points_per_camera,
            max_points_per_variant=max_points_per_variant,
            robust_bounds_percentile=float(robust_bounds_percentile),
            crop_to_robust_bounds=bool(crop_to_robust_bounds),
            crop_margin_ratio=float(crop_margin_ratio),
            projection_mode=str(projection_mode),
            ortho_margin=float(ortho_margin),
            mask_erode_pixels=int(mask_erode_pixels),
            enhanced_pt_like_postprocess=bool(enhanced_pt_like_postprocess),
            highlight_enhanced_pt_like_removed=bool(highlight_enhanced_pt_like_removed),
            phystwin_radius_m=float(phystwin_radius_m),
            phystwin_nb_points=int(phystwin_nb_points),
            enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        payloads.append(payload)

    first_ffs_config = dict(payloads[0]["metadata"].get("ffs_config", {}))
    ffs_model_name = str(first_ffs_config.get("model_name", "20-30-48"))
    ffs_valid_iters = int(first_ffs_config.get("valid_iters", 4))
    ffs_builder_level = int(first_ffs_config.get("builder_optimization_level", 5))
    enhanced_setting_label = (
        f"enhPT trace r={float(phystwin_radius_m):.3f}/{int(phystwin_nb_points)} "
        f"comp={float(enhanced_component_voxel_size_m):.3f} gap={float(enhanced_keep_near_main_gap_m):.3f}"
    )
    if bool(enhanced_pt_like_postprocess):
        postprocess_label = enhanced_setting_label.replace("trace", "delete", 1)
        tile_mode_label = "enhPT delete"
    elif bool(highlight_enhanced_pt_like_removed):
        postprocess_label = f"{enhanced_setting_label} | removed points highlighted only"
        tile_mode_label = f"erode {int(mask_erode_pixels)}px + mark"
    else:
        postprocess_label = "no PT-like postprocess"
        tile_mode_label = f"erode {int(mask_erode_pixels)}px"
    erode_title = f" | erode={int(mask_erode_pixels)}px" if selected_layout == "3x4" or int(mask_erode_pixels) != 0 else ""
    if selected_layout == "3x4":
        title_lines = [
            (
                f"Still Object R1-R4 + Rope R1-R2 | frame {int(frame_idx)} | "
                f"Cam{int(start_camera_idx)} keyed orbit 3x4{erode_title}"
            ),
            "raw RGB point colors | enhanced PT-like discovery marks would-delete points; no points are deleted",
            (
                f"FFS={ffs_model_name} iter{ffs_valid_iters} pad864 TRT L{ffs_builder_level} | "
                "colors: Cam0 magenta / Cam1 cyan / Cam2 amber"
            ),
        ]
        column_headers = [
            "Native Depth",
            f"FFS {ffs_model_name} iter{ffs_valid_iters} L{ffs_builder_level}",
            "Native Depth",
            f"FFS {ffs_model_name} iter{ffs_valid_iters} L{ffs_builder_level}",
        ]
    else:
        title_lines = [
            (
                f"Still Object R1-R4 + Rope R1-R2 | frame {int(frame_idx)} | "
                f"Cam{int(start_camera_idx)} 6x2 orbit{erode_title}"
            ),
            (
                f"raw RGB point colors | {postprocess_label} | FFS={ffs_model_name} "
                f"iter{ffs_valid_iters} pad864 TRT L{ffs_builder_level}"
            ),
        ]
        column_headers = ["Native Depth", f"FFS {ffs_model_name} iter{ffs_valid_iters} L{ffs_builder_level}"]
    row_headers = [str(payload["label"]) for payload in payloads]
    erode_suffix = "" if int(mask_erode_pixels) == 0 else f"_mask_erode_{int(mask_erode_pixels):02d}px"
    post_suffix = "_enhanced_pt_like" if bool(enhanced_pt_like_postprocess) else ""
    if bool(highlight_enhanced_pt_like_removed):
        post_suffix = "_enhanced_pt_like_marked"
    gif_path = output_dir / (
        f"still_object_rope_frame{int(frame_idx):04d}_cam{int(start_camera_idx)}_orbit_"
        f"{selected_layout}{post_suffix}{erode_suffix}.gif"
    )
    first_frame_path = output_dir / (
        f"still_object_rope_frame{int(frame_idx):04d}_cam{int(start_camera_idx)}_orbit_"
        f"{selected_layout}{post_suffix}{erode_suffix}_first.png"
    )
    summary_path = output_dir / "summary.json"

    selected_projection_mode = str(projection_mode)
    total_frames = max(1, int(num_frames))
    with imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(fps)), loop=0) as writer:
        for frame_number in range(total_frames):
            if selected_layout == "3x4":
                image_rows = []
                for left_idx in (0, 2, 4):
                    row_tiles = []
                    for payload in (payloads[left_idx], payloads[left_idx + 1]):
                        view_config = payload["views"][frame_number]
                        angle = float(view_config["orbit_angle_deg"])
                        native_tile = _render_variant_tile(
                            points=payload["native_points"],
                            colors=payload["native_colors"],
                            view_config=view_config,
                            render_mode=str(render_mode),
                            scalar_bounds=payload["scalar_bounds"],
                            tile_width=int(tile_width),
                            tile_height=int(tile_height),
                            point_radius_px=int(point_radius_px),
                            supersample_scale=int(supersample_scale),
                            projection_mode=selected_projection_mode,
                            ortho_scale=payload["ortho_scale"],
                            label=_orbit_tile_label(
                                case_label=str(payload["label"]),
                                point_count=len(payload["native_points"]),
                                angle_deg=angle,
                                removed_highlight_stats=payload["native_pt_like_removed_highlight_stats"],
                            ),
                        )
                        ffs_tile = _render_variant_tile(
                            points=payload["ffs_points"],
                            colors=payload["ffs_colors"],
                            view_config=view_config,
                            render_mode=str(render_mode),
                            scalar_bounds=payload["scalar_bounds"],
                            tile_width=int(tile_width),
                            tile_height=int(tile_height),
                            point_radius_px=int(point_radius_px),
                            supersample_scale=int(supersample_scale),
                            projection_mode=selected_projection_mode,
                            ortho_scale=payload["ortho_scale"],
                            label=_orbit_tile_label(
                                case_label=str(payload["label"]),
                                point_count=len(payload["ffs_points"]),
                                angle_deg=angle,
                                removed_highlight_stats=payload["ffs_pt_like_removed_highlight_stats"],
                            ),
                        )
                        row_tiles.extend([native_tile, ffs_tile])
                    image_rows.append(row_tiles)
                board = _compose_3x4_orbit_board(
                    title_lines=title_lines,
                    column_headers=column_headers,
                    image_rows=image_rows,
                )
            else:
                image_rows = []
                for payload in payloads:
                    view_config = payload["views"][frame_number]
                    angle = float(view_config["orbit_angle_deg"])
                    native_tile = _render_variant_tile(
                        points=payload["native_points"],
                        colors=payload["native_colors"],
                        view_config=view_config,
                        render_mode=str(render_mode),
                        scalar_bounds=payload["scalar_bounds"],
                        tile_width=int(tile_width),
                        tile_height=int(tile_height),
                        point_radius_px=int(point_radius_px),
                        supersample_scale=int(supersample_scale),
                        projection_mode=selected_projection_mode,
                        ortho_scale=payload["ortho_scale"],
                        label=(
                            f"Native | {tile_mode_label} | "
                            f"{len(payload['native_points'])} pts | {angle:05.1f} deg"
                        ),
                    )
                    ffs_tile = _render_variant_tile(
                        points=payload["ffs_points"],
                        colors=payload["ffs_colors"],
                        view_config=view_config,
                        render_mode=str(render_mode),
                        scalar_bounds=payload["scalar_bounds"],
                        tile_width=int(tile_width),
                        tile_height=int(tile_height),
                        point_radius_px=int(point_radius_px),
                        supersample_scale=int(supersample_scale),
                        projection_mode=selected_projection_mode,
                        ortho_scale=payload["ortho_scale"],
                        label=(
                            f"FFS L{ffs_builder_level} | {tile_mode_label} | "
                            f"{len(payload['ffs_points'])} pts | {angle:05.1f} deg"
                        ),
                    )
                    image_rows.append([native_tile, ffs_tile])
                board = _compose_6x2_orbit_board(
                    title_lines=title_lines,
                    row_headers=row_headers,
                    column_headers=column_headers,
                    image_rows=image_rows,
                    row_label_width=int(row_label_width),
                )
            if frame_number == 0:
                write_image(first_frame_path, board)
            writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            if frame_number == 0 or (frame_number + 1) % 30 == 0 or frame_number + 1 == total_frames:
                print(f"rendered {frame_number + 1}/{total_frames} frames", flush=True)

    case_summaries = []
    for payload in payloads:
        case_summaries.append(
            {
                "label": payload["label"],
                "case_dir": payload["case_dir"],
                "mask_root": payload["mask_root"],
                "text_prompt": payload["text_prompt"],
                "serial_numbers": payload["serial_numbers"],
                "mask_debug": payload["mask_debug"],
                "mask_erode_debug": payload["mask_erode_debug"],
                "native_stats": payload["native_stats"],
                "ffs_stats": payload["ffs_stats"],
                "native_pt_like_stats": payload["native_pt_like_stats"],
                "ffs_pt_like_stats": payload["ffs_pt_like_stats"],
                "native_pt_like_removed_highlight_stats": payload["native_pt_like_removed_highlight_stats"],
                "ffs_pt_like_removed_highlight_stats": payload["ffs_pt_like_removed_highlight_stats"],
                "native_cropped_point_count": payload["native_cropped_point_count"],
                "ffs_cropped_point_count": payload["ffs_cropped_point_count"],
                "native_render_point_count": int(len(payload["native_points"])),
                "ffs_render_point_count": int(len(payload["ffs_points"])),
                "bounds_min": _serialize_array(payload["bounds_min"]),
                "bounds_max": _serialize_array(payload["bounds_max"]),
                "ortho_scale": None if payload["ortho_scale"] is None else float(payload["ortho_scale"]),
                "orbit": payload["orbit"],
            }
        )
    summary = {
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "frame_idx": int(frame_idx),
        "start_camera_idx": int(start_camera_idx),
        "num_frames": total_frames,
        "fps": int(fps),
        "panel_layout": selected_layout,
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "row_label_width": int(row_label_width),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "max_points_per_camera": None if max_points_per_camera is None else int(max_points_per_camera),
        "max_points_per_variant": None if max_points_per_variant is None else int(max_points_per_variant),
        "crop_to_robust_bounds": bool(crop_to_robust_bounds),
        "crop_margin_ratio": float(crop_margin_ratio),
        "render_mode": str(render_mode),
        "projection_mode": selected_projection_mode,
        "point_radius_px": int(point_radius_px),
        "mask_erode_pixels": int(mask_erode_pixels),
        "pt_like_postprocess_enabled": bool(enhanced_pt_like_postprocess),
        "pt_like_postprocess_mode": "enhanced_radius_then_component" if bool(enhanced_pt_like_postprocess) else "none",
        "enhanced_pt_like_removed_highlight_enabled": bool(highlight_enhanced_pt_like_removed),
        "enhanced_pt_like_removed_highlight_mode": (
            "source_camera_color_mark_only" if bool(highlight_enhanced_pt_like_removed) else "none"
        ),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
        "enhanced_component_voxel_size_m": float(enhanced_component_voxel_size_m),
        "enhanced_keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        "source_camera_highlight_colors_bgr": _source_camera_highlight_color_summary(),
        "source_camera_highlight_labels": list(SOURCE_CAMERA_HIGHLIGHT_LABELS),
        "ffs_config": first_ffs_config,
        "case_summaries": case_summaries,
    }
    write_json(summary_path, summary)
    return summary


def run_still_object_rope_6x2_orbit_gif_erode_sweep_workflow(
    *,
    case_specs: list[dict[str, Any]],
    output_root: Path,
    erode_pixels: str | int | list[int] | tuple[int, ...] = DEFAULT_6X2_ERODE_SWEEP_PIXELS,
    frame_idx: int = 0,
    start_camera_idx: int = 0,
    num_frames: int = 360,
    fps: int = 30,
    tile_width: int = 360,
    tile_height: int = 220,
    row_label_width: int = 180,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_variant: int | None = 120_000,
    robust_bounds_percentile: float = 1.0,
    crop_to_robust_bounds: bool = True,
    crop_margin_ratio: float = 0.25,
    render_mode: str = "color_by_rgb",
    projection_mode: str = "orthographic",
    ortho_margin: float = 1.28,
    point_radius_px: int = 1,
    supersample_scale: int = 1,
    layout: str = "3x4",
    highlight_enhanced_pt_like_removed: bool = True,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    erode_values = parse_mask_erode_pixels(erode_pixels)

    variant_summaries: list[dict[str, Any]] = []
    for erode_value in erode_values:
        variant_output_dir = output_root / f"mask_erode_{int(erode_value):02d}px"
        variant_summary = run_still_object_rope_6x2_orbit_gif_workflow(
            case_specs=case_specs,
            output_dir=variant_output_dir,
            frame_idx=int(frame_idx),
            start_camera_idx=int(start_camera_idx),
            num_frames=int(num_frames),
            fps=int(fps),
            tile_width=int(tile_width),
            tile_height=int(tile_height),
            row_label_width=int(row_label_width),
            depth_min_m=float(depth_min_m),
            depth_max_m=float(depth_max_m),
            max_points_per_camera=max_points_per_camera,
            max_points_per_variant=max_points_per_variant,
            robust_bounds_percentile=float(robust_bounds_percentile),
            crop_to_robust_bounds=bool(crop_to_robust_bounds),
            crop_margin_ratio=float(crop_margin_ratio),
            render_mode=str(render_mode),
            projection_mode=str(projection_mode),
            ortho_margin=float(ortho_margin),
            point_radius_px=int(point_radius_px),
            supersample_scale=int(supersample_scale),
            layout=str(layout),
            mask_erode_pixels=int(erode_value),
            enhanced_pt_like_postprocess=False,
            highlight_enhanced_pt_like_removed=bool(highlight_enhanced_pt_like_removed),
            phystwin_radius_m=float(phystwin_radius_m),
            phystwin_nb_points=int(phystwin_nb_points),
            enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        variant_summaries.append(
            {
                "mask_erode_pixels": int(erode_value),
                "output_dir": variant_summary["output_dir"],
                "gif_path": variant_summary["gif_path"],
                "first_frame_path": variant_summary["first_frame_path"],
                "summary_path": str(variant_output_dir / "summary.json"),
                "panel_layout": str(variant_summary["panel_layout"]),
                "pt_like_postprocess_enabled": bool(variant_summary["pt_like_postprocess_enabled"]),
                "enhanced_pt_like_removed_highlight_enabled": bool(
                    variant_summary["enhanced_pt_like_removed_highlight_enabled"]
                ),
                "case_summaries": [
                    {
                        "label": item["label"],
                        "native_render_point_count": int(item["native_render_point_count"]),
                        "ffs_render_point_count": int(item["ffs_render_point_count"]),
                        "mask_erode_debug": item["mask_erode_debug"],
                        "native_pt_like_removed_highlight_stats": item["native_pt_like_removed_highlight_stats"],
                        "ffs_pt_like_removed_highlight_stats": item["ffs_pt_like_removed_highlight_stats"],
                    }
                    for item in variant_summary["case_summaries"]
                ],
            }
        )

    summary = {
        "output_root": str(output_root),
        "erode_pixels": [int(item) for item in erode_values],
        "frame_idx": int(frame_idx),
        "start_camera_idx": int(start_camera_idx),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "row_label_width": int(row_label_width),
        "panel_layout": str(layout).lower(),
        "render_mode": str(render_mode),
        "projection_mode": str(projection_mode),
        "pt_like_postprocess_enabled": False,
        "enhanced_pt_like_removed_highlight_enabled": bool(highlight_enhanced_pt_like_removed),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
        "enhanced_component_voxel_size_m": float(enhanced_component_voxel_size_m),
        "enhanced_keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        "source_camera_highlight_colors_bgr": _source_camera_highlight_color_summary(),
        "source_camera_highlight_labels": list(SOURCE_CAMERA_HIGHLIGHT_LABELS),
        "variants": variant_summaries,
    }
    write_json(output_root / "summary.json", summary)
    return summary
