from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from data_process.depth_backends import (
    FastFoundationStereoRunner,
    align_depth_to_color,
    align_ir_scalar_to_color,
    build_confidence_filtered_depth_uint16,
)
from ..calibration_io import load_calibration_transforms
from ..depth_diagnostics import label_tile
from ..floating_point_diagnostics import detect_radius_outlier_indices
from ..io_artifacts import write_image, write_json
from ..io_case import (
    decode_depth_to_meters,
    depth_to_camera_points,
    get_depth_scale_list,
    load_case_metadata,
    resolve_case_dir,
    transform_points,
)
from ..layouts import compose_registration_matrix_board
from ..object_roi import _build_voxel_components
from ..views import build_original_camera_view_configs
from .ffs_confidence_panels import DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT
from .ffs_confidence_pcd_panels import _render_open3d_offscreen_pinhole
from ..workflows.masked_camera_view_compare import _image_size_from_color_path, _scale_intrinsic_matrix
from ..workflows.masked_pointcloud_compare import PHYSTWIN_DATA_PROCESS_MASK_CONTRACT, load_union_masks_for_camera_clouds


DEFAULT_DEPTH_SCALE_M_PER_UNIT = 0.001
CONFIDENCE_FILTER_MODES: tuple[str, ...] = ("margin", "max_softmax", "entropy", "variance")
DEFAULT_STATIC_CONFIDENCE_FILTER_ROUNDS: tuple[dict[str, str], ...] = (
    {
        "round_id": "round1",
        "round_label": "Round 1",
        "native_case_ref": "static/native_30_static_round1_20260410_235202",
        "ffs_case_ref": "static/ffs_30_static_round1_20260410_235202",
        "mask_root": "static/masked_pointcloud_compare_round1_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks",
    },
    {
        "round_id": "round2",
        "round_label": "Round 2",
        "native_case_ref": "static/native_30_static_round2_20260414",
        "ffs_case_ref": "static/ffs_30_static_round2_20260414",
        "mask_root": "static/masked_pointcloud_compare_round2_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks",
    },
    {
        "round_id": "round3",
        "round_label": "Round 3",
        "native_case_ref": "static/native_30_static_round3_20260414",
        "ffs_case_ref": "static/ffs_30_static_round3_20260414",
        "mask_root": "static/masked_pointcloud_compare_round3_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks",
    },
)


def build_static_confidence_filter_round_specs(*, aligned_root: Path) -> list[dict[str, Any]]:
    root = Path(aligned_root).resolve()
    specs: list[dict[str, Any]] = []
    for item in DEFAULT_STATIC_CONFIDENCE_FILTER_ROUNDS:
        spec = dict(item)
        spec["mask_root"] = (root / str(item["mask_root"])).resolve()
        specs.append(spec)
    return specs


def _load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing color image: {path}")
    return image


def _load_ir_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing IR image: {path}")
    return image


def _depth_scale_for_camera(metadata: dict[str, Any], camera_idx: int) -> float:
    scales = get_depth_scale_list(metadata, len(metadata["serial_numbers"]))
    scale = scales[int(camera_idx)]
    if scale is None:
        return float(DEFAULT_DEPTH_SCALE_M_PER_UNIT)
    scale_value = float(scale)
    if scale_value <= 0.0:
        raise ValueError(f"depth_scale_m_per_unit must be positive for camera {camera_idx}, got {scale}.")
    return scale_value


def _load_depth_m_from_depth_dir(*, case_dir: Path, metadata: dict[str, Any], camera_idx: int, frame_idx: int) -> tuple[np.ndarray, str]:
    depth_path = case_dir / "depth" / str(int(camera_idx)) / f"{int(frame_idx)}.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth frame: {depth_path}")
    depth_raw = np.load(depth_path)
    scale = None if np.issubdtype(depth_raw.dtype, np.floating) else _depth_scale_for_camera(metadata, int(camera_idx))
    return decode_depth_to_meters(depth_raw, scale), str(depth_path.resolve())


def _build_world_cloud(
    *,
    depth_m: np.ndarray,
    color_image: np.ndarray,
    K_color: np.ndarray,
    c2w: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_camera: int | None,
    object_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    depth_for_cloud = np.asarray(depth_m, dtype=np.float32)
    if object_mask is not None:
        mask = np.asarray(object_mask, dtype=bool)
        if mask.shape != depth_for_cloud.shape:
            raise ValueError(f"object_mask shape must match depth shape. Got {mask.shape} vs {depth_for_cloud.shape}.")
        depth_for_cloud = np.where(mask, depth_for_cloud, 0.0).astype(np.float32, copy=False)
    camera_points, camera_colors, source_pixel_uv, stats = depth_to_camera_points(
        depth_for_cloud,
        K_color,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        color_image=color_image,
        pixel_roi=None,
        max_points_per_camera=max_points_per_camera,
    )
    world_points = transform_points(camera_points, c2w)
    return {
        "points": world_points,
        "colors": camera_colors,
        "source_pixel_uv": source_pixel_uv,
        "stats": dict(stats),
    }


def _fuse_camera_clouds(camera_clouds: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [np.asarray(item["points"], dtype=np.float32) for item in camera_clouds if len(item["points"]) > 0]
    color_sets = [np.asarray(item["colors"], dtype=np.uint8) for item in camera_clouds if len(item["points"]) > 0]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def _apply_phystwin_like_radius_postprocess(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    point_count = int(len(point_array))
    stats = {
        "enabled": bool(enabled),
        "mode": "phystwin_like_radius_neighbor_filter",
        "radius_m": float(radius_m),
        "nb_points": int(nb_points),
        "input_point_count": point_count,
        "inlier_point_count": point_count,
        "outlier_point_count": 0,
        "outlier_ratio": 0.0,
    }
    if not enabled:
        return point_array, color_array, stats
    if point_count == 0:
        return point_array, color_array, stats

    result = detect_radius_outlier_indices(
        point_array,
        radius_m=float(radius_m),
        nb_points=int(nb_points),
    )
    inlier_indices = np.sort(np.asarray(result["inlier_indices"], dtype=np.int32).reshape(-1))
    outlier_count = int(point_count - len(inlier_indices))
    stats.update(
        {
            "inlier_point_count": int(len(inlier_indices)),
            "outlier_point_count": outlier_count,
            "outlier_ratio": float(outlier_count / max(1, point_count)),
        }
    )
    return point_array[inlier_indices], color_array[inlier_indices], stats


def _bbox_gap_m(left_min: np.ndarray, left_max: np.ndarray, right_min: np.ndarray, right_max: np.ndarray) -> float:
    left_lo = np.asarray(left_min, dtype=np.float32).reshape(3)
    left_hi = np.asarray(left_max, dtype=np.float32).reshape(3)
    right_lo = np.asarray(right_min, dtype=np.float32).reshape(3)
    right_hi = np.asarray(right_max, dtype=np.float32).reshape(3)
    axis_gap = np.maximum(np.maximum(left_lo - right_hi, right_lo - left_hi), 0.0)
    return float(np.linalg.norm(axis_gap))


def _component_summary(
    *,
    component_idx: int,
    component_indices: np.ndarray,
    points: np.ndarray,
    main_bbox_min: np.ndarray,
    main_bbox_max: np.ndarray,
    kept: bool,
) -> dict[str, Any]:
    indices = np.asarray(component_indices, dtype=np.int32).reshape(-1)
    component_points = np.asarray(points, dtype=np.float32)[indices]
    bbox_min = component_points.min(axis=0).astype(np.float32)
    bbox_max = component_points.max(axis=0).astype(np.float32)
    centroid = component_points.mean(axis=0).astype(np.float32)
    extent = (bbox_max - bbox_min).astype(np.float32)
    return {
        "component_idx": int(component_idx),
        "kept": bool(kept),
        "point_count": int(len(indices)),
        "bbox_min": [float(item) for item in bbox_min],
        "bbox_max": [float(item) for item in bbox_max],
        "bbox_extent": [float(item) for item in extent],
        "centroid": [float(item) for item in centroid],
        "bbox_gap_to_main_m": float(_bbox_gap_m(bbox_min, bbox_max, main_bbox_min, main_bbox_max)),
    }


def _apply_enhanced_phystwin_like_postprocess_with_trace(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
    component_voxel_size_m: float,
    keep_near_main_gap_m: float = 0.0,
    max_component_report_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    input_point_count = int(len(point_array))
    radius_kept_mask = np.ones((input_point_count,), dtype=bool)
    radius_stats = {
        "enabled": bool(enabled),
        "mode": "phystwin_like_radius_neighbor_filter",
        "radius_m": float(radius_m),
        "nb_points": int(nb_points),
        "input_point_count": input_point_count,
        "inlier_point_count": input_point_count,
        "outlier_point_count": 0,
        "outlier_ratio": 0.0,
    }
    if bool(enabled) and input_point_count > 0:
        result = detect_radius_outlier_indices(
            point_array,
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        inlier_indices = np.sort(np.asarray(result["inlier_indices"], dtype=np.int32).reshape(-1))
        radius_kept_mask[:] = False
        radius_kept_mask[inlier_indices] = True
        outlier_count = int(input_point_count - len(inlier_indices))
        radius_stats.update(
            {
                "inlier_point_count": int(len(inlier_indices)),
                "outlier_point_count": outlier_count,
                "outlier_ratio": float(outlier_count / max(1, input_point_count)),
            }
        )
    radius_indices = np.where(radius_kept_mask)[0].astype(np.int32)
    radius_points = point_array[radius_indices]
    radius_colors = color_array[radius_indices]
    component_removed_mask = np.zeros((input_point_count,), dtype=bool)
    trace = {
        "kept_mask": radius_kept_mask.copy(),
        "radius_removed_mask": ~radius_kept_mask,
        "component_removed_mask": component_removed_mask.copy(),
        "removed_mask": ~radius_kept_mask,
    }
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": "enhanced_phystwin_like_radius_then_component_filter",
        "radius_postprocess": dict(radius_stats),
        "component_filter_enabled": bool(enabled),
        "component_voxel_size_m": float(component_voxel_size_m),
        "keep_near_main_gap_m": float(keep_near_main_gap_m),
        "input_point_count": input_point_count,
        "after_radius_point_count": int(len(radius_points)),
        "output_point_count": int(len(radius_points)),
        "component_count": 0,
        "kept_component_indices": [],
        "removed_component_count": 0,
        "removed_point_count": 0,
        "removed_point_ratio_after_radius": 0.0,
        "components": [],
        "removed_components": [],
    }
    if not enabled or len(radius_points) == 0:
        return radius_points, radius_colors, stats, trace
    if float(component_voxel_size_m) <= 0.0:
        raise ValueError(f"component_voxel_size_m must be positive, got {component_voxel_size_m}.")
    if float(keep_near_main_gap_m) < 0.0:
        raise ValueError(f"keep_near_main_gap_m must be >= 0, got {keep_near_main_gap_m}.")

    components = _build_voxel_components(radius_points, voxel_size=float(component_voxel_size_m))
    stats["component_count"] = int(len(components))
    if len(components) <= 1:
        if components:
            stats["kept_component_indices"] = [0]
            main_points = radius_points[np.asarray(components[0], dtype=np.int32)]
            main_bbox_min = main_points.min(axis=0)
            main_bbox_max = main_points.max(axis=0)
            stats["components"] = [
                _component_summary(
                    component_idx=0,
                    component_indices=components[0],
                    points=radius_points,
                    main_bbox_min=main_bbox_min,
                    main_bbox_max=main_bbox_max,
                    kept=True,
                )
            ]
        return radius_points, radius_colors, stats, trace

    main_points = radius_points[np.asarray(components[0], dtype=np.int32)]
    main_bbox_min = main_points.min(axis=0).astype(np.float32)
    main_bbox_max = main_points.max(axis=0).astype(np.float32)
    component_keep_mask = np.zeros((len(radius_points),), dtype=bool)
    kept_component_indices: list[int] = []
    component_summaries: list[dict[str, Any]] = []
    removed_summaries: list[dict[str, Any]] = []
    for component_idx, component_indices in enumerate(components):
        indices = np.asarray(component_indices, dtype=np.int32).reshape(-1)
        if component_idx == 0:
            keep_component = True
        else:
            component_points = radius_points[indices]
            bbox_min = component_points.min(axis=0).astype(np.float32)
            bbox_max = component_points.max(axis=0).astype(np.float32)
            keep_component = _bbox_gap_m(bbox_min, bbox_max, main_bbox_min, main_bbox_max) <= float(keep_near_main_gap_m)
        if keep_component:
            component_keep_mask[indices] = True
            kept_component_indices.append(int(component_idx))
        summary = _component_summary(
            component_idx=int(component_idx),
            component_indices=indices,
            points=radius_points,
            main_bbox_min=main_bbox_min,
            main_bbox_max=main_bbox_max,
            kept=bool(keep_component),
        )
        component_summaries.append(summary)
        if not keep_component:
            removed_summaries.append(summary)

    kept_count = int(np.count_nonzero(component_keep_mask))
    removed_count = int(len(radius_points) - kept_count)
    component_removed_radius_mask = ~component_keep_mask
    component_removed_mask[radius_indices[component_removed_radius_mask]] = True
    kept_mask = radius_kept_mask & ~component_removed_mask
    trace = {
        "kept_mask": kept_mask,
        "radius_removed_mask": ~radius_kept_mask,
        "component_removed_mask": component_removed_mask,
        "removed_mask": (~radius_kept_mask) | component_removed_mask,
    }
    stats.update(
        {
            "output_point_count": kept_count,
            "kept_component_indices": kept_component_indices,
            "removed_component_count": int(len(removed_summaries)),
            "removed_point_count": removed_count,
            "removed_point_ratio_after_radius": float(removed_count / max(1, len(radius_points))),
            "components": component_summaries[: max(1, int(max_component_report_count))],
            "removed_components": removed_summaries[: max(1, int(max_component_report_count))],
        }
    )
    return radius_points[component_keep_mask], radius_colors[component_keep_mask], stats, trace


def _apply_enhanced_phystwin_like_postprocess(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
    component_voxel_size_m: float,
    keep_near_main_gap_m: float = 0.0,
    max_component_report_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    filtered_points, filtered_colors, stats, _trace = _apply_enhanced_phystwin_like_postprocess_with_trace(
        points=points,
        colors=colors,
        enabled=enabled,
        radius_m=radius_m,
        nb_points=nb_points,
        component_voxel_size_m=component_voxel_size_m,
        keep_near_main_gap_m=keep_near_main_gap_m,
        max_component_report_count=max_component_report_count,
    )
    return filtered_points, filtered_colors, stats


def _format_point_count(point_count: int) -> str:
    count = int(point_count)
    if count >= 1_000_000:
        return f"pts={count / 1_000_000.0:.2f}M"
    if count >= 1_000:
        return f"pts={count / 1_000.0:.1f}k"
    return f"pts={count}"


def _confidence_stats(confidence: np.ndarray, *, valid_depth: np.ndarray) -> dict[str, float]:
    values = np.asarray(confidence, dtype=np.float32)[np.asarray(valid_depth, dtype=bool)]
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _serialize_view_config(view_config: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in view_config.items():
        if isinstance(value, np.ndarray):
            payload[key] = [float(item) for item in np.asarray(value, dtype=np.float32).reshape(-1)]
        elif isinstance(value, (np.floating, np.integer)):
            payload[key] = float(value)
        else:
            payload[key] = value
    return payload


def _variant_rows(threshold: float) -> list[dict[str, str]]:
    suffix = f"{float(threshold):.2f}"
    return [
        {"key": "native", "row_header": "Native", "summary_label": "native_depth"},
        {"key": "ffs_original", "row_header": "FFS raw", "summary_label": "ffs_rerun_unfiltered"},
        {"key": "ffs_margin", "row_header": f"margin {suffix}", "summary_label": "ffs_margin_filtered"},
        {"key": "ffs_max_softmax", "row_header": f"maxsm {suffix}", "summary_label": "ffs_max_softmax_filtered"},
        {"key": "ffs_entropy", "row_header": f"entropy {suffix}", "summary_label": "ffs_entropy_filtered"},
        {"key": "ffs_variance", "row_header": f"variance {suffix}", "summary_label": "ffs_variance_filtered"},
    ]


def build_confidence_filter_pcd_board(
    *,
    round_label: str,
    frame_idx: int,
    confidence_threshold: float,
    model_config: dict[str, Any],
    column_headers: list[str],
    variant_rows: list[dict[str, str]],
    rendered_rows: list[list[np.ndarray]],
) -> np.ndarray:
    row_headers = [str(item["row_header"]) for item in variant_rows]
    postprocess_suffix = ""
    if bool(model_config.get("phystwin_like_postprocess_enabled", False)):
        postprocess_suffix = (
            f" | post=radius {float(model_config.get('phystwin_radius_m', 0.0)):.3f}m "
            f"nb={int(model_config.get('phystwin_nb_points', 0))}"
        )
    return compose_registration_matrix_board(
        title_lines=[
            f"Native vs FFS Confidence Filter PCD | {round_label} | frame {int(frame_idx)} | conf >= {float(confidence_threshold):.2f}",
            (
                f"object_mask={str(model_config['object_mask_enabled']).lower()} | scale={float(model_config['scale']):.2f} | "
                f"iters={int(model_config['valid_iters'])} | "
                f"disp={int(model_config['max_disp'])} | depth=[{float(model_config['depth_min_m']):.2f},"
                f"{float(model_config['depth_max_m']):.2f}]m{postprocess_suffix}"
            ),
        ],
        row_headers=row_headers,
        column_headers=column_headers,
        image_rows=rendered_rows,
    )


def _build_view_configs(
    *,
    metadata: dict[str, Any],
    c2w_list: list[np.ndarray],
    case_dir: Path,
    frame_idx: int,
    camera_ids: list[int],
    tile_width: int,
    tile_height: int,
    look_distance: float,
) -> list[dict[str, Any]]:
    serial_numbers = [str(metadata["serial_numbers"][camera_idx]) for camera_idx in camera_ids]
    selected_c2w = [np.asarray(c2w_list[camera_idx], dtype=np.float32) for camera_idx in camera_ids]
    view_configs = build_original_camera_view_configs(
        c2w_list=selected_c2w,
        serial_numbers=serial_numbers,
        look_distance=float(look_distance),
        camera_ids=list(range(len(camera_ids))),
    )
    for idx, view_config in enumerate(view_configs):
        camera_idx = int(camera_ids[idx])
        color_path = case_dir / "color" / str(camera_idx) / f"{int(frame_idx)}.png"
        source_image_size = _image_size_from_color_path(color_path)
        target_image_size = (int(tile_width), int(tile_height))
        view_config["camera_idx"] = camera_idx
        view_config["view_name"] = f"cam{camera_idx}"
        view_config["label"] = f"Cam{camera_idx} | {serial_numbers[idx]}"
        view_config["intrinsic_matrix"] = _scale_intrinsic_matrix(
            np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
            source_size=source_image_size,
            target_size=target_image_size,
        )
        view_config["extrinsic_matrix"] = np.linalg.inv(np.asarray(c2w_list[camera_idx], dtype=np.float32).reshape(4, 4)).astype(np.float32)
        view_config["image_size"] = [int(tile_width), int(tile_height)]
    return view_configs


def run_ffs_confidence_filter_pcd_compare_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    ffs_repo: str | Path,
    model_path: str | Path,
    scale: float = 1.0,
    valid_iters: int = 4,
    max_disp: int = 192,
    frame_idx: int = 0,
    confidence_threshold: float = 0.10,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    point_size: float = 2.0,
    look_distance: float = 1.0,
    tile_width: int = 480,
    tile_height: int = 360,
    max_points_per_camera: int | None = 80_000,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    use_object_mask: bool = True,
    phystwin_like_postprocess: bool = False,
    phystwin_radius_m: float = float(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["radius_m"]),
    phystwin_nb_points: int = int(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["nb_points"]),
    round_specs: list[dict[str, Any]] | None = None,
    runner_factory: Callable[..., Any] | None = None,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    round_specs = build_static_confidence_filter_round_specs(aligned_root=aligned_root) if round_specs is None else list(round_specs)
    if float(confidence_threshold) < 0.0 or float(confidence_threshold) > 1.0:
        raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}.")
    if float(phystwin_radius_m) <= 0.0:
        raise ValueError(f"phystwin_radius_m must be positive, got {phystwin_radius_m}.")
    if int(phystwin_nb_points) < 1:
        raise ValueError(f"phystwin_nb_points must be >= 1, got {phystwin_nb_points}.")

    runner_factory = FastFoundationStereoRunner if runner_factory is None else runner_factory
    runner = runner_factory(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=float(scale),
        valid_iters=int(valid_iters),
        max_disp=int(max_disp),
    )
    render_frame_fn = render_frame_fn or _render_open3d_offscreen_pinhole
    variant_rows = _variant_rows(float(confidence_threshold))
    model_config = {
        "ffs_repo": str(Path(ffs_repo).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "scale": float(scale),
        "valid_iters": int(valid_iters),
        "max_disp": int(max_disp),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "point_size": float(point_size),
        "look_distance": float(look_distance),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "max_points_per_camera": None if max_points_per_camera is None else int(max_points_per_camera),
        "object_mask_enabled": bool(use_object_mask),
        "text_prompt": str(text_prompt),
        "phystwin_like_postprocess_enabled": bool(phystwin_like_postprocess),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
    }

    rounds_summary: list[dict[str, Any]] = []
    for round_spec in round_specs:
        native_case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(round_spec["native_case_ref"]))
        ffs_case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(round_spec["ffs_case_ref"]))
        native_metadata = load_case_metadata(native_case_dir)
        ffs_metadata = load_case_metadata(ffs_case_dir)
        selected_frame_idx = int(frame_idx)
        max_frame = min(int(native_metadata["frame_num"]), int(ffs_metadata["frame_num"])) - 1
        if selected_frame_idx < 0 or selected_frame_idx > max_frame:
            raise ValueError(
                f"frame_idx={selected_frame_idx} is out of range for {round_spec['round_id']}; "
                f"expected 0 <= frame_idx <= {max_frame}."
            )

        camera_ids = list(range(len(ffs_metadata["serial_numbers"])))
        if len(camera_ids) != 3:
            raise ValueError(f"Expected exactly 3 cameras for {round_spec['round_id']}, got {camera_ids}.")
        if len(native_metadata["serial_numbers"]) != len(camera_ids):
            raise ValueError(
                f"Native and FFS camera counts differ for {round_spec['round_id']}: "
                f"{len(native_metadata['serial_numbers'])} vs {len(camera_ids)}."
            )

        round_output_dir = output_root / str(round_spec["round_id"])
        round_output_dir.mkdir(parents=True, exist_ok=True)
        native_c2w_list = load_calibration_transforms(
            native_case_dir / "calibrate.pkl",
            serial_numbers=native_metadata["serial_numbers"],
            calibration_reference_serials=native_metadata.get("calibration_reference_serials", native_metadata["serial_numbers"]),
        )
        ffs_c2w_list = load_calibration_transforms(
            ffs_case_dir / "calibrate.pkl",
            serial_numbers=ffs_metadata["serial_numbers"],
            calibration_reference_serials=ffs_metadata.get("calibration_reference_serials", ffs_metadata["serial_numbers"]),
        )
        view_configs = _build_view_configs(
            metadata=ffs_metadata,
            c2w_list=ffs_c2w_list,
            case_dir=ffs_case_dir,
            frame_idx=selected_frame_idx,
            camera_ids=camera_ids,
            tile_width=int(tile_width),
            tile_height=int(tile_height),
            look_distance=float(look_distance),
        )
        column_headers = [str(view_config["label"]) for view_config in view_configs]
        mask_by_camera: dict[int, np.ndarray] = {}
        mask_debug: dict[int, dict[str, Any]] = {}
        if use_object_mask:
            mask_root = Path(round_spec["mask_root"]).resolve()
            minimal_clouds = [
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                    "color_path": str(ffs_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"),
                }
                for camera_idx in camera_ids
            ]
            mask_by_camera, mask_debug = load_union_masks_for_camera_clouds(
                mask_root=mask_root,
                camera_clouds=minimal_clouds,
                frame_token=str(selected_frame_idx),
                text_prompt=str(text_prompt),
            )

        camera_clouds_by_variant: dict[str, list[dict[str, Any]]] = {str(item["key"]): [] for item in variant_rows}
        per_variant_camera_stats: dict[str, list[dict[str, Any]]] = {str(item["key"]): [] for item in variant_rows}

        for camera_idx in camera_ids:
            native_color_path = native_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            native_color_image = _load_color_image(native_color_path)
            native_depth_m, native_depth_path = _load_depth_m_from_depth_dir(
                case_dir=native_case_dir,
                metadata=native_metadata,
                camera_idx=camera_idx,
                frame_idx=selected_frame_idx,
            )
            native_cloud = _build_world_cloud(
                depth_m=native_depth_m,
                color_image=native_color_image,
                K_color=np.asarray(native_metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(native_c2w_list[camera_idx], dtype=np.float32),
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                max_points_per_camera=max_points_per_camera,
                object_mask=mask_by_camera.get(int(camera_idx)) if use_object_mask else None,
            )
            camera_clouds_by_variant["native"].append(native_cloud)
            per_variant_camera_stats["native"].append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(native_metadata["serial_numbers"][camera_idx]),
                    "color_path": str(native_color_path.resolve()),
                    "depth_path": native_depth_path,
                    "object_mask_enabled": bool(use_object_mask),
                    "mask_pixel_count": int(np.count_nonzero(mask_by_camera.get(int(camera_idx), np.zeros(native_depth_m.shape, dtype=bool)))),
                    "point_count": int(len(native_cloud["points"])),
                    **dict(native_cloud["stats"]),
                }
            )

            ffs_color_path = ffs_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            ir_left_path = ffs_case_dir / "ir_left" / str(camera_idx) / f"{selected_frame_idx}.png"
            ir_right_path = ffs_case_dir / "ir_right" / str(camera_idx) / f"{selected_frame_idx}.png"
            ffs_color_image = _load_color_image(ffs_color_path)
            ir_left = _load_ir_image(ir_left_path)
            ir_right = _load_ir_image(ir_right_path)
            ffs_output = runner.run_pair_with_confidence(
                ir_left,
                ir_right,
                K_ir_left=np.asarray(ffs_metadata["K_ir_left"][camera_idx], dtype=np.float32),
                baseline_m=float(ffs_metadata["ir_baseline_m"][camera_idx]),
                audit_mode=False,
            )
            depth_ir_left_m = np.asarray(ffs_output["depth_ir_left_m"], dtype=np.float32)
            K_ir_left_used = np.asarray(ffs_output["K_ir_left_used"], dtype=np.float32)
            T_ir_left_to_color = np.asarray(ffs_metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
            K_color = np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32)
            output_shape = (int(ffs_color_image.shape[0]), int(ffs_color_image.shape[1]))
            depth_color_m = align_depth_to_color(
                depth_ir_left_m,
                K_ir_left_used,
                T_ir_left_to_color,
                K_color,
                output_shape=output_shape,
                invalid_value=0.0,
            )
            ffs_cloud = _build_world_cloud(
                depth_m=depth_color_m,
                color_image=ffs_color_image,
                K_color=K_color,
                c2w=np.asarray(ffs_c2w_list[camera_idx], dtype=np.float32),
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                max_points_per_camera=max_points_per_camera,
                object_mask=mask_by_camera.get(int(camera_idx)) if use_object_mask else None,
            )
            camera_clouds_by_variant["ffs_original"].append(ffs_cloud)
            depth_valid_mask = np.isfinite(depth_color_m) & (depth_color_m > 0.0)
            per_variant_camera_stats["ffs_original"].append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                    "color_path": str(ffs_color_path.resolve()),
                    "ir_left_path": str(ir_left_path.resolve()),
                    "ir_right_path": str(ir_right_path.resolve()),
                    "source": "pyTorch_ffs_rerun_unfiltered",
                    "object_mask_enabled": bool(use_object_mask),
                    "mask_pixel_count": int(np.count_nonzero(mask_by_camera.get(int(camera_idx), np.zeros(output_shape, dtype=bool)))),
                    "point_count": int(len(ffs_cloud["points"])),
                    "aligned_valid_pixel_count": int(np.count_nonzero(depth_valid_mask)),
                    **dict(ffs_cloud["stats"]),
                }
            )

            depth_scale = _depth_scale_for_camera(ffs_metadata, int(camera_idx))
            for mode in CONFIDENCE_FILTER_MODES:
                confidence_ir = np.asarray(ffs_output[f"confidence_{mode}_ir_left"], dtype=np.float32)
                confidence_color = align_ir_scalar_to_color(
                    depth_ir_left_m,
                    confidence_ir,
                    K_ir_left_used,
                    T_ir_left_to_color,
                    K_color,
                    output_shape=output_shape,
                    invalid_value=0.0,
                )
                filter_output = build_confidence_filtered_depth_uint16(
                    depth_m=depth_color_m,
                    confidence=confidence_color,
                    confidence_threshold=float(confidence_threshold),
                    depth_scale_m_per_unit=float(depth_scale),
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    object_mask=mask_by_camera.get(int(camera_idx)) if use_object_mask else None,
                )
                filtered_depth_m = decode_depth_to_meters(
                    np.asarray(filter_output["depth_uint16"], dtype=np.uint16),
                    float(depth_scale),
                )
                filtered_cloud = _build_world_cloud(
                    depth_m=filtered_depth_m,
                    color_image=ffs_color_image,
                    K_color=K_color,
                    c2w=np.asarray(ffs_c2w_list[camera_idx], dtype=np.float32),
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    max_points_per_camera=max_points_per_camera,
                    object_mask=None,
                )
                variant_key = f"ffs_{mode}"
                camera_clouds_by_variant[variant_key].append(filtered_cloud)
                per_variant_camera_stats[variant_key].append(
                    {
                        "camera_idx": int(camera_idx),
                        "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                        "source": "pyTorch_ffs_confidence_filtered",
                        "confidence_mode": str(mode),
                        "confidence_threshold": float(confidence_threshold),
                        "depth_scale_m_per_unit": float(depth_scale),
                        "object_mask_enabled": bool(use_object_mask),
                        "mask_pixel_count": int(np.count_nonzero(mask_by_camera.get(int(camera_idx), np.zeros(output_shape, dtype=bool)))),
                        "point_count": int(len(filtered_cloud["points"])),
                        "confidence": _confidence_stats(confidence_color, valid_depth=depth_valid_mask),
                        "filter_stats": dict(filter_output["stats"]),
                        **dict(filtered_cloud["stats"]),
                    }
                )

        fused_by_variant: dict[str, dict[str, Any]] = {}
        for variant in variant_rows:
            variant_key = str(variant["key"])
            raw_points, raw_colors = _fuse_camera_clouds(camera_clouds_by_variant[variant_key])
            points, colors, postprocess_stats = _apply_phystwin_like_radius_postprocess(
                points=raw_points,
                colors=raw_colors,
                enabled=bool(phystwin_like_postprocess),
                radius_m=float(phystwin_radius_m),
                nb_points=int(phystwin_nb_points),
            )
            fused_by_variant[variant_key] = {
                "points": points,
                "colors": colors,
                "point_count_before_postprocess": int(len(raw_points)),
                "point_count": int(len(points)),
                "postprocess": postprocess_stats,
            }

        rendered_rows: list[list[np.ndarray]] = []
        render_summary: dict[str, list[dict[str, Any]]] = {}
        for variant in variant_rows:
            variant_key = str(variant["key"])
            fused = fused_by_variant[variant_key]
            row_images: list[np.ndarray] = []
            render_summary[variant_key] = []
            for view_config in view_configs:
                target_w, target_h = [int(item) for item in view_config["image_size"]]
                rendered = render_frame_fn(
                    fused["points"],
                    fused["colors"],
                    width=int(target_w),
                    height=int(target_h),
                    center=np.asarray(view_config["center"], dtype=np.float32),
                    eye=np.asarray(view_config["camera_position"], dtype=np.float32),
                    up=np.asarray(view_config["up"], dtype=np.float32),
                    zoom=0.55,
                    point_size=float(point_size),
                    intrinsic_matrix=np.asarray(view_config["intrinsic_matrix"], dtype=np.float32),
                    extrinsic_matrix=np.asarray(view_config["extrinsic_matrix"], dtype=np.float32),
                    render_kind="confidence_filter_pcd_compare",
                    metric_name=variant_key,
                    camera_idx=int(view_config["camera_idx"]),
                )
                rendered = label_tile(
                    rendered,
                    _format_point_count(int(fused["point_count"])),
                    (int(tile_width), int(tile_height)),
                )
                row_images.append(rendered)
                render_summary[variant_key].append(
                    {
                        "camera_idx": int(view_config["camera_idx"]),
                        "point_count": int(fused["point_count"]),
                        "tile_width": int(tile_width),
                        "tile_height": int(tile_height),
                    }
                )
            rendered_rows.append(row_images)

        board = build_confidence_filter_pcd_board(
            round_label=str(round_spec["round_label"]),
            frame_idx=selected_frame_idx,
            confidence_threshold=float(confidence_threshold),
            model_config=model_config,
            column_headers=column_headers,
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )
        board_path = round_output_dir / f"confidence_filter_pcd_6x3_frame_{selected_frame_idx:04d}_threshold_{float(confidence_threshold):.2f}.png"
        write_image(board_path, board)

        round_summary = {
            "round_id": str(round_spec["round_id"]),
            "round_label": str(round_spec["round_label"]),
            "native_case_ref": str(round_spec["native_case_ref"]),
            "ffs_case_ref": str(round_spec["ffs_case_ref"]),
            "native_case_dir": str(native_case_dir.resolve()),
            "ffs_case_dir": str(ffs_case_dir.resolve()),
            "frame_idx": int(selected_frame_idx),
            "confidence_threshold": float(confidence_threshold),
            "confidence_modes": list(CONFIDENCE_FILTER_MODES),
            "row_headers": [str(item["row_header"]) for item in variant_rows],
            "variant_rows": variant_rows,
            "column_headers": column_headers,
            "board_path": str(board_path.resolve()),
            "model_config": dict(model_config),
            "mask_root": str(Path(round_spec["mask_root"]).resolve()) if use_object_mask else None,
            "mask_debug": {str(key): value for key, value in mask_debug.items()},
            "render_contract": {
                "renderer": "open3d_offscreen_renderer",
                "projection_mode": "original_camera_pinhole",
                "columns": "original_ffs_camera_views",
                "rows": "native_depth_ffs_raw_and_confidence_filtered_ffs",
                "display_postprocess": "phystwin_like_radius_neighbor_filter" if phystwin_like_postprocess else "none",
                "display_postprocess_applied_to": "fused_object_pcd_rows_before_rendering",
                "filtered_depth_encoding": "uint16_depth_scale_m_per_unit_in_memory_invalid_zero",
                "object_masked": bool(use_object_mask),
                "mask_source": "static_sam31_ffs_masks",
                "formal_depth_written": False,
            },
            "column_views": [_serialize_view_config(view_config) for view_config in view_configs],
            "fused_point_counts": {
                str(variant["key"]): int(fused_by_variant[str(variant["key"])]["point_count"])
                for variant in variant_rows
            },
            "fused_point_counts_before_postprocess": {
                str(variant["key"]): int(fused_by_variant[str(variant["key"])]["point_count_before_postprocess"])
                for variant in variant_rows
            },
            "postprocess": {
                "enabled": bool(phystwin_like_postprocess),
                "mode": "phystwin_like_radius_neighbor_filter" if phystwin_like_postprocess else "none",
                "radius_m": float(phystwin_radius_m),
                "nb_points": int(phystwin_nb_points),
                "applied_to": "fused_object_pcd_rows_before_rendering",
                "reference_contract": dict(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT),
            },
            "postprocess_stats_by_variant": {
                str(variant["key"]): dict(fused_by_variant[str(variant["key"])]["postprocess"])
                for variant in variant_rows
            },
            "per_variant_camera": per_variant_camera_stats,
            "render_summary": render_summary,
            "output_dir": str(round_output_dir.resolve()),
        }
        write_json(round_output_dir / "summary.json", round_summary)
        rounds_summary.append(round_summary)

    manifest = {
        "output_dir": str(output_root.resolve()),
        "frame_idx": int(frame_idx),
        "confidence_threshold": float(confidence_threshold),
        "confidence_modes": list(CONFIDENCE_FILTER_MODES),
        "model_config": dict(model_config),
        "rounds": rounds_summary,
    }
    write_json(output_root / "summary.json", manifest)
    return manifest
