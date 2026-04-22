from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..floating_point_diagnostics import detect_radius_outlier_indices
from ..io_artifacts import write_image, write_json, write_ply_ascii
from ..io_case import (
    get_frame_count,
    load_case_frame_cloud_with_sources,
    load_case_metadata,
    resolve_case_dirs,
)
from ..layouts import fit_image_to_canvas, overlay_large_panel_label
from ..object_compare import point_mask_from_pixel_mask
from ..pointcloud_defaults import DEFAULT_POINTCLOUD_DEPTH_MAX_M, DEFAULT_POINTCLOUD_DEPTH_MIN_M
from ..roi import crop_points_to_bounds
from ..triplet_ply_compare import _case_has_ffs_raw_depth
from ..triplet_video_compare import _render_open3d_hidden_window
from ..views import compute_view_config


PROMPT_SPLIT_PATTERN = re.compile(r"[,\n;]+|(?<!\d)\.(?!\d)")
MASKED_POINTCLOUD_RENDER_CONTRACT = {
    "renderer": "open3d_hidden_visualizer",
    "view_name": "oblique",
    "shared_view_across_variants": True,
    "shared_crop_across_variants": True,
    "render_mode": "color_by_rgb",
}
MASKED_POINTCLOUD_VARIANTS = (
    ("native_unmasked", "Native Unmasked"),
    ("native_masked", "Native Masked"),
    ("ffs_unmasked", "FFS Unmasked"),
    ("ffs_masked", "FFS Masked"),
)
MIN_MASKED_POINT_COUNT_FOR_FOCUS = 32
PHYSTWIN_DATA_PROCESS_MASK_CONTRACT = {
    "mode": "phystwin_data_process_mask",
    "nb_points": 40,
    "radius_m": 0.01,
    "implementation": "masked_fused_pointcloud_remove_radius_outlier_then_clear_source_pixels",
}


def parse_text_prompts(text_prompt: str) -> list[str]:
    prompts: list[str] = []
    for chunk in PROMPT_SPLIT_PATTERN.split(text_prompt):
        normalized = " ".join(chunk.strip().lower().split())
        if normalized and normalized not in prompts:
            prompts.append(normalized)
    return prompts


def _resolve_single_frame_index(*, native_count: int, ffs_count: int, frame_idx: int) -> tuple[int, int]:
    max_index = min(int(native_count), int(ffs_count)) - 1
    selected = int(frame_idx)
    if selected < 0 or selected > max_index:
        raise ValueError(
            f"frame_idx={selected} is out of range for native_count={native_count}, "
            f"ffs_count={ffs_count}. Expected 0 <= frame_idx <= {max_index}."
        )
    return selected, selected


def _resolve_mask_root(mask_root: str | Path) -> Path:
    root = Path(mask_root).resolve()
    if root.name == "mask":
        root = root.parent
    if not (root / "mask").is_dir():
        raise FileNotFoundError(f"Mask root does not contain a mask/ directory: {root}")
    return root


def _image_shape_for_camera_cloud(camera_cloud: dict[str, Any]) -> tuple[int, int]:
    image = cv2.imread(str(camera_cloud["color_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for mask load: {camera_cloud['color_path']}")
    return int(image.shape[0]), int(image.shape[1])


def _load_mask_info(mask_root: Path, *, camera_idx: int) -> dict[int, str]:
    info_path = mask_root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file():
        return {}
    data = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"mask_info file must contain a dict: {info_path}")
    return {int(obj_id): str(label) for obj_id, label in data.items()}


def load_union_masks_for_camera_clouds(
    *,
    mask_root: str | Path,
    camera_clouds: list[dict[str, Any]],
    frame_token: str,
    text_prompt: str,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    resolved_root = _resolve_mask_root(mask_root)
    prompts = set(parse_text_prompts(text_prompt))
    mask_by_camera: dict[int, np.ndarray] = {}
    debug_by_camera: dict[int, dict[str, Any]] = {}
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        serial = str(camera_cloud["serial"])
        height, width = _image_shape_for_camera_cloud(camera_cloud)
        union_mask = np.zeros((height, width), dtype=bool)
        mask_info = _load_mask_info(resolved_root, camera_idx=camera_idx)
        matched_object_ids = [
            int(obj_id)
            for obj_id, label in mask_info.items()
            if " ".join(str(label).strip().lower().split()) in prompts
        ]
        loaded_object_ids: list[int] = []
        missing_frame_mask_object_ids: list[int] = []
        for obj_id in matched_object_ids:
            mask_path = resolved_root / "mask" / str(camera_idx) / str(obj_id) / f"{frame_token}.png"
            if not mask_path.is_file():
                missing_frame_mask_object_ids.append(int(obj_id))
                continue
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise RuntimeError(f"Failed to load mask image: {mask_path}")
            union_mask |= mask_image > 0
            loaded_object_ids.append(int(obj_id))
        mask_by_camera[camera_idx] = union_mask
        debug_by_camera[camera_idx] = {
            "camera_idx": camera_idx,
            "serial": serial,
            "mask_root": str(resolved_root),
            "frame_token": str(frame_token),
            "matched_object_ids": matched_object_ids,
            "loaded_object_ids": loaded_object_ids,
            "missing_frame_mask_object_ids": missing_frame_mask_object_ids,
            "mask_pixel_count": int(np.count_nonzero(union_mask)),
        }
    return mask_by_camera, debug_by_camera


def _filter_aligned_array(values: Any, keep_mask: np.ndarray) -> Any:
    array = np.asarray(values)
    if array.ndim == 0 or len(array) != len(keep_mask):
        return values
    return array[keep_mask]


def filter_camera_clouds_with_pixel_masks(
    camera_clouds: list[dict[str, Any]],
    *,
    pixel_mask_by_camera: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered_clouds: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        pixel_mask = pixel_mask_by_camera.get(camera_idx)
        if pixel_mask is None:
            pixel_mask = np.zeros(_image_shape_for_camera_cloud(camera_cloud), dtype=bool)
        point_keep_mask = point_mask_from_pixel_mask(camera_cloud, pixel_mask=pixel_mask)
        filtered_cloud = dict(camera_cloud)
        filtered_cloud["points"] = np.asarray(camera_cloud["points"], dtype=np.float32)[point_keep_mask]
        filtered_cloud["colors"] = np.asarray(camera_cloud["colors"], dtype=np.uint8)[point_keep_mask]
        if "source_pixel_uv" in camera_cloud:
            filtered_cloud["source_pixel_uv"] = _filter_aligned_array(camera_cloud["source_pixel_uv"], point_keep_mask)
        if "source_depth_m" in camera_cloud:
            filtered_cloud["source_depth_m"] = _filter_aligned_array(camera_cloud["source_depth_m"], point_keep_mask)
        if "source_camera_idx" in camera_cloud:
            filtered_cloud["source_camera_idx"] = _filter_aligned_array(camera_cloud["source_camera_idx"], point_keep_mask)
        if "source_serial" in camera_cloud:
            filtered_cloud["source_serial"] = _filter_aligned_array(camera_cloud["source_serial"], point_keep_mask)
        filtered_clouds.append(filtered_cloud)
        metrics.append(
            {
                "camera_idx": camera_idx,
                "serial": str(camera_cloud["serial"]),
                "pre_mask_point_count": int(len(np.asarray(camera_cloud["points"]))),
                "post_mask_point_count": int(len(filtered_cloud["points"])),
                "mask_pixel_count": int(np.count_nonzero(pixel_mask)),
            }
        )
    return filtered_clouds, metrics


def _copy_pixel_masks_for_camera_clouds(
    camera_clouds: list[dict[str, Any]],
    *,
    pixel_mask_by_camera: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    copied: dict[int, np.ndarray] = {}
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        pixel_mask = pixel_mask_by_camera.get(camera_idx)
        if pixel_mask is None:
            pixel_mask = np.zeros(_image_shape_for_camera_cloud(camera_cloud), dtype=bool)
        copied[camera_idx] = np.asarray(pixel_mask, dtype=bool).copy()
    return copied


def _point_tuple_membership_mask(points: np.ndarray, *, point_set: set[tuple[float, float, float]]) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0 or not point_set:
        return np.zeros((len(cloud),), dtype=bool)
    return np.fromiter(
        (tuple(float(value) for value in point) in point_set for point in cloud),
        count=len(cloud),
        dtype=bool,
    )


def refine_pixel_masks_with_phystwin_data_process_mask(
    camera_clouds: list[dict[str, Any]],
    *,
    pixel_mask_by_camera: dict[int, np.ndarray],
    nb_points: int = int(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["nb_points"]),
    radius_m: float = float(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["radius_m"]),
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    refined_masks = _copy_pixel_masks_for_camera_clouds(
        camera_clouds,
        pixel_mask_by_camera=pixel_mask_by_camera,
    )
    masked_clouds_before, pre_metrics = filter_camera_clouds_with_pixel_masks(
        camera_clouds,
        pixel_mask_by_camera=refined_masks,
    )
    fused_points_before, _ = _fuse_camera_clouds(masked_clouds_before)
    outlier_result = detect_radius_outlier_indices(
        fused_points_before,
        radius_m=float(radius_m),
        nb_points=int(nb_points),
    )
    outlier_indices = np.asarray(outlier_result["outlier_indices"], dtype=np.int32).reshape(-1)
    outlier_point_set = {
        tuple(float(value) for value in point)
        for point in np.asarray(fused_points_before, dtype=np.float32)[outlier_indices]
    }

    per_camera_refine_metrics: list[dict[str, Any]] = []
    total_removed_pixels = 0
    for masked_camera_cloud, pre_metric in zip(masked_clouds_before, pre_metrics):
        camera_idx = int(masked_camera_cloud["camera_idx"])
        source_pixel_uv = masked_camera_cloud.get("source_pixel_uv")
        if source_pixel_uv is None:
            raise ValueError("PhysTwin-aligned mask refinement requires source_pixel_uv on camera clouds.")
        rejected_mask = _point_tuple_membership_mask(
            masked_camera_cloud["points"],
            point_set=outlier_point_set,
        )
        candidate_uv = np.asarray(source_pixel_uv, dtype=np.int32).reshape(-1, 2)[rejected_mask]
        inside = (
            (candidate_uv[:, 0] >= 0)
            & (candidate_uv[:, 0] < refined_masks[camera_idx].shape[1])
            & (candidate_uv[:, 1] >= 0)
            & (candidate_uv[:, 1] < refined_masks[camera_idx].shape[0])
        ) if len(candidate_uv) > 0 else np.zeros((0,), dtype=bool)
        candidate_uv = candidate_uv[inside]
        removed_pixels = 0
        removed_point_matches = int(np.count_nonzero(rejected_mask))
        if len(candidate_uv) > 0:
            unique_uv = np.unique(candidate_uv, axis=0)
            mask_before = int(np.count_nonzero(refined_masks[camera_idx]))
            refined_masks[camera_idx][unique_uv[:, 1], unique_uv[:, 0]] = False
            mask_after = int(np.count_nonzero(refined_masks[camera_idx]))
            removed_pixels = mask_before - mask_after
        total_removed_pixels += int(removed_pixels)
        per_camera_refine_metrics.append(
            {
                "camera_idx": camera_idx,
                "serial": str(masked_camera_cloud["serial"]),
                "pre_mask_pixel_count": int(pre_metric["mask_pixel_count"]),
                "pre_masked_point_count": int(pre_metric["post_mask_point_count"]),
                "outlier_point_matches": int(removed_point_matches),
                "removed_mask_pixel_count": int(removed_pixels),
            }
        )

    masked_clouds_after, post_metrics = filter_camera_clouds_with_pixel_masks(
        camera_clouds,
        pixel_mask_by_camera=refined_masks,
    )
    fused_points_after, _ = _fuse_camera_clouds(masked_clouds_after)
    post_metrics_by_camera = {int(item["camera_idx"]): item for item in post_metrics}
    for metric in per_camera_refine_metrics:
        camera_idx = int(metric["camera_idx"])
        post_metric = post_metrics_by_camera[camera_idx]
        metric["post_mask_pixel_count"] = int(post_metric["mask_pixel_count"])
        metric["post_masked_point_count"] = int(post_metric["post_mask_point_count"])

    summary = {
        **PHYSTWIN_DATA_PROCESS_MASK_CONTRACT,
        "fused_point_count_before": int(len(fused_points_before)),
        "fused_point_count_after": int(len(fused_points_after)),
        "fused_outlier_point_count": int(len(outlier_indices)),
        "removed_mask_pixel_count_total": int(total_removed_pixels),
        "per_camera": per_camera_refine_metrics,
    }
    return refined_masks, summary


def _fuse_camera_clouds(camera_clouds: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [np.asarray(item["points"], dtype=np.float32) for item in camera_clouds if len(item["points"]) > 0]
    color_sets = [np.asarray(item["colors"], dtype=np.uint8) for item in camera_clouds if len(item["points"]) > 0]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def _overlay_mask_on_rgb(color_path: str | Path, *, mask: np.ndarray, label: str) -> np.ndarray:
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for mask overlay: {color_path}")
    overlay = image.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 220
    alpha = np.zeros(mask.shape + (1,), dtype=np.float32)
    alpha[mask] = 0.60
    overlay = np.clip(
        overlay.astype(np.float32) * (1.0 - alpha) + green.astype(np.float32) * alpha,
        0.0,
        255.0,
    ).astype(np.uint8)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"mask pixels={int(np.count_nonzero(mask))}",
        (10, overlay.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (90, 235, 120),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _compute_focus_bounds(
    *,
    masked_native_points: np.ndarray,
    masked_ffs_points: np.ndarray,
    unmasked_native_points: np.ndarray,
    unmasked_ffs_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    masked_sets = [points for points in (masked_native_points, masked_ffs_points) if len(points) > 0]
    masked_count = int(sum(len(points) for points in masked_sets))
    if masked_sets and masked_count >= MIN_MASKED_POINT_COUNT_FOR_FOCUS:
        stacked = np.concatenate(masked_sets, axis=0)
        return (
            stacked.min(axis=0).astype(np.float32),
            stacked.max(axis=0).astype(np.float32),
            "masked_union",
            False,
        )

    unmasked_sets = [points for points in (unmasked_native_points, unmasked_ffs_points) if len(points) > 0]
    if not unmasked_sets:
        raise RuntimeError("Masked pointcloud compare could not load any fused points.")
    stacked = np.concatenate(unmasked_sets, axis=0)
    return (
        stacked.min(axis=0).astype(np.float32),
        stacked.max(axis=0).astype(np.float32),
        "unmasked_fallback",
        True,
    )


def _expand_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray, *, margin_ratio: float = 0.15) -> dict[str, np.ndarray]:
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    extents = np.maximum(bounds_max - bounds_min, 1e-6)
    margin = np.maximum(extents * float(margin_ratio), np.array([0.02, 0.02, 0.02], dtype=np.float32))
    return {
        "mode": "masked_object_bounds",
        "min": (bounds_min - margin).astype(np.float32),
        "max": (bounds_max + margin).astype(np.float32),
    }


def _render_variant(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    crop_bounds: dict[str, np.ndarray],
    center: np.ndarray,
    eye: np.ndarray,
    up: np.ndarray,
    zoom: float,
    point_size: float,
    render_width: int,
    render_height: int,
    render_frame_fn: Callable[..., np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cropped_points, cropped_colors = crop_points_to_bounds(points, colors, crop_bounds)
    image = render_frame_fn(
        cropped_points,
        cropped_colors,
        width=int(render_width),
        height=int(render_height),
        center=np.asarray(center, dtype=np.float32),
        eye=np.asarray(eye, dtype=np.float32),
        up=np.asarray(up, dtype=np.float32),
        zoom=float(zoom),
        point_size=float(point_size),
    )
    return image, cropped_points, cropped_colors


def compose_masked_pointcloud_board(
    *,
    title_lines: list[str],
    native_unmasked_image: np.ndarray,
    native_masked_image: np.ndarray,
    ffs_unmasked_image: np.ndarray,
    ffs_masked_image: np.ndarray,
) -> np.ndarray:
    panels = [
        overlay_large_panel_label(native_unmasked_image, label="Native | Unmasked", accent_bgr=(80, 180, 255)),
        overlay_large_panel_label(native_masked_image, label="Native | Masked", accent_bgr=(70, 220, 140)),
        overlay_large_panel_label(ffs_unmasked_image, label="FFS | Unmasked", accent_bgr=(255, 180, 80)),
        overlay_large_panel_label(ffs_masked_image, label="FFS | Masked", accent_bgr=(120, 220, 120)),
    ]
    target_size = (max(int(panel.shape[1]) for panel in panels), max(int(panel.shape[0]) for panel in panels))
    panels = [fit_image_to_canvas(panel, canvas_size=target_size, background_bgr=(16, 18, 22)) for panel in panels]
    top_row = np.hstack([panels[0], panels[1]])
    bottom_row = np.hstack([panels[2], panels[3]])
    body = np.vstack([top_row, bottom_row])
    title_h = 86
    title_bar = np.full((title_h, body.shape[1], 3), (12, 14, 18), dtype=np.uint8)
    for line_idx, line in enumerate(title_lines[:2]):
        y = 30 + line_idx * 28
        cv2.putText(
            title_bar,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 if line_idx == 0 else 0.58,
            (255, 255, 255),
            2 if line_idx == 0 else 1,
            cv2.LINE_AA,
        )
    return np.vstack([title_bar, body])


def run_masked_pointcloud_compare_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    native_mask_root: str | Path,
    ffs_mask_root: str | Path,
    native_mask_source: str,
    ffs_mask_source: str,
    mask_source_mode: str,
    text_prompt: str,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    camera_ids: list[int] | None = None,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = None,
    depth_min_m: float = DEFAULT_POINTCLOUD_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_POINTCLOUD_DEPTH_MAX_M,
    use_float_ffs_depth_when_available: bool = True,
    render_width: int = 960,
    render_height: int = 720,
    point_size: float = 2.0,
    zoom: float = 0.55,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=aligned_root,
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)
    if not _case_has_ffs_raw_depth(ffs_case_dir):
        raise ValueError(
            "Masked pointcloud compare requires an aligned FFS case containing depth_ffs/ or depth_ffs_float_m/."
        )
    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras.")

    native_frame_idx, ffs_frame_idx = _resolve_single_frame_index(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_idx=frame_idx,
    )
    selected_camera_ids = list(range(len(native_metadata["serial_numbers"]))) if camera_ids is None else [int(item) for item in camera_ids]
    if not selected_camera_ids:
        raise ValueError("camera_ids must not be empty.")
    max_camera_index = len(native_metadata["serial_numbers"]) - 1
    for camera_idx in selected_camera_ids:
        if camera_idx < 0 or camera_idx > max_camera_index:
            raise ValueError(f"camera_idx out of range: {camera_idx}")

    native_points, native_colors, native_stats, native_clouds = load_case_frame_cloud_with_sources(
        case_dir=native_case_dir,
        metadata=native_metadata,
        frame_idx=native_frame_idx,
        depth_source="realsense",
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    ffs_points, ffs_colors, ffs_stats, ffs_clouds = load_case_frame_cloud_with_sources(
        case_dir=ffs_case_dir,
        metadata=ffs_metadata,
        frame_idx=ffs_frame_idx,
        depth_source="ffs",
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )

    native_clouds = [cloud for cloud in native_clouds if int(cloud["camera_idx"]) in selected_camera_ids]
    ffs_clouds = [cloud for cloud in ffs_clouds if int(cloud["camera_idx"]) in selected_camera_ids]
    frame_token_native = str(native_frame_idx)
    frame_token_ffs = str(ffs_frame_idx)
    native_masks, native_mask_debug = load_union_masks_for_camera_clouds(
        mask_root=native_mask_root,
        camera_clouds=native_clouds,
        frame_token=frame_token_native,
        text_prompt=text_prompt,
    )
    ffs_masks, ffs_mask_debug = load_union_masks_for_camera_clouds(
        mask_root=ffs_mask_root,
        camera_clouds=ffs_clouds,
        frame_token=frame_token_ffs,
        text_prompt=text_prompt,
    )
    native_masked_clouds, native_mask_metrics = filter_camera_clouds_with_pixel_masks(
        native_clouds,
        pixel_mask_by_camera=native_masks,
    )
    ffs_masked_clouds, ffs_mask_metrics = filter_camera_clouds_with_pixel_masks(
        ffs_clouds,
        pixel_mask_by_camera=ffs_masks,
    )
    native_masked_points, native_masked_colors = _fuse_camera_clouds(native_masked_clouds)
    ffs_masked_points, ffs_masked_colors = _fuse_camera_clouds(ffs_masked_clouds)

    focus_bounds_min, focus_bounds_max, focus_source, fallback_used = _compute_focus_bounds(
        masked_native_points=native_masked_points,
        masked_ffs_points=ffs_masked_points,
        unmasked_native_points=native_points,
        unmasked_ffs_points=ffs_points,
    )
    crop_bounds = _expand_bounds(focus_bounds_min, focus_bounds_max)
    focus_point = ((focus_bounds_min + focus_bounds_max) * 0.5).astype(np.float32)
    view_config = compute_view_config(crop_bounds["min"], crop_bounds["max"], view_name="oblique")
    eye_offset = np.asarray(view_config["camera_position"], dtype=np.float32) - np.asarray(view_config["center"], dtype=np.float32)
    eye = focus_point + eye_offset
    up = np.asarray(view_config["up"], dtype=np.float32)
    render_frame_fn = render_frame_fn or _render_open3d_hidden_window

    native_unmasked_image, native_unmasked_cropped_points, native_unmasked_cropped_colors = _render_variant(
        points=native_points,
        colors=native_colors,
        crop_bounds=crop_bounds,
        center=focus_point,
        eye=eye,
        up=up,
        zoom=zoom,
        point_size=point_size,
        render_width=render_width,
        render_height=render_height,
        render_frame_fn=render_frame_fn,
    )
    native_masked_image, native_masked_cropped_points, native_masked_cropped_colors = _render_variant(
        points=native_masked_points,
        colors=native_masked_colors,
        crop_bounds=crop_bounds,
        center=focus_point,
        eye=eye,
        up=up,
        zoom=zoom,
        point_size=point_size,
        render_width=render_width,
        render_height=render_height,
        render_frame_fn=render_frame_fn,
    )
    ffs_unmasked_image, ffs_unmasked_cropped_points, ffs_unmasked_cropped_colors = _render_variant(
        points=ffs_points,
        colors=ffs_colors,
        crop_bounds=crop_bounds,
        center=focus_point,
        eye=eye,
        up=up,
        zoom=zoom,
        point_size=point_size,
        render_width=render_width,
        render_height=render_height,
        render_frame_fn=render_frame_fn,
    )
    ffs_masked_image, ffs_masked_cropped_points, ffs_masked_cropped_colors = _render_variant(
        points=ffs_masked_points,
        colors=ffs_masked_colors,
        crop_bounds=crop_bounds,
        center=focus_point,
        eye=eye,
        up=up,
        zoom=zoom,
        point_size=point_size,
        render_width=render_width,
        render_height=render_height,
        render_frame_fn=render_frame_fn,
    )

    board = compose_masked_pointcloud_board(
        title_lines=[
            "Masked Pointcloud Compare",
            f"frame={frame_idx:04d} prompt={text_prompt}",
        ],
        native_unmasked_image=native_unmasked_image,
        native_masked_image=native_masked_image,
        ffs_unmasked_image=ffs_unmasked_image,
        ffs_masked_image=ffs_masked_image,
    )
    board_path = output_dir / "01_masked_pointcloud_board.png"
    write_image(board_path, board)

    overlay_paths: dict[str, list[str]] = {"native": [], "ffs": []}
    for source_name, camera_clouds, pixel_masks in (
        ("native", native_clouds, native_masks),
        ("ffs", ffs_clouds, ffs_masks),
    ):
        for camera_cloud in camera_clouds:
            camera_idx = int(camera_cloud["camera_idx"])
            overlay = _overlay_mask_on_rgb(
                camera_cloud["color_path"],
                mask=pixel_masks[camera_idx],
                label=f"{source_name.title()} Cam{camera_idx} mask",
            )
            overlay_path = debug_dir / f"{source_name}_mask_overlay_cam{camera_idx}.png"
            write_image(overlay_path, overlay)
            overlay_paths[source_name].append(str(overlay_path.resolve()))

    ply_paths = {
        "native_unmasked": debug_dir / "native_unmasked_fused.ply",
        "native_masked": debug_dir / "native_masked_fused.ply",
        "ffs_unmasked": debug_dir / "ffs_unmasked_fused.ply",
        "ffs_masked": debug_dir / "ffs_masked_fused.ply",
    }
    write_ply_ascii(ply_paths["native_unmasked"], native_unmasked_cropped_points, native_unmasked_cropped_colors)
    write_ply_ascii(ply_paths["native_masked"], native_masked_cropped_points, native_masked_cropped_colors)
    write_ply_ascii(ply_paths["ffs_unmasked"], ffs_unmasked_cropped_points, ffs_unmasked_cropped_colors)
    write_ply_ascii(ply_paths["ffs_masked"], ffs_masked_cropped_points, ffs_masked_cropped_colors)

    native_metrics_by_camera = {int(item["camera_idx"]): item for item in native_mask_metrics}
    ffs_metrics_by_camera = {int(item["camera_idx"]): item for item in ffs_mask_metrics}
    summary = {
        "aligned_root": str(aligned_root),
        "output_dir": str(output_dir),
        "same_case_mode": bool(same_case_mode),
        "case_name": case_name,
        "native_case_name": str(native_case_dir.name),
        "ffs_case_name": str(ffs_case_dir.name),
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_idx": int(frame_idx),
        "native_frame_idx": int(native_frame_idx),
        "ffs_frame_idx": int(ffs_frame_idx),
        "camera_ids": [int(item) for item in selected_camera_ids],
        "text_prompt": str(text_prompt),
        "parsed_prompts": parse_text_prompts(text_prompt),
        "mask_source_mode": str(mask_source_mode),
        "mask_sources": {
            "native": {
                "mask_source": str(native_mask_source),
                "mask_root": str(_resolve_mask_root(native_mask_root)),
                "frame_token": frame_token_native,
                "per_camera": [
                    {
                        **native_mask_debug[int(cloud["camera_idx"])],
                        "pre_mask_point_count": int(native_metrics_by_camera[int(cloud["camera_idx"])]["pre_mask_point_count"]),
                        "post_mask_point_count": int(native_metrics_by_camera[int(cloud["camera_idx"])]["post_mask_point_count"]),
                    }
                    for cloud in native_clouds
                ],
            },
            "ffs": {
                "mask_source": str(ffs_mask_source),
                "mask_root": str(_resolve_mask_root(ffs_mask_root)),
                "frame_token": frame_token_ffs,
                "per_camera": [
                    {
                        **ffs_mask_debug[int(cloud["camera_idx"])],
                        "pre_mask_point_count": int(ffs_metrics_by_camera[int(cloud["camera_idx"])]["pre_mask_point_count"]),
                        "post_mask_point_count": int(ffs_metrics_by_camera[int(cloud["camera_idx"])]["post_mask_point_count"]),
                    }
                    for cloud in ffs_clouds
                ],
            },
        },
        "focus_source": focus_source,
        "empty_mask_fallback_used": bool(fallback_used),
        "shared_crop_bounds": {
            "mode": str(crop_bounds["mode"]),
            "min": [float(value) for value in crop_bounds["min"]],
            "max": [float(value) for value in crop_bounds["max"]],
        },
        "shared_view": {
            "view_name": str(view_config["view_name"]),
            "center": [float(value) for value in focus_point],
            "eye": [float(value) for value in eye],
            "up": [float(value) for value in up],
            "zoom": float(zoom),
            "point_size": float(point_size),
            "image_size": [int(render_width), int(render_height)],
        },
        "render_contract": dict(MASKED_POINTCLOUD_RENDER_CONTRACT),
        "board_path": str(board_path.resolve()),
        "variants": {
            "native_unmasked": {
                "fused_point_count": int(len(native_unmasked_cropped_points)),
                "ply_path": str(ply_paths["native_unmasked"].resolve()),
            },
            "native_masked": {
                "fused_point_count": int(len(native_masked_cropped_points)),
                "ply_path": str(ply_paths["native_masked"].resolve()),
            },
            "ffs_unmasked": {
                "fused_point_count": int(len(ffs_unmasked_cropped_points)),
                "ply_path": str(ply_paths["ffs_unmasked"].resolve()),
            },
            "ffs_masked": {
                "fused_point_count": int(len(ffs_masked_cropped_points)),
                "ply_path": str(ply_paths["ffs_masked"].resolve()),
            },
        },
        "debug_artifacts": {
            "native_mask_overlay_paths": overlay_paths["native"],
            "ffs_mask_overlay_paths": overlay_paths["ffs"],
        },
        "source_stats": {
            "native": native_stats,
            "ffs": ffs_stats,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
