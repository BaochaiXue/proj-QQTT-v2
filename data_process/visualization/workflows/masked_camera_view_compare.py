from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..io_artifacts import write_image, write_json, write_ply_ascii
from ..io_case import (
    get_frame_count,
    load_case_frame_cloud_with_sources,
    load_case_metadata,
    resolve_case_dirs,
)
from ..layouts import compose_grid_2x3, compose_single_row_board
from ..pointcloud_defaults import DEFAULT_POINTCLOUD_DEPTH_MAX_M, DEFAULT_POINTCLOUD_DEPTH_MIN_M
from ..roi import crop_points_to_bounds
from ..triplet_ply_compare import _case_has_ffs_raw_depth
from ..triplet_video_compare import _render_open3d_hidden_window
from ..views import build_original_camera_view_configs
from .masked_pointcloud_compare import (
    MIN_MASKED_POINT_COUNT_FOR_FOCUS,
    _compute_focus_bounds,
    _expand_bounds,
    _overlay_mask_on_rgb,
    _resolve_mask_root,
    filter_camera_clouds_with_pixel_masks,
    load_union_masks_for_camera_clouds,
    parse_text_prompts,
)


MASKED_CAMERA_VIEW_RENDER_CONTRACT = {
    "renderer": "open3d_hidden_visualizer",
    "render_mode": "color_by_rgb",
    "view_mode": "original_camera_extrinsics",
    "projection_mode": "original_camera_pinhole",
    "shared_crop_across_panels": True,
    "shared_per_column_view_between_rows": True,
    "masked_rgb_reference_panel": True,
    "supports_native_depth_postprocess": True,
    "supports_ffs_native_like_postprocess": True,
}


def _resolve_single_frame_index(*, native_count: int, ffs_count: int, frame_idx: int) -> tuple[int, int]:
    max_index = min(int(native_count), int(ffs_count)) - 1
    selected = int(frame_idx)
    if selected < 0 or selected > max_index:
        raise ValueError(
            f"frame_idx={selected} is out of range for native_count={native_count}, "
            f"ffs_count={ffs_count}. Expected 0 <= frame_idx <= {max_index}."
        )
    return selected, selected


def _fuse_camera_clouds(camera_clouds: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [np.asarray(item["points"], dtype=np.float32) for item in camera_clouds if len(item["points"]) > 0]
    color_sets = [np.asarray(item["colors"], dtype=np.uint8) for item in camera_clouds if len(item["points"]) > 0]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


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


def _image_size_from_color_path(color_path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for masked camera-view compare: {color_path}")
    return int(image.shape[1]), int(image.shape[0])


def _scale_intrinsic_matrix(
    intrinsic_matrix: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_w, source_h = [max(1, int(value)) for value in source_size]
    target_w, target_h = [max(1, int(value)) for value in target_size]
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    scaled = np.asarray(intrinsic_matrix, dtype=np.float32).reshape(3, 3).copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def _mask_rgb_image(color_path: str | Path, *, mask: np.ndarray) -> np.ndarray:
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for masked RGB board: {color_path}")
    masked = np.zeros_like(image)
    masked[np.asarray(mask, dtype=bool)] = image[np.asarray(mask, dtype=bool)]
    return masked


def run_masked_camera_view_compare_workflow(
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
    native_depth_postprocess: bool = False,
    ffs_native_like_postprocess: bool = False,
    render_width: int | None = None,
    render_height: int | None = None,
    point_size: float = 2.0,
    zoom: float = 0.55,
    look_distance: float = 1.0,
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
            "Masked camera-view compare requires an aligned FFS case containing depth_ffs/ or depth_ffs_float_m/."
        )
    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras.")

    native_frame_idx, ffs_frame_idx = _resolve_single_frame_index(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_idx=frame_idx,
    )
    selected_camera_ids = list(range(len(native_metadata["serial_numbers"]))) if camera_ids is None else [int(item) for item in camera_ids]
    if len(selected_camera_ids) != 3:
        raise ValueError(f"Masked camera-view compare requires exactly 3 camera_ids. Got: {selected_camera_ids}")
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
        native_depth_postprocess=bool(native_depth_postprocess),
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
        native_depth_postprocess=False,
        ffs_native_like_postprocess=bool(ffs_native_like_postprocess),
    )

    native_clouds = [cloud for cloud in native_clouds if int(cloud["camera_idx"]) in selected_camera_ids]
    ffs_clouds = [cloud for cloud in ffs_clouds if int(cloud["camera_idx"]) in selected_camera_ids]
    native_points, native_colors = _fuse_camera_clouds(native_clouds)
    ffs_points, ffs_colors = _fuse_camera_clouds(ffs_clouds)

    native_masks, native_mask_debug = load_union_masks_for_camera_clouds(
        mask_root=native_mask_root,
        camera_clouds=native_clouds,
        frame_token=str(native_frame_idx),
        text_prompt=text_prompt,
    )
    ffs_masks, ffs_mask_debug = load_union_masks_for_camera_clouds(
        mask_root=ffs_mask_root,
        camera_clouds=ffs_clouds,
        frame_token=str(ffs_frame_idx),
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
    render_frame_fn = render_frame_fn or _render_open3d_hidden_window

    native_camera_cloud_map = {int(cloud["camera_idx"]): cloud for cloud in native_clouds}
    native_c2w_list = [np.asarray(native_camera_cloud_map[camera_idx]["c2w"], dtype=np.float32) for camera_idx in selected_camera_ids]
    serial_numbers = [str(native_camera_cloud_map[camera_idx]["serial"]) for camera_idx in selected_camera_ids]
    view_configs = build_original_camera_view_configs(
        c2w_list=native_c2w_list,
        serial_numbers=serial_numbers,
        look_distance=float(look_distance),
        camera_ids=list(range(len(selected_camera_ids))),
    )
    # Restore actual camera ids in labels/payload after building on the selected-order list.
    for idx, view_config in enumerate(view_configs):
        actual_camera_idx = int(selected_camera_ids[idx])
        camera_cloud = native_camera_cloud_map[actual_camera_idx]
        source_image_size = _image_size_from_color_path(camera_cloud["color_path"])
        target_image_size = (
            int(render_width) if render_width is not None else int(source_image_size[0]),
            int(render_height) if render_height is not None else int(source_image_size[1]),
        )
        view_config["camera_idx"] = actual_camera_idx
        view_config["view_name"] = f"cam{actual_camera_idx}"
        view_config["label"] = f"Cam{actual_camera_idx} | {serial_numbers[idx]}"
        view_config["intrinsic_matrix"] = _scale_intrinsic_matrix(
            np.asarray(camera_cloud["K_color"], dtype=np.float32),
            source_size=source_image_size,
            target_size=target_image_size,
        )
        view_config["extrinsic_matrix"] = np.linalg.inv(np.asarray(camera_cloud["c2w"], dtype=np.float32).reshape(4, 4)).astype(np.float32)
        view_config["image_size"] = [int(target_image_size[0]), int(target_image_size[1])]

    rgb_images: list[np.ndarray] = []
    native_images: list[np.ndarray] = []
    ffs_images: list[np.ndarray] = []
    rgb_render_paths: list[str] = []
    native_render_paths: list[str] = []
    ffs_render_paths: list[str] = []
    for view_config in view_configs:
        camera_idx = int(view_config["camera_idx"])
        target_w, target_h = [int(item) for item in view_config["image_size"]]
        rgb_image = _mask_rgb_image(
            native_camera_cloud_map[camera_idx]["color_path"],
            mask=native_masks[camera_idx],
        )
        native_image = render_frame_fn(
            *crop_points_to_bounds(native_masked_points, native_masked_colors, crop_bounds),
            width=int(target_w),
            height=int(target_h),
            center=np.asarray(view_config["center"], dtype=np.float32),
            eye=np.asarray(view_config["camera_position"], dtype=np.float32),
            up=np.asarray(view_config["up"], dtype=np.float32),
            zoom=float(zoom),
            point_size=float(point_size),
            intrinsic_matrix=np.asarray(view_config["intrinsic_matrix"], dtype=np.float32),
            extrinsic_matrix=np.asarray(view_config["extrinsic_matrix"], dtype=np.float32),
        )
        ffs_image = render_frame_fn(
            *crop_points_to_bounds(ffs_masked_points, ffs_masked_colors, crop_bounds),
            width=int(target_w),
            height=int(target_h),
            center=np.asarray(view_config["center"], dtype=np.float32),
            eye=np.asarray(view_config["camera_position"], dtype=np.float32),
            up=np.asarray(view_config["up"], dtype=np.float32),
            zoom=float(zoom),
            point_size=float(point_size),
            intrinsic_matrix=np.asarray(view_config["intrinsic_matrix"], dtype=np.float32),
            extrinsic_matrix=np.asarray(view_config["extrinsic_matrix"], dtype=np.float32),
        )
        rgb_images.append(rgb_image)
        native_images.append(native_image)
        ffs_images.append(ffs_image)

        rgb_render_path = debug_dir / f"masked_rgb_cam{int(view_config['camera_idx'])}.png"
        native_render_path = debug_dir / f"native_cam{int(view_config['camera_idx'])}.png"
        ffs_render_path = debug_dir / f"ffs_cam{int(view_config['camera_idx'])}.png"
        write_image(rgb_render_path, rgb_image)
        write_image(native_render_path, native_image)
        write_image(ffs_render_path, ffs_image)
        rgb_render_paths.append(str(rgb_render_path.resolve()))
        native_render_paths.append(str(native_render_path.resolve()))
        ffs_render_paths.append(str(ffs_render_path.resolve()))

    rgb_board = compose_single_row_board(
        title_lines=[
            "Masked RGB Reference",
            f"frame={int(frame_idx):04d}  prompt={text_prompt}",
        ],
        column_headers=[str(view["label"]) for view in view_configs],
        images=rgb_images,
    )
    rgb_board_path = output_dir / "00_masked_rgb_board.png"
    write_image(rgb_board_path, rgb_board)

    board = compose_grid_2x3(
        title=f"Masked Camera-View PCD Compare | frame={int(frame_idx):04d} | prompt={text_prompt}",
        column_headers=[str(view["label"]) for view in view_configs],
        row_headers=["Native", "FFS"],
        native_images=native_images,
        ffs_images=ffs_images,
    )
    board_path = output_dir / "01_masked_camera_view_board.png"
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
        "native_masked": debug_dir / "native_masked_fused.ply",
        "ffs_masked": debug_dir / "ffs_masked_fused.ply",
    }
    write_ply_ascii(*((ply_paths["native_masked"],) + crop_points_to_bounds(native_masked_points, native_masked_colors, crop_bounds)))
    write_ply_ascii(*((ply_paths["ffs_masked"],) + crop_points_to_bounds(ffs_masked_points, ffs_masked_colors, crop_bounds)))

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
        "native_depth_postprocess": bool(native_depth_postprocess),
        "ffs_native_like_postprocess": bool(ffs_native_like_postprocess),
        "mask_sources": {
            "native": {
                "mask_source": str(native_mask_source),
                "mask_root": str(_resolve_mask_root(native_mask_root)),
                "frame_token": str(native_frame_idx),
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
                "frame_token": str(ffs_frame_idx),
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
        "column_views": [_serialize_view_config(view_config) for view_config in view_configs],
        "render_contract": dict(MASKED_CAMERA_VIEW_RENDER_CONTRACT),
        "rgb_board_path": str(rgb_board_path.resolve()),
        "board_path": str(board_path.resolve()),
        "variants": {
            "masked_rgb_reference": {
                "panel_count": int(len(rgb_images)),
            },
            "native_masked": {
                "fused_point_count": int(len(crop_points_to_bounds(native_masked_points, native_masked_colors, crop_bounds)[0])),
                "ply_path": str(ply_paths["native_masked"].resolve()),
            },
            "ffs_masked": {
                "fused_point_count": int(len(crop_points_to_bounds(ffs_masked_points, ffs_masked_colors, crop_bounds)[0])),
                "ply_path": str(ply_paths["ffs_masked"].resolve()),
            },
        },
        "debug_artifacts": {
            "masked_rgb_paths": rgb_render_paths,
            "native_mask_overlay_paths": overlay_paths["native"],
            "ffs_mask_overlay_paths": overlay_paths["ffs"],
            "native_render_paths": native_render_paths,
            "ffs_render_paths": ffs_render_paths,
        },
        "source_stats": {
            "native": native_stats,
            "ffs": ffs_stats,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
