from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..calibration_io import load_calibration_transforms
from ..depth_colormap import colorize_depth_meters
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_metadata, resolve_case_dir
from ..layouts import compose_registration_matrix_board
from ..workflows.masked_pointcloud_compare import load_union_masks_for_camera_clouds
from .enhanced_phystwin_postprocess_pcd_compare import (
    DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
)
from .enhanced_phystwin_removed_overlay import (
    DEFAULT_SOURCE_CAMERA_HIGHLIGHT_LABELS,
    _build_ir_tile,
    _build_masked_depth_tile,
    _build_rgb_object_mask_tile,
    _load_ir_image,
    _overlay_mask,
    _pixel_mask_from_uv,
    _source_camera_highlight_color_bgr,
)
from .ffs_confidence_filter_pcd_compare import (
    _apply_enhanced_phystwin_like_postprocess_with_trace,
    _build_world_cloud,
    _format_point_count,
    _load_color_image,
)
from .ffs_confidence_panels import DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT
from .native_ffs_fused_pcd_compare import DEFAULT_PHYSTWIN_NB_POINTS, DEFAULT_PHYSTWIN_RADIUS_M, _load_depth_m


DEFAULT_OUTPUT_ROOT_NAME = "still_object_round1_projection_panel_13x3_ffs203048_iter4_trt_level5"
DEFAULT_TILE_WIDTH = 480
DEFAULT_TILE_HEIGHT = 360
DEFAULT_ROW_LABEL_WIDTH = 390
ROW_HEADERS_13 = [
    "RGB + mask",
    "IR left",
    "IR right",
    "RS depth mask",
    "RS PCD -> RGB",
    "RS PCD kept",
    "RS PCD + removed",
    "RS RGB removed",
    "FFS depth mask",
    "FFS PCD -> RGB",
    "FFS PCD kept",
    "FFS PCD + removed",
    "FFS RGB removed",
]


def _soft_white_background(color_image: np.ndarray) -> np.ndarray:
    image = np.asarray(color_image, dtype=np.uint8)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=5.0, sigmaY=5.0)
    white = np.full_like(blurred, 248, dtype=np.uint8)
    return cv2.addWeighted(blurred, 0.34, white, 0.66, 0.0)


def _project_world_points(
    points_world: np.ndarray,
    *,
    K_color: np.ndarray,
    c2w: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        empty_i = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_i

    height, width = [int(item) for item in image_shape[:2]]
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    points_cam = (w2c @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    valid_z = np.isfinite(z) & (z > 1e-5)
    u = np.zeros((len(points),), dtype=np.float32)
    v = np.zeros((len(points),), dtype=np.float32)
    u[valid_z] = (K[0, 0] * points_cam[valid_z, 0] / z[valid_z]) + K[0, 2]
    v[valid_z] = (K[1, 1] * points_cam[valid_z, 1] / z[valid_z]) + K[1, 2]
    x = np.rint(u).astype(np.int32)
    y = np.rint(v).astype(np.int32)
    valid = valid_z & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    indices = np.where(valid)[0].astype(np.int32)
    return x[indices], y[indices], z[indices].astype(np.float32), indices


def _nearest_pixel_indices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    width: int,
) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0,), dtype=np.int32)
    linear = np.asarray(y, dtype=np.int64) * int(width) + np.asarray(x, dtype=np.int64)
    order = np.lexsort((np.asarray(z, dtype=np.float32), linear))
    sorted_linear = linear[order]
    first = np.unique(sorted_linear, return_index=True)[1]
    return order[first].astype(np.int32)


def _draw_projected_points(
    canvas: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    colors_bgr: np.ndarray,
    radius_px: int,
) -> None:
    radius = int(radius_px)
    colors = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 3)
    for px, py, color in zip(np.asarray(x, dtype=np.int32), np.asarray(y, dtype=np.int32), colors):
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        if radius <= 0:
            canvas[int(py), int(px)] = color
        else:
            cv2.circle(canvas, (int(px), int(py)), radius, color_tuple, -1, cv2.LINE_AA)


def build_projected_pcd_rgb_tile(
    *,
    color_image: np.ndarray,
    points_world: np.ndarray,
    colors_bgr: np.ndarray,
    source_camera_idx: np.ndarray,
    K_color: np.ndarray,
    c2w: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
    keep_mask: np.ndarray | None = None,
    removed_mask: np.ndarray | None = None,
    point_radius_px: int = 1,
    removed_radius_px: int = 2,
) -> np.ndarray:
    image = np.asarray(color_image, dtype=np.uint8)
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 3)
    sources = np.asarray(source_camera_idx, dtype=np.int32).reshape(-1)
    if keep_mask is None:
        selected = np.ones((len(points),), dtype=bool)
    else:
        selected = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if selected.shape[0] != len(points):
        raise ValueError("keep_mask must match point count.")
    removed = np.zeros((len(points),), dtype=bool) if removed_mask is None else np.asarray(removed_mask, dtype=bool).reshape(-1)
    if removed.shape[0] != len(points):
        raise ValueError("removed_mask must match point count.")

    canvas = _soft_white_background(image)
    x, y, z, valid_indices = _project_world_points(
        points[selected],
        K_color=K_color,
        c2w=c2w,
        image_shape=image.shape[:2],
    )
    if len(valid_indices) > 0:
        selected_indices = np.where(selected)[0][valid_indices]
        nearest = _nearest_pixel_indices(x, y, z, width=image.shape[1])
        draw_indices = selected_indices[nearest]
        _draw_projected_points(
            canvas,
            x=x[nearest],
            y=y[nearest],
            colors_bgr=colors[draw_indices],
            radius_px=int(point_radius_px),
        )

    removed_selected = selected & removed
    x_r, y_r, _z_r, valid_removed_indices = _project_world_points(
        points[removed_selected],
        K_color=K_color,
        c2w=c2w,
        image_shape=image.shape[:2],
    )
    if len(valid_removed_indices) > 0:
        removed_indices = np.where(removed_selected)[0][valid_removed_indices]
        removed_colors = np.asarray([_source_camera_highlight_color_bgr(int(idx)) for idx in sources[removed_indices]], dtype=np.uint8)
        _draw_projected_points(
            canvas,
            x=x_r,
            y=y_r,
            colors_bgr=removed_colors,
            radius_px=int(removed_radius_px),
        )

    return label_tile(canvas, label, tile_size)


def _build_rgb_removed_source_tile(
    *,
    color_image: np.ndarray,
    source_pixel_uv: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
    source_camera_idx: int,
    radius_px: int,
    alpha: float,
) -> np.ndarray:
    base = _soft_white_background(color_image)
    removed_pixel_mask = _pixel_mask_from_uv(
        source_pixel_uv,
        image_shape=color_image.shape[:2],
        radius_px=int(radius_px),
    )
    overlay = _overlay_mask(
        base,
        removed_pixel_mask,
        color_bgr=_source_camera_highlight_color_bgr(int(source_camera_idx)),
        alpha=float(alpha),
    )
    return label_tile(overlay, label, tile_size)


def build_still_object_projection_board(
    *,
    round_label: str,
    frame_idx: int,
    model_config: dict[str, Any],
    column_headers: list[str],
    image_rows: list[list[np.ndarray]],
) -> np.ndarray:
    if len(image_rows) != 13 or any(len(row) != 3 for row in image_rows):
        raise ValueError("Still object projection panel requires a 13x3 image matrix.")
    return compose_registration_matrix_board(
        title_lines=[
            f"Still Object 13-row Projection Panel | {round_label} | frame {int(frame_idx)}",
            (
                "rows=RGB/mask + IR + native depth/projection/removal + FFS depth/projection/removal | "
                f"FFS={model_config['ffs_model_name']} iter{int(model_config['ffs_valid_iters'])} TRT level{int(model_config['trt_builder_optimization_level'])}"
            ),
        ],
        row_headers=list(ROW_HEADERS_13),
        column_headers=column_headers,
        image_rows=image_rows,
        row_label_width=int(model_config.get("row_label_width", DEFAULT_ROW_LABEL_WIDTH)),
    )


def _format_result_summary_text(summary: dict[str, Any]) -> str:
    counts = summary["total_counts"]
    model_config = summary["model_config"]
    lines = [
        "Still Object Round 1 13-row projection panel",
        "",
        "Setting:",
        f"  FFS model: {model_config['ffs_model_name']}",
        f"  FFS valid_iters: {model_config['ffs_valid_iters']}",
        f"  TRT builder optimization level: {model_config['trt_builder_optimization_level']}",
        f"  depth range m: {model_config['depth_min_m']} - {model_config['depth_max_m']}",
        f"  PhySTwin-like radius m: {model_config['phystwin_radius_m']}",
        f"  PhySTwin-like nb_points: {model_config['phystwin_nb_points']}",
        "",
        "Total masked point counts:",
        f"  RealSense masked: {counts['native_masked_point_count']}",
        f"  RealSense kept: {counts['native_kept_point_count']}",
        f"  RealSense removed: {counts['native_removed_point_count']}",
        f"  FFS masked: {counts['ffs_masked_point_count']}",
        f"  FFS kept: {counts['ffs_kept_point_count']}",
        f"  FFS removed: {counts['ffs_removed_point_count']}",
        "",
        "Removed points by source camera/view:",
    ]
    for item in summary["per_camera"]:
        lines.extend(
            [
                f"  Cam{item['camera_idx']} | {item['serial']}:",
                f"    mask pixels: {item['mask_pixel_count']}",
                f"    RealSense masked points: {item['native_masked_point_count']}",
                f"    RealSense removed source points: {item['native_removed_source_point_count']}",
                f"    FFS masked points: {item['ffs_masked_point_count']}",
                f"    FFS removed source points: {item['ffs_removed_source_point_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"  panel: {summary['board_path']}",
            f"  aligned case: {summary['aligned_case_dir']}",
            f"  mask root: {summary['mask_root']}",
            "",
        ]
    )
    return "\n".join(lines)


def _concat_or_empty(arrays: list[np.ndarray], *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    usable = [np.asarray(item, dtype=dtype) for item in arrays if len(item) > 0]
    if not usable:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(usable, axis=0)


def _build_depth_payload(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    c2w_list: list[np.ndarray],
    masks_by_camera: dict[int, np.ndarray],
    camera_ids: list[int],
    frame_idx: int,
    depth_source: str,
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_camera: int | None,
    use_float_ffs_depth_when_available: bool,
) -> dict[str, Any]:
    per_camera: dict[int, dict[str, Any]] = {}
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    source_camera_sets: list[np.ndarray] = []
    source_uv_sets: list[np.ndarray] = []
    for camera_idx in camera_ids:
        color_path = case_dir / "color" / str(camera_idx) / f"{int(frame_idx)}.png"
        color_image = _load_color_image(color_path)
        depth_m, depth_info = _load_depth_m(
            case_dir=case_dir,
            metadata=metadata,
            camera_idx=int(camera_idx),
            frame_idx=int(frame_idx),
            depth_source=depth_source,
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        )
        object_mask = np.asarray(masks_by_camera[int(camera_idx)], dtype=bool)
        cloud = _build_world_cloud(
            depth_m=depth_m,
            color_image=color_image,
            K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
            c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
            depth_min_m=float(depth_min_m),
            depth_max_m=float(depth_max_m),
            max_points_per_camera=max_points_per_camera,
            object_mask=object_mask,
        )
        points = np.asarray(cloud["points"], dtype=np.float32)
        point_sets.append(points)
        color_sets.append(np.asarray(cloud["colors"], dtype=np.uint8))
        source_camera_sets.append(np.full((len(points),), int(camera_idx), dtype=np.int32))
        source_uv_sets.append(np.asarray(cloud["source_pixel_uv"], dtype=np.int32).reshape(-1, 2))
        per_camera[int(camera_idx)] = {
            "camera_idx": int(camera_idx),
            "color_path": str(color_path.resolve()),
            "color_image": color_image,
            "depth_m": np.asarray(depth_m, dtype=np.float32),
            "depth_info": depth_info,
            "object_mask": object_mask,
            "masked_point_count": int(len(points)),
            "mask_pixel_count": int(np.count_nonzero(object_mask)),
        }

    points = _concat_or_empty(point_sets, shape=(0, 3), dtype=np.float32)
    colors = _concat_or_empty(color_sets, shape=(0, 3), dtype=np.uint8)
    source_camera_idx = _concat_or_empty(source_camera_sets, shape=(0,), dtype=np.int32)
    source_pixel_uv = _concat_or_empty(source_uv_sets, shape=(0, 2), dtype=np.int32)
    return {
        "per_camera": per_camera,
        "points": points,
        "colors": colors,
        "source_camera_idx": source_camera_idx,
        "source_pixel_uv": source_pixel_uv,
    }


def _apply_enhanced_filter(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    return _apply_enhanced_phystwin_like_postprocess_with_trace(
        points=points,
        colors=colors,
        enabled=True,
        radius_m=float(phystwin_radius_m),
        nb_points=int(phystwin_nb_points),
        component_voxel_size_m=float(enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )


def run_still_object_projection_panel_workflow(
    *,
    aligned_root: Path,
    case_ref: str,
    mask_root: Path,
    output_root: Path,
    frame_idx: int = 0,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    row_label_width: int = DEFAULT_ROW_LABEL_WIDTH,
    max_points_per_camera: int | None = None,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
    highlight_alpha: float = 0.70,
    highlight_radius_px: int = 2,
    projection_point_radius_px: int = 1,
    use_float_ffs_depth_when_available: bool = True,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(case_ref))
    metadata = load_case_metadata(case_dir)
    camera_ids = list(range(len(metadata["serial_numbers"])))
    if len(camera_ids) != 3:
        raise ValueError(f"Expected exactly 3 cameras, got {camera_ids}.")
    if int(frame_idx) < 0 or int(frame_idx) >= int(metadata["frame_num"]):
        raise ValueError(f"frame_idx={frame_idx} is outside frame_num={metadata['frame_num']}.")
    if float(depth_max_m) <= float(depth_min_m):
        raise ValueError("depth_max_m must be greater than depth_min_m.")

    c2w_list = load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=metadata["serial_numbers"],
        calibration_reference_serials=metadata.get("calibration_reference_serials", metadata["serial_numbers"]),
    )
    minimal_clouds = [
        {
            "camera_idx": int(camera_idx),
            "serial": str(metadata["serial_numbers"][camera_idx]),
            "color_path": str(case_dir / "color" / str(camera_idx) / f"{int(frame_idx)}.png"),
        }
        for camera_idx in camera_ids
    ]
    masks_by_camera, mask_debug = load_union_masks_for_camera_clouds(
        mask_root=Path(mask_root).resolve(),
        camera_clouds=minimal_clouds,
        frame_token=str(int(frame_idx)),
        text_prompt=str(text_prompt),
    )

    native_payload = _build_depth_payload(
        case_dir=case_dir,
        metadata=metadata,
        c2w_list=c2w_list,
        masks_by_camera=masks_by_camera,
        camera_ids=camera_ids,
        frame_idx=int(frame_idx),
        depth_source="realsense",
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=False,
    )
    ffs_payload = _build_depth_payload(
        case_dir=case_dir,
        metadata=metadata,
        c2w_list=c2w_list,
        masks_by_camera=masks_by_camera,
        camera_ids=camera_ids,
        frame_idx=int(frame_idx),
        depth_source="ffs_raw",
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
    )
    native_filtered_points, native_filtered_colors, native_filter_stats, native_trace = _apply_enhanced_filter(
        points=native_payload["points"],
        colors=native_payload["colors"],
        phystwin_radius_m=float(phystwin_radius_m),
        phystwin_nb_points=int(phystwin_nb_points),
        enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
        enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )
    ffs_filtered_points, ffs_filtered_colors, ffs_filter_stats, ffs_trace = _apply_enhanced_filter(
        points=ffs_payload["points"],
        colors=ffs_payload["colors"],
        phystwin_radius_m=float(phystwin_radius_m),
        phystwin_nb_points=int(phystwin_nb_points),
        enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
        enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )
    native_kept_mask = np.asarray(native_trace["kept_mask"], dtype=bool)
    ffs_kept_mask = np.asarray(ffs_trace["kept_mask"], dtype=bool)
    native_removed_mask = np.asarray(native_trace["removed_mask"], dtype=bool)
    ffs_removed_mask = np.asarray(ffs_trace["removed_mask"], dtype=bool)

    tile_size = (int(tile_width), int(tile_height))
    image_rows: list[list[np.ndarray]] = [[] for _ in range(13)]
    per_camera_summary: list[dict[str, Any]] = []
    for camera_idx in camera_ids:
        color_image = native_payload["per_camera"][camera_idx]["color_image"]
        object_mask = np.asarray(masks_by_camera[int(camera_idx)], dtype=bool)
        column_label = f"Cam{camera_idx} | {metadata['serial_numbers'][camera_idx]}"
        image_rows[0].append(
            _build_rgb_object_mask_tile(
                color_image=color_image,
                object_mask=object_mask,
                label=f"mask={int(np.count_nonzero(object_mask))} px",
                tile_size=tile_size,
            )
        )
        image_rows[1].append(
            _build_ir_tile(
                ir_image=_load_ir_image(case_dir / "ir_left" / str(camera_idx) / f"{int(frame_idx)}.png"),
                label="IR left",
                tile_size=tile_size,
            )
        )
        image_rows[2].append(
            _build_ir_tile(
                ir_image=_load_ir_image(case_dir / "ir_right" / str(camera_idx) / f"{int(frame_idx)}.png"),
                label="IR right",
                tile_size=tile_size,
            )
        )
        image_rows[3].append(
            _build_masked_depth_tile(
                depth_m=native_payload["per_camera"][camera_idx]["depth_m"],
                object_mask=object_mask,
                label=f"RS depth valid={native_payload['per_camera'][camera_idx]['masked_point_count']}",
                tile_size=tile_size,
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
            )
        )
        image_rows[4].append(
            build_projected_pcd_rgb_tile(
                color_image=color_image,
                points_world=native_payload["points"],
                colors_bgr=native_payload["colors"],
                source_camera_idx=native_payload["source_camera_idx"],
                K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
                label=f"RS PCD {_format_point_count(len(native_payload['points']))}",
                tile_size=tile_size,
                point_radius_px=int(projection_point_radius_px),
            )
        )
        image_rows[5].append(
            build_projected_pcd_rgb_tile(
                color_image=color_image,
                points_world=native_filtered_points,
                colors_bgr=native_filtered_colors,
                source_camera_idx=native_payload["source_camera_idx"][native_kept_mask],
                K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
                label=f"RS kept {_format_point_count(len(native_filtered_points))}",
                tile_size=tile_size,
                point_radius_px=int(projection_point_radius_px),
            )
        )
        image_rows[6].append(
            build_projected_pcd_rgb_tile(
                color_image=color_image,
                points_world=native_payload["points"],
                colors_bgr=native_payload["colors"],
                source_camera_idx=native_payload["source_camera_idx"],
                K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
                label=f"RS removed={int(np.count_nonzero(native_removed_mask))}",
                tile_size=tile_size,
                removed_mask=native_removed_mask,
                point_radius_px=int(projection_point_radius_px),
                removed_radius_px=int(highlight_radius_px),
            )
        )
        native_source_removed = native_removed_mask & (native_payload["source_camera_idx"] == int(camera_idx))
        image_rows[7].append(
            _build_rgb_removed_source_tile(
                color_image=color_image,
                source_pixel_uv=native_payload["source_pixel_uv"][native_source_removed],
                label=f"RS RGB removed={int(np.count_nonzero(native_source_removed))}",
                tile_size=tile_size,
                source_camera_idx=int(camera_idx),
                radius_px=int(highlight_radius_px),
                alpha=float(highlight_alpha),
            )
        )
        image_rows[8].append(
            _build_masked_depth_tile(
                depth_m=ffs_payload["per_camera"][camera_idx]["depth_m"],
                object_mask=object_mask,
                label=f"FFS depth valid={ffs_payload['per_camera'][camera_idx]['masked_point_count']}",
                tile_size=tile_size,
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
            )
        )
        image_rows[9].append(
            build_projected_pcd_rgb_tile(
                color_image=color_image,
                points_world=ffs_payload["points"],
                colors_bgr=ffs_payload["colors"],
                source_camera_idx=ffs_payload["source_camera_idx"],
                K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
                label=f"FFS PCD {_format_point_count(len(ffs_payload['points']))}",
                tile_size=tile_size,
                point_radius_px=int(projection_point_radius_px),
            )
        )
        image_rows[10].append(
            build_projected_pcd_rgb_tile(
                color_image=color_image,
                points_world=ffs_filtered_points,
                colors_bgr=ffs_filtered_colors,
                source_camera_idx=ffs_payload["source_camera_idx"][ffs_kept_mask],
                K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
                label=f"FFS kept {_format_point_count(len(ffs_filtered_points))}",
                tile_size=tile_size,
                point_radius_px=int(projection_point_radius_px),
            )
        )
        image_rows[11].append(
            build_projected_pcd_rgb_tile(
                color_image=color_image,
                points_world=ffs_payload["points"],
                colors_bgr=ffs_payload["colors"],
                source_camera_idx=ffs_payload["source_camera_idx"],
                K_color=np.asarray(metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(c2w_list[camera_idx], dtype=np.float32),
                label=f"FFS removed={int(np.count_nonzero(ffs_removed_mask))}",
                tile_size=tile_size,
                removed_mask=ffs_removed_mask,
                point_radius_px=int(projection_point_radius_px),
                removed_radius_px=int(highlight_radius_px),
            )
        )
        ffs_source_removed = ffs_removed_mask & (ffs_payload["source_camera_idx"] == int(camera_idx))
        image_rows[12].append(
            _build_rgb_removed_source_tile(
                color_image=color_image,
                source_pixel_uv=ffs_payload["source_pixel_uv"][ffs_source_removed],
                label=f"FFS RGB removed={int(np.count_nonzero(ffs_source_removed))}",
                tile_size=tile_size,
                source_camera_idx=int(camera_idx),
                radius_px=int(highlight_radius_px),
                alpha=float(highlight_alpha),
            )
        )
        per_camera_summary.append(
            {
                "camera_idx": int(camera_idx),
                "serial": str(metadata["serial_numbers"][camera_idx]),
                "mask_pixel_count": int(np.count_nonzero(object_mask)),
                "native_masked_point_count": int(native_payload["per_camera"][camera_idx]["masked_point_count"]),
                "ffs_masked_point_count": int(ffs_payload["per_camera"][camera_idx]["masked_point_count"]),
                "native_removed_source_point_count": int(np.count_nonzero(native_source_removed)),
                "ffs_removed_source_point_count": int(np.count_nonzero(ffs_source_removed)),
            }
        )

    model_config = {
        "ffs_model_name": str(metadata.get("ffs_config", {}).get("model_name", "20-30-48")),
        "ffs_valid_iters": int(metadata.get("ffs_config", {}).get("valid_iters", 4)),
        "trt_builder_optimization_level": int(metadata.get("ffs_config", {}).get("builder_optimization_level", 5)),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
        "enhanced_component_voxel_size_m": float(enhanced_component_voxel_size_m),
        "enhanced_keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        "row_label_width": int(row_label_width),
        "source_camera_highlight_labels": list(DEFAULT_SOURCE_CAMERA_HIGHLIGHT_LABELS),
    }
    board = build_still_object_projection_board(
        round_label="Still Object Round 1",
        frame_idx=int(frame_idx),
        model_config=model_config,
        column_headers=[f"Cam{idx} | {metadata['serial_numbers'][idx]}" for idx in camera_ids],
        image_rows=image_rows,
    )
    board_path = output_root / f"still_object_round1_projection_panel_13x3_frame_{int(frame_idx):04d}.png"
    write_image(board_path, board)
    summary = {
        "output_dir": str(output_root.resolve()),
        "board_path": str(board_path.resolve()),
        "aligned_case_dir": str(case_dir.resolve()),
        "mask_root": str(Path(mask_root).resolve()),
        "frame_idx": int(frame_idx),
        "row_headers": list(ROW_HEADERS_13),
        "model_config": model_config,
        "mask_debug": mask_debug,
        "native_filter_stats": native_filter_stats,
        "ffs_filter_stats": ffs_filter_stats,
        "total_counts": {
            "native_masked_point_count": int(len(native_payload["points"])),
            "native_kept_point_count": int(len(native_filtered_points)),
            "native_removed_point_count": int(np.count_nonzero(native_removed_mask)),
            "ffs_masked_point_count": int(len(ffs_payload["points"])),
            "ffs_kept_point_count": int(len(ffs_filtered_points)),
            "ffs_removed_point_count": int(np.count_nonzero(ffs_removed_mask)),
        },
        "per_camera": per_camera_summary,
        "render_contract": {
            "rows": "rgb_mask_ir_left_ir_right_rs_depth_rs_pcd_rs_kept_rs_removed_rs_rgb_removed_ffs_depth_ffs_pcd_ffs_kept_ffs_removed_ffs_rgb_removed",
            "projection_mode": "fused_masked_world_pcd_projected_to_each_original_rgb_camera",
            "background_mode": "white_blurred_rgb",
            "removed_highlight_color_mode": "source_camera",
        },
    }
    result_summary_path = output_root / "result_summary.txt"
    result_summary_path.write_text(_format_result_summary_text(summary), encoding="utf-8")
    summary["result_summary_path"] = str(result_summary_path.resolve())
    write_json(output_root / "summary.json", summary)
    return summary
