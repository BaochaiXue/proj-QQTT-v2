from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..calibration_io import load_calibration_transforms
from ..depth_colormap import colorize_depth_meters
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_metadata, resolve_case_dir
from ..layouts import compose_registration_matrix_board
from ..workflows.masked_pointcloud_compare import PHYSTWIN_DATA_PROCESS_MASK_CONTRACT, load_union_masks_for_camera_clouds
from .enhanced_phystwin_postprocess_pcd_compare import (
    DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
)
from .ffs_confidence_filter_pcd_compare import (
    _apply_enhanced_phystwin_like_postprocess_with_trace,
    _build_view_configs,
    _build_world_cloud,
    _format_point_count,
    _load_color_image,
)
from .ffs_confidence_panels import DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT
from .ffs_confidence_pcd_panels import _render_open3d_offscreen_pinhole
from .native_ffs_fused_pcd_compare import (
    DEFAULT_PHYSTWIN_NB_POINTS,
    DEFAULT_PHYSTWIN_RADIUS_M,
    _load_depth_m,
)


DEFAULT_OUTPUT_ROOT_NAME = "enhanced_phystwin_removed_overlay_frame_0000"
DEFAULT_ENHANCED_PHYSTWIN_REMOVED_OVERLAY_ROUNDS: tuple[dict[str, str], ...] = (
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
    {
        "round_id": "round4",
        "round_label": "Round 4",
        "native_case_ref": "static/native_30_static_round4_20260427",
        "ffs_case_ref": "static/ffs_30_static_round4_20260427",
        "mask_root": "static/masked_pointcloud_compare_round4_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks",
    },
    {
        "round_id": "round5",
        "round_label": "Round 5",
        "native_case_ref": "static/native_30_static_round5_20260427",
        "ffs_case_ref": "static/ffs_30_static_round5_20260427",
        "mask_root": "static/masked_pointcloud_compare_round5_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks",
    },
    {
        "round_id": "round6",
        "round_label": "Round 6",
        "native_case_ref": "static/native_30_static_round6_20260427",
        "ffs_case_ref": "static/ffs_30_static_round6_20260427",
        "mask_root": "static/masked_pointcloud_compare_round6_frame_0000_stuffed_animal/_generated_masks/ffs/sam31_masks",
    },
)
DEFAULT_TILE_WIDTH = 480
DEFAULT_TILE_HEIGHT = 360
DEFAULT_ROW_LABEL_WIDTH = 300
DEFAULT_HIGHLIGHT_COLOR_BGR = (255, 0, 255)
DEFAULT_SOURCE_CAMERA_HIGHLIGHT_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 0, 255),  # Cam0: magenta
    (255, 255, 0),  # Cam1: cyan
    (0, 191, 255),  # Cam2: amber
)
DEFAULT_SOURCE_CAMERA_HIGHLIGHT_LABELS: tuple[str, ...] = ("Cam0 magenta", "Cam1 cyan", "Cam2 amber")
HIGHLIGHT_SCOPES = ("all", "radius", "component")
NATIVE_ROW_MODES = ("native_depth", "ir_pair")


def build_static_enhanced_phystwin_removed_overlay_round_specs(*, aligned_root: Path) -> list[dict[str, Any]]:
    root = Path(aligned_root).resolve()
    round_specs: list[dict[str, Any]] = []
    for item in DEFAULT_ENHANCED_PHYSTWIN_REMOVED_OVERLAY_ROUNDS:
        spec = dict(item)
        spec["mask_root"] = root / str(item["mask_root"])
        round_specs.append(spec)
    return round_specs


def _concat_or_empty(arrays: list[np.ndarray], *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    non_empty = [np.asarray(item, dtype=dtype) for item in arrays if len(item) > 0]
    if not non_empty:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(non_empty, axis=0)


def _pixel_mask_from_uv(
    source_pixel_uv: np.ndarray,
    *,
    image_shape: tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    height, width = [int(item) for item in image_shape[:2]]
    mask = np.zeros((height, width), dtype=bool)
    uv = np.asarray(source_pixel_uv, dtype=np.int32).reshape(-1, 2)
    if len(uv) == 0:
        return mask
    x = np.clip(uv[:, 0], 0, width - 1)
    y = np.clip(uv[:, 1], 0, height - 1)
    mask[y, x] = True
    radius = int(radius_px)
    if radius > 0:
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    bool_mask = np.asarray(mask, dtype=bool)
    if bool_mask.shape != canvas.shape[:2]:
        raise ValueError(f"overlay mask shape must match image. Got {bool_mask.shape} vs {canvas.shape[:2]}.")
    if not np.any(bool_mask):
        return canvas
    color = np.zeros_like(canvas, dtype=np.uint8)
    color[...] = np.asarray(color_bgr, dtype=np.uint8)
    blended = np.clip(
        canvas.astype(np.float32) * (1.0 - float(alpha)) + color.astype(np.float32) * float(alpha),
        0.0,
        255.0,
    ).astype(np.uint8)
    canvas[bool_mask] = blended[bool_mask]
    return canvas


def _source_camera_highlight_color_bgr(camera_idx: int) -> tuple[int, int, int]:
    palette = DEFAULT_SOURCE_CAMERA_HIGHLIGHT_COLORS_BGR
    return tuple(int(item) for item in palette[int(camera_idx) % len(palette)])


def _build_rgb_object_mask_tile(
    *,
    color_image: np.ndarray,
    object_mask: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
) -> np.ndarray:
    overlay = _overlay_mask(
        color_image,
        object_mask,
        color_bgr=(0, 220, 90),
        alpha=0.42,
    )
    return label_tile(overlay, label, tile_size)


def _build_masked_depth_removed_tile(
    *,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    removed_pixel_mask: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
    depth_min_m: float,
    depth_max_m: float,
    highlight_color_bgr: tuple[int, int, int],
    highlight_alpha: float,
) -> np.ndarray:
    valid_mask = np.asarray(object_mask, dtype=bool) & np.isfinite(depth_m) & (np.asarray(depth_m, dtype=np.float32) > 0.0)
    masked_depth = np.where(valid_mask, np.asarray(depth_m, dtype=np.float32), 0.0)
    depth_vis = colorize_depth_meters(
        masked_depth,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        invalid_color=(0, 0, 0),
    )
    overlay = _overlay_mask(
        depth_vis,
        removed_pixel_mask,
        color_bgr=highlight_color_bgr,
        alpha=float(highlight_alpha),
    )
    return label_tile(overlay, label, tile_size)


def _build_masked_depth_tile(
    *,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    valid_mask = np.asarray(object_mask, dtype=bool) & np.isfinite(depth_m) & (np.asarray(depth_m, dtype=np.float32) > 0.0)
    masked_depth = np.where(valid_mask, np.asarray(depth_m, dtype=np.float32), 0.0)
    depth_vis = colorize_depth_meters(
        masked_depth,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        invalid_color=(0, 0, 0),
    )
    return label_tile(depth_vis, label, tile_size)


def _build_rgb_removed_tile(
    *,
    color_image: np.ndarray,
    removed_pixel_mask: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
    highlight_color_bgr: tuple[int, int, int],
    highlight_alpha: float,
) -> np.ndarray:
    overlay = _overlay_mask(
        color_image,
        removed_pixel_mask,
        color_bgr=highlight_color_bgr,
        alpha=float(highlight_alpha),
    )
    return label_tile(overlay, label, tile_size)


def _load_ir_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Missing IR image: {path}")
    if image.ndim == 2:
        if image.dtype == np.uint8:
            gray = image
        else:
            gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8)
    raise ValueError(f"Unsupported IR image shape for {path}: {image.shape}")


def _build_ir_tile(
    *,
    ir_image: np.ndarray,
    label: str,
    tile_size: tuple[int, int],
) -> np.ndarray:
    return label_tile(np.asarray(ir_image, dtype=np.uint8), label, tile_size)


def _select_removed_mask(trace: dict[str, np.ndarray], *, highlight_scope: str) -> np.ndarray:
    scope = str(highlight_scope).strip().lower()
    if scope == "all":
        return np.asarray(trace["removed_mask"], dtype=bool)
    if scope == "radius":
        return np.asarray(trace["radius_removed_mask"], dtype=bool)
    if scope == "component":
        return np.asarray(trace["component_removed_mask"], dtype=bool)
    raise ValueError(f"Unsupported highlight_scope={highlight_scope!r}; expected one of {HIGHLIGHT_SCOPES}.")


def build_enhanced_phystwin_removed_overlay_board(
    *,
    round_label: str,
    frame_idx: int,
    model_config: dict[str, Any],
    column_headers: list[str],
    image_rows: list[list[np.ndarray]],
    row_headers: list[str] | None = None,
) -> np.ndarray:
    if len(image_rows) not in {5, 6} or any(len(row) != 3 for row in image_rows):
        raise ValueError("Enhanced removed overlay board requires a 5x3 or 6x3 image matrix.")
    selected_row_headers = row_headers or (
        [
            "RGB + object mask",
            "RealSense depth mask",
            "PCD + removed",
            "FFS depth + removed",
            "RGB + removed",
        ]
        if len(image_rows) == 5
        else [
            "RGB + object mask",
            "IR left",
            "IR right",
            "PCD + removed",
            "FFS depth + removed",
            "RGB + removed",
        ]
    )
    if len(selected_row_headers) != len(image_rows):
        raise ValueError("row_headers must match image_rows.")
    row_summary = (
        "rows=RGB mask / RealSense depth / PCD + removed / FFS depth + removed / RGB + removed"
        if len(image_rows) == 5
        else "rows=RGB mask / IR left / IR right / PCD + removed / FFS depth + removed / RGB + removed"
    )
    return compose_registration_matrix_board(
        title_lines=[
            f"Enhanced PT-like Removed Points Overlay | {round_label} | frame {int(frame_idx)}",
            f"{row_summary} | highlight={model_config['highlight_scope']} | source-camera colors/pixels",
            (
                f"radius={float(model_config['phystwin_radius_m']):.3f}m/"
                f"{int(model_config['phystwin_nb_points'])}nn | "
                f"component_voxel={float(model_config['enhanced_component_voxel_size_m']):.3f}m | "
                f"alpha={float(model_config['highlight_alpha']):.2f}"
            ),
        ],
        row_headers=selected_row_headers,
        column_headers=column_headers,
        image_rows=image_rows,
        row_label_width=int(model_config.get("row_label_width", DEFAULT_ROW_LABEL_WIDTH)),
    )


def run_enhanced_phystwin_removed_overlay_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    frame_idx: int = 0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    point_size: float = 2.0,
    look_distance: float = 1.0,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    row_label_width: int = DEFAULT_ROW_LABEL_WIDTH,
    max_points_per_camera: int | None = 80_000,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
    highlight_scope: str = "all",
    highlight_alpha: float = 0.65,
    highlight_radius_px: int = 2,
    use_float_ffs_depth_when_available: bool = True,
    native_row_mode: str = "native_depth",
    round_specs: list[dict[str, Any]] | None = None,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if float(depth_max_m) <= float(depth_min_m):
        raise ValueError(f"depth_max_m must be greater than depth_min_m. Got {depth_min_m}, {depth_max_m}.")
    if str(highlight_scope).strip().lower() not in HIGHLIGHT_SCOPES:
        raise ValueError(f"Unsupported highlight_scope={highlight_scope!r}; expected one of {HIGHLIGHT_SCOPES}.")
    if int(highlight_radius_px) < 0:
        raise ValueError(f"highlight_radius_px must be >= 0, got {highlight_radius_px}.")
    native_row_mode = str(native_row_mode).strip().lower()
    if native_row_mode not in NATIVE_ROW_MODES:
        raise ValueError(f"Unsupported native_row_mode={native_row_mode!r}; expected one of {NATIVE_ROW_MODES}.")

    selected_round_specs = (
        build_static_enhanced_phystwin_removed_overlay_round_specs(aligned_root=aligned_root)
        if round_specs is None
        else [dict(item) for item in round_specs]
    )
    render_frame_fn = render_frame_fn or _render_open3d_offscreen_pinhole
    model_config = {
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "point_size": float(point_size),
        "look_distance": float(look_distance),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "row_label_width": int(row_label_width),
        "max_points_per_camera": None if max_points_per_camera is None else int(max_points_per_camera),
        "text_prompt": str(text_prompt),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
        "enhanced_component_voxel_size_m": float(enhanced_component_voxel_size_m),
        "enhanced_keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        "highlight_scope": str(highlight_scope).strip().lower(),
        "highlight_alpha": float(highlight_alpha),
        "highlight_radius_px": int(highlight_radius_px),
        "highlight_color_bgr": list(DEFAULT_HIGHLIGHT_COLOR_BGR),
        "source_camera_highlight_colors_bgr": {
            str(camera_idx): list(_source_camera_highlight_color_bgr(camera_idx)) for camera_idx in range(3)
        },
        "source_camera_highlight_labels": list(DEFAULT_SOURCE_CAMERA_HIGHLIGHT_LABELS),
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "native_row_mode": native_row_mode,
    }

    rounds_summary: list[dict[str, Any]] = []
    for round_spec in selected_round_specs:
        native_case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(round_spec["native_case_ref"]))
        ffs_case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(round_spec["ffs_case_ref"]))
        native_metadata = load_case_metadata(native_case_dir)
        ffs_metadata = load_case_metadata(ffs_case_dir)
        selected_frame_idx = int(frame_idx)
        native_max_frame = int(native_metadata["frame_num"]) - 1
        ffs_max_frame = int(ffs_metadata["frame_num"]) - 1
        max_frame = min(native_max_frame, ffs_max_frame)
        if selected_frame_idx < 0 or selected_frame_idx > max_frame:
            raise ValueError(
                f"frame_idx={selected_frame_idx} is out of range for {round_spec['round_id']}; "
                f"expected 0 <= frame_idx <= {max_frame} "
                f"(native max={native_max_frame}, ffs max={ffs_max_frame})."
            )
        camera_ids = list(range(len(ffs_metadata["serial_numbers"])))
        if len(camera_ids) != 3:
            raise ValueError(f"Expected exactly 3 FFS cameras for {round_spec['round_id']}, got {camera_ids}.")
        if len(native_metadata["serial_numbers"]) != len(camera_ids):
            raise ValueError(
                f"Expected native camera count to match FFS for {round_spec['round_id']}; "
                f"got native={len(native_metadata['serial_numbers'])}, ffs={len(camera_ids)}."
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
        minimal_clouds = [
            {
                "camera_idx": int(camera_idx),
                "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                "color_path": str(ffs_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"),
            }
            for camera_idx in camera_ids
        ]
        masks_by_camera, raw_mask_debug = load_union_masks_for_camera_clouds(
            mask_root=Path(round_spec["mask_root"]).resolve(),
            camera_clouds=minimal_clouds,
            frame_token=str(selected_frame_idx),
            text_prompt=str(text_prompt),
        )

        camera_payloads: list[dict[str, Any]] = []
        point_sets: list[np.ndarray] = []
        color_sets: list[np.ndarray] = []
        source_camera_sets: list[np.ndarray] = []
        source_uv_sets: list[np.ndarray] = []
        for camera_idx in camera_ids:
            color_path = ffs_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            color_image = _load_color_image(color_path)
            depth_m, depth_info = _load_depth_m(
                case_dir=ffs_case_dir,
                metadata=ffs_metadata,
                camera_idx=int(camera_idx),
                frame_idx=selected_frame_idx,
                depth_source="ffs_raw",
                use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
            )
            if native_row_mode == "native_depth":
                native_depth_m, native_depth_info = _load_depth_m(
                    case_dir=native_case_dir,
                    metadata=native_metadata,
                    camera_idx=int(camera_idx),
                    frame_idx=selected_frame_idx,
                    depth_source="realsense",
                    use_float_ffs_depth_when_available=False,
                )
            else:
                native_depth_m = np.zeros(depth_m.shape, dtype=np.float32)
                native_depth_info = {
                    "depth_dir_used": None,
                    "source_depth_dir_used": None,
                    "used_float_depth": False,
                    "depth_path": None,
                    "native_depth_row_disabled": True,
                    "native_row_mode": native_row_mode,
                }
            object_mask = np.asarray(masks_by_camera.get(int(camera_idx), np.zeros(depth_m.shape, dtype=bool)), dtype=bool)
            cloud = _build_world_cloud(
                depth_m=depth_m,
                color_image=color_image,
                K_color=np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(ffs_c2w_list[camera_idx], dtype=np.float32),
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                max_points_per_camera=max_points_per_camera,
                object_mask=object_mask,
            )
            points = np.asarray(cloud["points"], dtype=np.float32)
            colors = np.asarray(cloud["colors"], dtype=np.uint8)
            source_uv = np.asarray(cloud["source_pixel_uv"], dtype=np.int32).reshape(-1, 2)
            point_sets.append(points)
            color_sets.append(colors)
            source_camera_sets.append(np.full((len(points),), int(camera_idx), dtype=np.int32))
            source_uv_sets.append(source_uv)
            camera_payloads.append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                    "color_path": str(color_path.resolve()),
                    "color_image": color_image,
                    "depth_m": depth_m,
                    "depth_info": dict(depth_info),
                    "native_depth_m": native_depth_m,
                    "native_depth_info": dict(native_depth_info),
                    "object_mask": object_mask,
                    "raw_point_count": int(len(points)),
                    "mask_pixel_count": int(np.count_nonzero(object_mask)),
                    **dict(cloud["stats"]),
                }
            )

        fused_points = _concat_or_empty(point_sets, shape=(0, 3), dtype=np.float32)
        fused_colors = _concat_or_empty(color_sets, shape=(0, 3), dtype=np.uint8)
        source_camera_idx = _concat_or_empty(source_camera_sets, shape=(0,), dtype=np.int32)
        source_pixel_uv = _concat_or_empty(source_uv_sets, shape=(0, 2), dtype=np.int32)
        filtered_points, _filtered_colors, postprocess_stats, trace = _apply_enhanced_phystwin_like_postprocess_with_trace(
            points=fused_points,
            colors=fused_colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
            component_voxel_size_m=float(enhanced_component_voxel_size_m),
            keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        selected_removed_mask = _select_removed_mask(trace, highlight_scope=str(highlight_scope))
        pcd_colors = np.asarray(fused_colors, dtype=np.uint8).copy()
        if len(pcd_colors) > 0:
            for camera_idx in camera_ids:
                source_removed_mask = selected_removed_mask & (source_camera_idx == int(camera_idx))
                pcd_colors[source_removed_mask] = np.asarray(
                    _source_camera_highlight_color_bgr(int(camera_idx)),
                    dtype=np.uint8,
                )
        selected_removed_by_source_camera = {
            str(camera_idx): int(np.count_nonzero(selected_removed_mask & (source_camera_idx == int(camera_idx))))
            for camera_idx in camera_ids
        }

        image_rows = [[], [], [], [], []] if native_row_mode == "native_depth" else [[], [], [], [], [], []]
        pcd_row_idx = 2 if native_row_mode == "native_depth" else 3
        ffs_depth_row_idx = 3 if native_row_mode == "native_depth" else 4
        rgb_removed_row_idx = 4 if native_row_mode == "native_depth" else 5
        per_camera_summary: list[dict[str, Any]] = []
        render_summary: list[dict[str, Any]] = []
        pcd_row_images: list[np.ndarray] = []
        for view_config in view_configs:
            target_w, target_h = [int(item) for item in view_config["image_size"]]
            rendered = render_frame_fn(
                fused_points,
                pcd_colors,
                width=int(target_w),
                height=int(target_h),
                center=np.asarray(view_config["center"], dtype=np.float32),
                eye=np.asarray(view_config["camera_position"], dtype=np.float32),
                up=np.asarray(view_config["up"], dtype=np.float32),
                zoom=0.55,
                point_size=float(point_size),
                intrinsic_matrix=np.asarray(view_config["intrinsic_matrix"], dtype=np.float32),
                extrinsic_matrix=np.asarray(view_config["extrinsic_matrix"], dtype=np.float32),
                render_kind="enhanced_phystwin_removed_overlay_pcd",
                metric_name=str(highlight_scope),
                camera_idx=int(view_config["camera_idx"]),
            )
            rendered = label_tile(
                rendered,
                (
                    f"{_format_point_count(len(fused_points))} | removed={int(np.count_nonzero(selected_removed_mask))} | "
                    "C0/C1/C2="
                    f"{selected_removed_by_source_camera.get('0', 0)}/"
                    f"{selected_removed_by_source_camera.get('1', 0)}/"
                    f"{selected_removed_by_source_camera.get('2', 0)}"
                ),
                (int(tile_width), int(tile_height)),
            )
            pcd_row_images.append(rendered)
            render_summary.append(
                {
                    "camera_idx": int(view_config["camera_idx"]),
                    "point_count": int(len(fused_points)),
                    "removed_point_count": int(np.count_nonzero(selected_removed_mask)),
                    "removed_point_count_by_source_camera": dict(selected_removed_by_source_camera),
                    "tile_width": int(tile_width),
                    "tile_height": int(tile_height),
                }
            )
        image_rows[pcd_row_idx] = pcd_row_images

        for camera_payload in camera_payloads:
            camera_idx = int(camera_payload["camera_idx"])
            camera_point_mask = source_camera_idx == camera_idx
            radius_removed_point_mask = camera_point_mask & np.asarray(trace["radius_removed_mask"], dtype=bool)
            component_removed_point_mask = camera_point_mask & np.asarray(trace["component_removed_mask"], dtype=bool)
            removed_point_mask = camera_point_mask & selected_removed_mask
            removed_pixel_mask = _pixel_mask_from_uv(
                source_pixel_uv[removed_point_mask],
                image_shape=np.asarray(camera_payload["color_image"]).shape[:2],
                radius_px=int(highlight_radius_px),
            )
            radius_removed_count = int(np.count_nonzero(radius_removed_point_mask))
            component_removed_count = int(np.count_nonzero(component_removed_point_mask))
            total_removed_count = int(np.count_nonzero(removed_point_mask))
            highlight_color_bgr = _source_camera_highlight_color_bgr(camera_idx)
            highlight_color_label = DEFAULT_SOURCE_CAMERA_HIGHLIGHT_LABELS[
                camera_idx % len(DEFAULT_SOURCE_CAMERA_HIGHLIGHT_LABELS)
            ]
            image_rows[0].append(
                _build_rgb_object_mask_tile(
                    color_image=np.asarray(camera_payload["color_image"], dtype=np.uint8),
                    object_mask=np.asarray(camera_payload["object_mask"], dtype=bool),
                    label=f"mask={int(camera_payload['mask_pixel_count'])} px | raw={int(camera_payload['raw_point_count'])}",
                    tile_size=(int(tile_width), int(tile_height)),
                )
            )
            native_valid_mask = (
                np.asarray(camera_payload["object_mask"], dtype=bool)
                & np.isfinite(np.asarray(camera_payload["native_depth_m"], dtype=np.float32))
                & (np.asarray(camera_payload["native_depth_m"], dtype=np.float32) > 0.0)
            )
            if native_row_mode == "native_depth":
                image_rows[1].append(
                    _build_masked_depth_tile(
                        depth_m=np.asarray(camera_payload["native_depth_m"], dtype=np.float32),
                        object_mask=np.asarray(camera_payload["object_mask"], dtype=bool),
                        label=f"native valid={int(np.count_nonzero(native_valid_mask))} px",
                        tile_size=(int(tile_width), int(tile_height)),
                        depth_min_m=float(depth_min_m),
                        depth_max_m=float(depth_max_m),
                    )
                )
            else:
                for row_idx, stream_name, stream_label in (
                    (1, "ir_left", "IR left"),
                    (2, "ir_right", "IR right"),
                ):
                    image_rows[row_idx].append(
                        _build_ir_tile(
                            ir_image=_load_ir_image(
                                ffs_case_dir / stream_name / str(camera_idx) / f"{selected_frame_idx}.png"
                            ),
                            label=stream_label,
                            tile_size=(int(tile_width), int(tile_height)),
                        )
                    )
            image_rows[ffs_depth_row_idx].append(
                _build_masked_depth_removed_tile(
                    depth_m=np.asarray(camera_payload["depth_m"], dtype=np.float32),
                    object_mask=np.asarray(camera_payload["object_mask"], dtype=bool),
                    removed_pixel_mask=removed_pixel_mask,
                    label=(
                        f"removed={total_removed_count} | r={radius_removed_count} c={component_removed_count} | "
                        f"{highlight_color_label}"
                    ),
                    tile_size=(int(tile_width), int(tile_height)),
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    highlight_color_bgr=highlight_color_bgr,
                    highlight_alpha=float(highlight_alpha),
                )
            )
            image_rows[rgb_removed_row_idx].append(
                _build_rgb_removed_tile(
                    color_image=np.asarray(camera_payload["color_image"], dtype=np.uint8),
                    removed_pixel_mask=removed_pixel_mask,
                    label=(
                        f"RGB removed={total_removed_count} | pixels={int(np.count_nonzero(removed_pixel_mask))} | "
                        f"{highlight_color_label}"
                    ),
                    tile_size=(int(tile_width), int(tile_height)),
                    highlight_color_bgr=highlight_color_bgr,
                    highlight_alpha=float(highlight_alpha),
                )
            )
            per_camera_summary.append(
                {
                    "camera_idx": camera_idx,
                    "serial": str(camera_payload["serial"]),
                    "mask_pixel_count": int(camera_payload["mask_pixel_count"]),
                    "raw_point_count": int(camera_payload["raw_point_count"]),
                    "radius_removed_point_count": radius_removed_count,
                    "component_removed_point_count": component_removed_count,
                    "total_removed_point_count": total_removed_count,
                    "removed_overlay_pixel_count": int(np.count_nonzero(removed_pixel_mask)),
                    "highlight_color_bgr": list(highlight_color_bgr),
                    "highlight_color_label": str(highlight_color_label),
                    "depth_info": dict(camera_payload["depth_info"]),
                    "native_depth_info": dict(camera_payload["native_depth_info"]),
                    "native_masked_valid_pixel_count": int(np.count_nonzero(native_valid_mask)),
                }
            )

        round_output_dir = output_root / str(round_spec["round_id"])
        round_output_dir.mkdir(parents=True, exist_ok=True)
        column_headers = [str(view_config["label"]) for view_config in view_configs]
        row_headers = (
            [
                "RGB + object mask",
                "RealSense depth mask",
                "PCD + removed",
                "FFS depth + removed",
                "RGB + removed",
            ]
            if native_row_mode == "native_depth"
            else [
                "RGB + object mask",
                "IR left",
                "IR right",
                "PCD + removed",
                "FFS depth + removed",
                "RGB + removed",
            ]
        )
        board = build_enhanced_phystwin_removed_overlay_board(
            round_label=str(round_spec["round_label"]),
            frame_idx=selected_frame_idx,
            model_config=model_config,
            column_headers=column_headers,
            image_rows=image_rows,
            row_headers=row_headers,
        )
        matrix_label = f"{len(image_rows)}x{len(image_rows[0])}"
        suffix = "" if native_row_mode == "native_depth" else f"_{native_row_mode}"
        board_path = (
            round_output_dir
            / f"enhanced_phystwin_removed_overlay_{matrix_label}{suffix}_frame_{selected_frame_idx:04d}.png"
        )
        write_image(board_path, board)
        round_summary = {
            "round_id": str(round_spec["round_id"]),
            "round_label": str(round_spec["round_label"]),
            "native_case_ref": str(round_spec["native_case_ref"]),
            "native_case_dir": str(native_case_dir.resolve()),
            "ffs_case_ref": str(round_spec["ffs_case_ref"]),
            "ffs_case_dir": str(ffs_case_dir.resolve()),
            "frame_idx": int(selected_frame_idx),
            "board_path": str(board_path.resolve()),
            "model_config": dict(model_config),
            "mask_root": str(Path(round_spec["mask_root"]).resolve()),
            "mask_debug": {str(camera_idx): dict(raw_mask_debug.get(camera_idx, {})) for camera_idx in camera_ids},
            "fused_point_count": int(len(fused_points)),
            "filtered_point_count": int(len(filtered_points)),
            "radius_removed_point_count": int(np.count_nonzero(trace["radius_removed_mask"])),
            "component_removed_point_count": int(np.count_nonzero(trace["component_removed_mask"])),
            "total_removed_point_count": int(np.count_nonzero(selected_removed_mask)),
            "total_removed_point_count_by_source_camera": dict(selected_removed_by_source_camera),
            "postprocess_stats": dict(postprocess_stats),
            "per_camera": per_camera_summary,
            "render_summary": render_summary,
            "render_contract": {
                "projection_mode": "source_camera_pixels_no_cross_reprojection",
                "pcd_projection_mode": "original_camera_pinhole",
                "rows": (
                    "rgb_mask_native_depth_mask_pcd_removed_ffs_depth_removed_rgb_removed"
                    if native_row_mode == "native_depth"
                    else "rgb_mask_ir_left_ir_right_pcd_removed_ffs_depth_removed_rgb_removed"
                ),
                "native_row_mode": native_row_mode,
                "highlight_scope": str(highlight_scope),
                "highlight_color_mode": "source_camera",
                "source_camera_highlight_colors_bgr": {
                    str(camera_idx): list(_source_camera_highlight_color_bgr(camera_idx)) for camera_idx in camera_ids
                },
                "object_masked": True,
                "formal_depth_written": False,
            },
            "postprocess": {
                "mode": "enhanced_phystwin_like_radius_then_component_filter",
                "radius_m": float(phystwin_radius_m),
                "nb_points": int(phystwin_nb_points),
                "component_voxel_size_m": float(enhanced_component_voxel_size_m),
                "keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
                "reference_contract": dict(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT),
            },
            "output_dir": str(round_output_dir.resolve()),
        }
        write_json(round_output_dir / "summary.json", round_summary)
        rounds_summary.append(round_summary)

    manifest = {
        "output_dir": str(output_root.resolve()),
        "frame_idx": int(frame_idx),
        "model_config": dict(model_config),
        "rounds": rounds_summary,
    }
    write_json(output_root / "summary.json", manifest)
    return manifest
