from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..calibration_io import load_calibration_transforms
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_metadata, load_depth_frame, resolve_case_dir
from ..layouts import compose_registration_matrix_board
from .ffs_confidence_filter_pcd_compare import (
    _apply_phystwin_like_radius_postprocess,
    _build_view_configs,
    _build_world_cloud,
    _format_point_count,
    _fuse_camera_clouds,
    _load_color_image,
    _serialize_view_config,
)
from .ffs_confidence_panels import DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT
from .ffs_confidence_pcd_panels import _render_open3d_offscreen_pinhole
from .masked_pointcloud_compare import PHYSTWIN_DATA_PROCESS_MASK_CONTRACT, load_union_masks_for_camera_clouds


DEFAULT_NATIVE_FFS_FUSED_PCD_ROUNDS: tuple[dict[str, str], ...] = (
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
DEFAULT_PHYSTWIN_RADIUS_M = float(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["radius_m"])
DEFAULT_PHYSTWIN_NB_POINTS = int(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["nb_points"])


def build_static_native_ffs_fused_pcd_round_specs(*, aligned_root: Path) -> list[dict[str, Any]]:
    root = Path(aligned_root).resolve()
    specs: list[dict[str, Any]] = []
    for item in DEFAULT_NATIVE_FFS_FUSED_PCD_ROUNDS:
        spec = dict(item)
        spec["mask_root"] = (root / str(item["mask_root"])).resolve()
        specs.append(spec)
    return specs


def fuse_native_ffs_depth(
    native_depth_m: np.ndarray,
    ffs_depth_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    native = np.asarray(native_depth_m, dtype=np.float32)
    ffs = np.asarray(ffs_depth_m, dtype=np.float32)
    if native.shape != ffs.shape:
        raise ValueError(f"native and FFS depth shapes must match. Got {native.shape} vs {ffs.shape}.")
    if native.ndim != 2:
        raise ValueError(f"Expected 2D depth maps, got shape={native.shape}.")

    native_positive = np.isfinite(native) & (native > 0.0)
    native_missing = ~native_positive
    native_keep = native_positive
    ffs_valid = np.isfinite(ffs) & (ffs > 0.0)
    use_ffs_candidate = native_missing
    use_ffs_valid = use_ffs_candidate & ffs_valid

    fused = np.zeros(native.shape, dtype=np.float32)
    fused[native_keep] = native[native_keep]
    fused[use_ffs_valid] = ffs[use_ffs_valid]
    pixel_count = int(native.size)
    stats = {
        "mode": "missing_only",
        "pixel_count": pixel_count,
        "native_valid_pixel_count": int(np.count_nonzero(native_positive)),
        "native_missing_pixel_count": int(np.count_nonzero(native_missing)),
        "native_kept_pixel_count": int(np.count_nonzero(native_keep)),
        "ffs_valid_pixel_count": int(np.count_nonzero(ffs_valid)),
        "ffs_candidate_pixel_count": int(np.count_nonzero(use_ffs_candidate)),
        "ffs_filled_pixel_count": int(np.count_nonzero(use_ffs_valid)),
        "fused_valid_pixel_count": int(np.count_nonzero(np.isfinite(fused) & (fused > 0.0))),
        "unfilled_pixel_count": int(np.count_nonzero(use_ffs_candidate & ~ffs_valid)),
        "ffs_fill_ratio": float(np.count_nonzero(use_ffs_valid) / max(1, pixel_count)),
        "native_keep_ratio": float(np.count_nonzero(native_keep) / max(1, pixel_count)),
    }
    return fused, stats


def _erode_object_masks(
    mask_by_camera: dict[int, np.ndarray],
    *,
    erode_pixels: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    iterations = int(erode_pixels)
    if iterations < 0:
        raise ValueError(f"mask_erode_pixels must be >= 0, got {erode_pixels}.")
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_by_camera: dict[int, np.ndarray] = {}
    debug_by_camera: dict[int, dict[str, Any]] = {}
    for camera_idx, mask in mask_by_camera.items():
        mask_bool = np.asarray(mask, dtype=bool)
        original_count = int(np.count_nonzero(mask_bool))
        if iterations == 0:
            eroded = mask_bool.copy()
        else:
            eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=iterations) > 0
        eroded_count = int(np.count_nonzero(eroded))
        eroded_by_camera[int(camera_idx)] = eroded
        debug_by_camera[int(camera_idx)] = {
            "camera_idx": int(camera_idx),
            "mask_erode_pixels": iterations,
            "mask_pixel_count_before_erode": original_count,
            "mask_pixel_count_after_erode": eroded_count,
            "eroded_removed_pixel_count": int(original_count - eroded_count),
        }
    return eroded_by_camera, debug_by_camera


def _variant_rows() -> list[dict[str, str]]:
    return [
        {"key": "native", "row_header": "Native depth", "summary_label": "native_depth"},
        {"key": "ffs_original", "row_header": "Original FFS", "summary_label": "original_ffs"},
        {"key": "fused", "row_header": "Fused depth", "summary_label": "native_ffs_fused"},
    ]


def build_native_ffs_fused_pcd_board(
    *,
    round_label: str,
    frame_idx: int,
    model_config: dict[str, Any],
    column_headers: list[str],
    variant_rows: list[dict[str, str]],
    rendered_rows: list[list[np.ndarray]],
) -> np.ndarray:
    if len(variant_rows) != 3 or len(rendered_rows) != 3:
        raise ValueError("Native/FFS fused PCD board requires exactly 3 rows.")
    for row in rendered_rows:
        if len(row) != 3:
            raise ValueError("Native/FFS fused PCD board requires exactly 3 columns per row.")
    postprocess_label = "none"
    if bool(model_config.get("phystwin_like_postprocess_enabled", False)):
        postprocess_label = (
            f"PhysTwin radius {float(model_config.get('phystwin_radius_m', 0.0)):.3f}m/"
            f"{int(model_config.get('phystwin_nb_points', 0))}nn"
        )
    return compose_registration_matrix_board(
        title_lines=[
            f"Static Object PCD Native/FFS/Fused | {round_label} | frame {int(frame_idx)}",
            (
                f"rows=native/original FFS/native-FFS fused | object_mask="
                f"{str(model_config['object_mask_enabled']).lower()} "
                f"erode={int(model_config['mask_erode_pixels'])}px | post={postprocess_label}"
            ),
            (
                "fused rule: use FFS only where native depth is missing | "
                f"depth=[{float(model_config['depth_min_m']):.2f},{float(model_config['depth_max_m']):.2f}]m"
            ),
        ],
        row_headers=[str(item["row_header"]) for item in variant_rows],
        column_headers=column_headers,
        image_rows=rendered_rows,
    )


def _load_depth_m(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    camera_idx: int,
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, depth_m, depth_info = load_depth_frame(
        case_dir=case_dir,
        metadata=metadata,
        camera_idx=int(camera_idx),
        frame_idx=int(frame_idx),
        depth_source=depth_source,
        use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
    )
    return np.asarray(depth_m, dtype=np.float32), dict(depth_info)


def run_native_ffs_fused_pcd_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    frame_idx: int = 0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    point_size: float = 2.0,
    look_distance: float = 1.0,
    tile_width: int = 480,
    tile_height: int = 360,
    max_points_per_camera: int | None = 80_000,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    use_object_mask: bool = True,
    mask_erode_pixels: int = 1,
    phystwin_like_postprocess: bool = True,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    use_float_ffs_depth_when_available: bool = True,
    round_specs: list[dict[str, Any]] | None = None,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if float(depth_max_m) <= float(depth_min_m):
        raise ValueError(f"depth_max_m must be greater than depth_min_m. Got {depth_min_m}, {depth_max_m}.")
    if int(mask_erode_pixels) < 0:
        raise ValueError(f"mask_erode_pixels must be >= 0, got {mask_erode_pixels}.")
    if float(phystwin_radius_m) <= 0.0:
        raise ValueError(f"phystwin_radius_m must be positive, got {phystwin_radius_m}.")
    if int(phystwin_nb_points) < 1:
        raise ValueError(f"phystwin_nb_points must be >= 1, got {phystwin_nb_points}.")

    selected_round_specs = (
        build_static_native_ffs_fused_pcd_round_specs(aligned_root=aligned_root)
        if round_specs is None
        else [dict(item) for item in round_specs]
    )
    render_frame_fn = render_frame_fn or _render_open3d_offscreen_pinhole
    variant_rows = _variant_rows()
    model_config = {
        "fusion_mode": "missing_only",
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "point_size": float(point_size),
        "look_distance": float(look_distance),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "max_points_per_camera": None if max_points_per_camera is None else int(max_points_per_camera),
        "text_prompt": str(text_prompt),
        "object_mask_enabled": bool(use_object_mask),
        "mask_erode_pixels": int(mask_erode_pixels),
        "phystwin_like_postprocess_enabled": bool(phystwin_like_postprocess),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
    }

    rounds_summary: list[dict[str, Any]] = []
    for round_spec in selected_round_specs:
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

        camera_ids = list(range(len(native_metadata["serial_numbers"])))
        if len(camera_ids) != 3:
            raise ValueError(f"Expected exactly 3 native cameras for {round_spec['round_id']}, got {camera_ids}.")
        if len(ffs_metadata["serial_numbers"]) != len(camera_ids):
            raise ValueError(
                f"Native and FFS camera counts differ for {round_spec['round_id']}: "
                f"{len(camera_ids)} vs {len(ffs_metadata['serial_numbers'])}."
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
        raw_mask_debug: dict[int, dict[str, Any]] = {}
        erode_debug: dict[int, dict[str, Any]] = {}
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
            raw_masks, raw_mask_debug = load_union_masks_for_camera_clouds(
                mask_root=mask_root,
                camera_clouds=minimal_clouds,
                frame_token=str(selected_frame_idx),
                text_prompt=str(text_prompt),
            )
            mask_by_camera, erode_debug = _erode_object_masks(
                raw_masks,
                erode_pixels=int(mask_erode_pixels),
            )

        camera_clouds_by_variant: dict[str, list[dict[str, Any]]] = {str(item["key"]): [] for item in variant_rows}
        per_variant_camera: dict[str, list[dict[str, Any]]] = {str(item["key"]): [] for item in variant_rows}

        for camera_idx in camera_ids:
            camera_mask = mask_by_camera.get(int(camera_idx)) if use_object_mask else None
            native_color_path = native_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            ffs_color_path = ffs_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            native_color_image = _load_color_image(native_color_path)
            ffs_color_image = _load_color_image(ffs_color_path)
            native_depth_m, native_depth_info = _load_depth_m(
                case_dir=native_case_dir,
                metadata=native_metadata,
                camera_idx=int(camera_idx),
                frame_idx=selected_frame_idx,
                depth_source="realsense",
                use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
            )
            ffs_depth_m, ffs_depth_info = _load_depth_m(
                case_dir=ffs_case_dir,
                metadata=ffs_metadata,
                camera_idx=int(camera_idx),
                frame_idx=selected_frame_idx,
                depth_source="ffs_raw",
                use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
            )
            fused_depth_m, fusion_stats = fuse_native_ffs_depth(
                native_depth_m,
                ffs_depth_m,
            )

            variant_inputs = {
                "native": {
                    "depth_m": native_depth_m,
                    "color_image": native_color_image,
                    "color_path": native_color_path,
                    "K_color": np.asarray(native_metadata["K_color"][camera_idx], dtype=np.float32),
                    "c2w": np.asarray(native_c2w_list[camera_idx], dtype=np.float32),
                    "depth_info": native_depth_info,
                    "source": "aligned_native_depth",
                    "serial": str(native_metadata["serial_numbers"][camera_idx]),
                },
                "ffs_original": {
                    "depth_m": ffs_depth_m,
                    "color_image": ffs_color_image,
                    "color_path": ffs_color_path,
                    "K_color": np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32),
                    "c2w": np.asarray(ffs_c2w_list[camera_idx], dtype=np.float32),
                    "depth_info": ffs_depth_info,
                    "source": "aligned_original_ffs_depth",
                    "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                },
                "fused": {
                    "depth_m": fused_depth_m,
                    "color_image": native_color_image,
                    "color_path": native_color_path,
                    "K_color": np.asarray(native_metadata["K_color"][camera_idx], dtype=np.float32),
                    "c2w": np.asarray(native_c2w_list[camera_idx], dtype=np.float32),
                    "depth_info": {"source_depth_dir_used": "in_memory_native_ffs_fused"},
                    "source": "in_memory_native_ffs_fused_depth",
                    "serial": str(native_metadata["serial_numbers"][camera_idx]),
                },
            }
            for variant_key, variant_input in variant_inputs.items():
                cloud = _build_world_cloud(
                    depth_m=np.asarray(variant_input["depth_m"], dtype=np.float32),
                    color_image=np.asarray(variant_input["color_image"], dtype=np.uint8),
                    K_color=np.asarray(variant_input["K_color"], dtype=np.float32),
                    c2w=np.asarray(variant_input["c2w"], dtype=np.float32),
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    max_points_per_camera=max_points_per_camera,
                    object_mask=camera_mask,
                )
                camera_clouds_by_variant[variant_key].append(cloud)
                per_variant_camera[variant_key].append(
                    {
                        "camera_idx": int(camera_idx),
                        "serial": str(variant_input["serial"]),
                        "source": str(variant_input["source"]),
                        "color_path": str(Path(variant_input["color_path"]).resolve()),
                        "depth_info": dict(variant_input["depth_info"]),
                        "object_mask_enabled": bool(use_object_mask),
                        "mask_erode_pixels": int(mask_erode_pixels),
                        "mask_pixel_count": int(np.count_nonzero(camera_mask)) if camera_mask is not None else 0,
                        "point_count": int(len(cloud["points"])),
                        "fusion": dict(fusion_stats) if variant_key == "fused" else None,
                        **dict(cloud["stats"]),
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
                    render_kind="native_ffs_fused_pcd_compare",
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

        board = build_native_ffs_fused_pcd_board(
            round_label=str(round_spec["round_label"]),
            frame_idx=selected_frame_idx,
            model_config=model_config,
            column_headers=column_headers,
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )
        board_path = round_output_dir / f"native_ffs_missing_fused_object_pcd_3x3_frame_{selected_frame_idx:04d}.png"
        write_image(board_path, board)

        round_summary = {
            "round_id": str(round_spec["round_id"]),
            "round_label": str(round_spec["round_label"]),
            "native_case_ref": str(round_spec["native_case_ref"]),
            "ffs_case_ref": str(round_spec["ffs_case_ref"]),
            "native_case_dir": str(native_case_dir.resolve()),
            "ffs_case_dir": str(ffs_case_dir.resolve()),
            "frame_idx": int(selected_frame_idx),
            "row_headers": [str(item["row_header"]) for item in variant_rows],
            "variant_rows": variant_rows,
            "column_headers": column_headers,
            "board_path": str(board_path.resolve()),
            "model_config": dict(model_config),
            "mask_root": str(Path(round_spec["mask_root"]).resolve()) if use_object_mask else None,
            "mask_debug": {
                str(camera_idx): {
                    **dict(raw_mask_debug.get(camera_idx, {})),
                    **dict(erode_debug.get(camera_idx, {})),
                }
                for camera_idx in camera_ids
            },
            "mask_contract": {
                "object_masked": bool(use_object_mask),
                "mask_source": "static_sam31_ffs_masks",
                "text_prompt": str(text_prompt),
                "erode_pixels": int(mask_erode_pixels),
                "erode_kernel": "3x3_ones",
            },
            "fusion_rule": {
                "mode": "native_unless_missing_else_ffs",
                "operation_order": "fuse_raw_aligned_depth_then_object_mask_then_fuse_cameras_then_display_postprocess",
            },
            "render_contract": {
                "renderer": "open3d_offscreen_renderer",
                "projection_mode": "original_camera_pinhole",
                "columns": "original_ffs_camera_views",
                "rows": "native_depth_original_ffs_native_ffs_fused",
                "display_postprocess": "phystwin_like_radius_neighbor_filter" if phystwin_like_postprocess else "none",
                "display_postprocess_applied_to": "fused_object_pcd_rows_before_rendering",
                "object_masked": bool(use_object_mask),
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
            "per_variant_camera": per_variant_camera,
            "render_summary": render_summary,
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
