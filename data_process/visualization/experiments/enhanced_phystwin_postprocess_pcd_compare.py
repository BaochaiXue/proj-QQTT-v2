from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..calibration_io import load_calibration_transforms
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_metadata, resolve_case_dir
from ..layouts import compose_registration_matrix_board
from ..workflows.masked_pointcloud_compare import PHYSTWIN_DATA_PROCESS_MASK_CONTRACT, load_union_masks_for_camera_clouds
from .ffs_confidence_filter_pcd_compare import (
    _apply_enhanced_phystwin_like_postprocess,
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
from .native_ffs_fused_pcd_compare import (
    DEFAULT_PHYSTWIN_NB_POINTS,
    DEFAULT_PHYSTWIN_RADIUS_M,
    _erode_object_masks,
    _load_depth_m,
    build_static_native_ffs_fused_pcd_round_specs,
)


DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M = 0.01
DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M = 0.0
DEFAULT_ROW_LABEL_WIDTH = 500


def _variant_rows() -> list[dict[str, str]]:
    return [
        {"key": "native_raw", "source": "native", "postprocess": "none", "row_header": "RealSense native depth\nno postprocessing"},
        {
            "key": "native_pt",
            "source": "native",
            "postprocess": "phystwin_like",
            "row_header": "RealSense native depth\nPhysTwin-like radius-neighbor filter",
        },
        {
            "key": "native_enhanced",
            "source": "native",
            "postprocess": "enhanced",
            "row_header": "RealSense native depth\nenhanced radius + component filter",
        },
        {
            "key": "ffs_raw",
            "source": "ffs",
            "postprocess": "none",
            "row_header": "Fast-FoundationStereo depth\nno postprocessing",
        },
        {
            "key": "ffs_pt",
            "source": "ffs",
            "postprocess": "phystwin_like",
            "row_header": "Fast-FoundationStereo depth\nPhysTwin-like radius-neighbor filter",
        },
        {
            "key": "ffs_enhanced",
            "source": "ffs",
            "postprocess": "enhanced",
            "row_header": "Fast-FoundationStereo depth\nenhanced radius + component filter",
        },
    ]


def _no_postprocess_stats(point_count: int) -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "none",
        "input_point_count": int(point_count),
        "output_point_count": int(point_count),
    }


def _apply_variant_postprocess(
    *,
    variant: dict[str, str],
    points: np.ndarray,
    colors: np.ndarray,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(variant["postprocess"])
    if mode == "none":
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        return point_array, color_array, _no_postprocess_stats(len(point_array))
    if mode == "phystwin_like":
        return _apply_phystwin_like_radius_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
        )
    if mode == "enhanced":
        return _apply_enhanced_phystwin_like_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
            component_voxel_size_m=float(enhanced_component_voxel_size_m),
            keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
    raise ValueError(f"Unsupported enhanced postprocess variant mode: {mode}")


def build_enhanced_phystwin_postprocess_pcd_board(
    *,
    round_label: str,
    frame_idx: int,
    model_config: dict[str, Any],
    column_headers: list[str],
    variant_rows: list[dict[str, str]],
    rendered_rows: list[list[np.ndarray]],
) -> np.ndarray:
    if len(variant_rows) != 6 or len(rendered_rows) != 6:
        raise ValueError("Enhanced PhysTwin postprocess board requires exactly 6 rows.")
    for row in rendered_rows:
        if len(row) != 3:
            raise ValueError("Enhanced PhysTwin postprocess board requires exactly 3 columns per row.")
    return compose_registration_matrix_board(
        title_lines=[
            f"Static Object Point Cloud Enhanced PhysTwin-like Postprocess | {round_label} | frame {int(frame_idx)}",
            (
                "rows=RealSense and Fast-FoundationStereo, each with no filter / radius-neighbor filter / enhanced component filter | "
                f"object_mask={str(model_config['object_mask_enabled']).lower()} | "
                f"mask_erode={int(model_config['mask_erode_pixels'])}px"
            ),
            (
                f"radius filter={float(model_config['phystwin_radius_m']):.3f}m, "
                f"{int(model_config['phystwin_nb_points'])} neighbors | "
                f"component voxel={float(model_config['enhanced_component_voxel_size_m']):.3f}m | "
                f"keep_gap={float(model_config['enhanced_keep_near_main_gap_m']):.3f}m"
            ),
        ],
        row_headers=[str(item["row_header"]) for item in variant_rows],
        column_headers=column_headers,
        image_rows=rendered_rows,
        row_label_width=int(model_config.get("row_label_width", DEFAULT_ROW_LABEL_WIDTH)),
    )


def run_enhanced_phystwin_postprocess_pcd_workflow(
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
    row_label_width: int = DEFAULT_ROW_LABEL_WIDTH,
    max_points_per_camera: int | None = 80_000,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    use_object_mask: bool = True,
    mask_erode_pixels: int = 0,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
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
    if float(enhanced_component_voxel_size_m) <= 0.0:
        raise ValueError(f"enhanced_component_voxel_size_m must be positive, got {enhanced_component_voxel_size_m}.")
    if float(enhanced_keep_near_main_gap_m) < 0.0:
        raise ValueError(f"enhanced_keep_near_main_gap_m must be >= 0, got {enhanced_keep_near_main_gap_m}.")

    selected_round_specs = (
        build_static_native_ffs_fused_pcd_round_specs(aligned_root=aligned_root)
        if round_specs is None
        else [dict(item) for item in round_specs]
    )
    render_frame_fn = render_frame_fn or _render_open3d_offscreen_pinhole
    variant_rows = _variant_rows()
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
        "object_mask_enabled": bool(use_object_mask),
        "mask_erode_pixels": int(mask_erode_pixels),
        "phystwin_radius_m": float(phystwin_radius_m),
        "phystwin_nb_points": int(phystwin_nb_points),
        "enhanced_component_voxel_size_m": float(enhanced_component_voxel_size_m),
        "enhanced_keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
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

        masks_by_camera: dict[int, np.ndarray] = {}
        raw_mask_debug: dict[int, dict[str, Any]] = {}
        erode_debug: dict[int, dict[str, Any]] = {}
        if use_object_mask:
            minimal_clouds = [
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                    "color_path": str(ffs_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"),
                }
                for camera_idx in camera_ids
            ]
            raw_masks, raw_mask_debug = load_union_masks_for_camera_clouds(
                mask_root=Path(round_spec["mask_root"]).resolve(),
                camera_clouds=minimal_clouds,
                frame_token=str(selected_frame_idx),
                text_prompt=str(text_prompt),
            )
            if int(mask_erode_pixels) > 0:
                masks_by_camera, erode_debug = _erode_object_masks(raw_masks, erode_pixels=int(mask_erode_pixels))
            else:
                masks_by_camera = {int(camera_idx): np.asarray(mask, dtype=bool).copy() for camera_idx, mask in raw_masks.items()}

        camera_clouds_by_source: dict[str, list[dict[str, Any]]] = {"native": [], "ffs": []}
        per_source_camera: dict[str, list[dict[str, Any]]] = {"native": [], "ffs": []}
        for camera_idx in camera_ids:
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
            mask = masks_by_camera.get(int(camera_idx)) if use_object_mask else None
            native_cloud = _build_world_cloud(
                depth_m=native_depth_m,
                color_image=native_color_image,
                K_color=np.asarray(native_metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(native_c2w_list[camera_idx], dtype=np.float32),
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                max_points_per_camera=max_points_per_camera,
                object_mask=mask,
            )
            ffs_cloud = _build_world_cloud(
                depth_m=ffs_depth_m,
                color_image=ffs_color_image,
                K_color=np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32),
                c2w=np.asarray(ffs_c2w_list[camera_idx], dtype=np.float32),
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                max_points_per_camera=max_points_per_camera,
                object_mask=mask,
            )
            camera_clouds_by_source["native"].append(native_cloud)
            camera_clouds_by_source["ffs"].append(ffs_cloud)
            per_source_camera["native"].append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(native_metadata["serial_numbers"][camera_idx]),
                    "source": "aligned_native_depth",
                    "color_path": str(native_color_path.resolve()),
                    "depth_info": dict(native_depth_info),
                    "object_mask_enabled": bool(use_object_mask),
                    "mask_pixel_count": int(np.count_nonzero(mask)) if mask is not None else 0,
                    "point_count": int(len(native_cloud["points"])),
                    **dict(native_cloud["stats"]),
                }
            )
            per_source_camera["ffs"].append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                    "source": "aligned_original_ffs_depth",
                    "color_path": str(ffs_color_path.resolve()),
                    "depth_info": dict(ffs_depth_info),
                    "object_mask_enabled": bool(use_object_mask),
                    "mask_pixel_count": int(np.count_nonzero(mask)) if mask is not None else 0,
                    "point_count": int(len(ffs_cloud["points"])),
                    **dict(ffs_cloud["stats"]),
                }
            )

        fused_source: dict[str, tuple[np.ndarray, np.ndarray]] = {
            source: _fuse_camera_clouds(clouds) for source, clouds in camera_clouds_by_source.items()
        }
        fused_by_variant: dict[str, dict[str, Any]] = {}
        for variant in variant_rows:
            raw_points, raw_colors = fused_source[str(variant["source"])]
            points, colors, postprocess_stats = _apply_variant_postprocess(
                variant=variant,
                points=raw_points,
                colors=raw_colors,
                phystwin_radius_m=float(phystwin_radius_m),
                phystwin_nb_points=int(phystwin_nb_points),
                enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
            )
            fused_by_variant[str(variant["key"])] = {
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
                    render_kind="enhanced_phystwin_postprocess_pcd_compare",
                    metric_name=variant_key,
                    camera_idx=int(view_config["camera_idx"]),
                )
                rendered = label_tile(rendered, _format_point_count(int(fused["point_count"])), (int(tile_width), int(tile_height)))
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

        board = build_enhanced_phystwin_postprocess_pcd_board(
            round_label=str(round_spec["round_label"]),
            frame_idx=selected_frame_idx,
            model_config=model_config,
            column_headers=column_headers,
            variant_rows=variant_rows,
            rendered_rows=rendered_rows,
        )
        board_path = round_output_dir / f"enhanced_phystwin_postprocess_object_pcd_6x3_frame_{selected_frame_idx:04d}.png"
        write_image(board_path, board)
        mask_debug = {
            str(camera_idx): {
                "raw": dict(raw_mask_debug.get(camera_idx, {})),
                "eroded": dict(erode_debug.get(camera_idx, {})),
            }
            for camera_idx in camera_ids
        }
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
            "mask_debug": mask_debug,
            "mask_contract": {
                "object_masked": bool(use_object_mask),
                "mask_source": "static_sam31_ffs_masks",
                "text_prompt": str(text_prompt),
                "mask_erode_pixels": int(mask_erode_pixels),
                "erode_kernel": "3x3_ones",
            },
            "render_contract": {
                "renderer": "open3d_offscreen_renderer",
                "projection_mode": "original_camera_pinhole",
                "columns": "original_ffs_camera_views",
                "rows": "native_none_pt_enhanced_ffs_none_pt_enhanced",
                "display_postprocess": "none_vs_phystwin_like_vs_enhanced_phystwin_like",
                "object_masked": bool(use_object_mask),
                "formal_depth_written": False,
            },
            "column_views": [_serialize_view_config(view_config) for view_config in view_configs],
            "fused_point_counts": {str(key): int(value["point_count"]) for key, value in fused_by_variant.items()},
            "fused_point_counts_before_postprocess": {
                str(key): int(value["point_count_before_postprocess"]) for key, value in fused_by_variant.items()
            },
            "postprocess_stats_by_variant": {str(key): dict(value["postprocess"]) for key, value in fused_by_variant.items()},
            "per_source_camera": per_source_camera,
            "render_summary": render_summary,
            "postprocess": {
                "phystwin_like": {
                    "mode": "phystwin_like_radius_neighbor_filter",
                    "radius_m": float(phystwin_radius_m),
                    "nb_points": int(phystwin_nb_points),
                    "reference_contract": dict(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT),
                },
                "enhanced_phystwin_like": {
                    "mode": "enhanced_phystwin_like_radius_then_component_filter",
                    "radius_m": float(phystwin_radius_m),
                    "nb_points": int(phystwin_nb_points),
                    "component_voxel_size_m": float(enhanced_component_voxel_size_m),
                    "keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
                },
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
