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
from ..io_artifacts import write_image, write_json
from ..io_case import decode_depth_to_meters, load_case_metadata, resolve_case_dir
from ..layouts import compose_registration_matrix_board
from .ffs_confidence_filter_pcd_compare import (
    CONFIDENCE_FILTER_MODES,
    _apply_phystwin_like_radius_postprocess,
    _build_view_configs,
    _build_world_cloud,
    _confidence_stats,
    _depth_scale_for_camera,
    _format_point_count,
    _fuse_camera_clouds,
    _load_color_image,
    _load_depth_m_from_depth_dir,
    _load_ir_image,
    _serialize_view_config,
    build_static_confidence_filter_round_specs,
)
from .ffs_confidence_panels import DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT
from .ffs_confidence_pcd_panels import _render_open3d_offscreen_pinhole
from ..workflows.masked_pointcloud_compare import PHYSTWIN_DATA_PROCESS_MASK_CONTRACT, load_union_masks_for_camera_clouds


DEFAULT_CONFIDENCE_SWEEP_THRESHOLDS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)


def parse_confidence_thresholds(thresholds: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(thresholds, str):
        values = [float(item.strip()) for item in thresholds.split(",") if item.strip()]
    else:
        values = [float(item) for item in thresholds]
    if not values:
        raise ValueError("At least one confidence threshold is required.")
    for threshold in values:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"confidence threshold must be in [0, 1], got {threshold}.")
    keys = [_threshold_key(threshold) for threshold in values]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Thresholds must be unique after two-decimal formatting, got {values}.")
    return values


def _threshold_key(threshold: float) -> str:
    return f"threshold_{float(threshold):.2f}".replace(".", "_")


def _variant_rows(threshold: float) -> list[dict[str, str]]:
    suffix = f"{float(threshold):.2f}"
    return [
        {"key": "native", "row_header": "RealSense native depth", "summary_label": "native_depth"},
        {
            "key": "ffs_original",
            "row_header": "Fast-FoundationStereo depth\nno confidence filter",
            "summary_label": "ffs_rerun_unfiltered",
        },
        {
            "key": "ffs_margin",
            "row_header": f"Fast-FoundationStereo depth\nmargin confidence >= {suffix}",
            "summary_label": "ffs_margin_filtered",
        },
        {
            "key": "ffs_max_softmax",
            "row_header": f"Fast-FoundationStereo depth\nmaximum softmax confidence >= {suffix}",
            "summary_label": "ffs_max_softmax_filtered",
        },
        {
            "key": "ffs_entropy",
            "row_header": f"Fast-FoundationStereo depth\nentropy confidence >= {suffix}",
            "summary_label": "ffs_entropy_filtered",
        },
        {
            "key": "ffs_variance",
            "row_header": f"Fast-FoundationStereo depth\nvariance confidence >= {suffix}",
            "summary_label": "ffs_variance_filtered",
        },
    ]


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


def build_confidence_threshold_sweep_pcd_board(
    *,
    round_label: str,
    frame_idx: int,
    confidence_threshold: float,
    model_config: dict[str, Any],
    column_headers: list[str],
    variant_rows: list[dict[str, str]],
    rendered_rows: list[list[np.ndarray]],
) -> np.ndarray:
    postprocess_label = "none"
    if bool(model_config.get("phystwin_like_postprocess_enabled", False)):
        postprocess_label = (
            f"PhysTwin-like radius-neighbor filter: {float(model_config.get('phystwin_radius_m', 0.0)):.3f}m, "
            f"{int(model_config.get('phystwin_nb_points', 0))} neighbors"
        )
    return compose_registration_matrix_board(
        title_lines=[
            (
                f"Static Object Point Cloud Confidence Sweep | {round_label} | frame {int(frame_idx)} | "
                f"keep confidence >= {float(confidence_threshold):.2f}"
            ),
            (
                f"rows=RealSense native / Fast-FoundationStereo no confidence filter / four confidence filters | "
                f"object_mask={str(model_config['object_mask_enabled']).lower()} "
                f"erode={int(model_config['mask_erode_pixels'])}px | post={postprocess_label}"
            ),
            (
                f"scale={float(model_config['scale']):.2f} | iters={int(model_config['valid_iters'])} | "
                f"disp={int(model_config['max_disp'])} | depth=[{float(model_config['depth_min_m']):.2f},"
                f"{float(model_config['depth_max_m']):.2f}]m"
            ),
        ],
        row_headers=[str(item["row_header"]) for item in variant_rows],
        column_headers=column_headers,
        image_rows=rendered_rows,
    )


def _empty_camera_variant_map(thresholds: list[float]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        _threshold_key(threshold): {str(row["key"]): [] for row in _variant_rows(threshold)}
        for threshold in thresholds
    }


def run_ffs_confidence_threshold_sweep_pcd_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    ffs_repo: str | Path,
    model_path: str | Path,
    scale: float = 1.0,
    valid_iters: int = 4,
    max_disp: int = 192,
    frame_idx: int = 0,
    thresholds: str | list[float] | tuple[float, ...] = DEFAULT_CONFIDENCE_SWEEP_THRESHOLDS,
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
    phystwin_radius_m: float = float(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["radius_m"]),
    phystwin_nb_points: int = int(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT["nb_points"]),
    round_specs: list[dict[str, Any]] | None = None,
    runner_factory: Callable[..., Any] | None = None,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    threshold_values = parse_confidence_thresholds(thresholds)
    round_specs = build_static_confidence_filter_round_specs(aligned_root=aligned_root) if round_specs is None else list(round_specs)
    if int(mask_erode_pixels) < 0:
        raise ValueError(f"mask_erode_pixels must be >= 0, got {mask_erode_pixels}.")
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
        "mask_erode_pixels": int(mask_erode_pixels),
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
            raw_mask_by_camera, raw_mask_debug = load_union_masks_for_camera_clouds(
                mask_root=mask_root,
                camera_clouds=minimal_clouds,
                frame_token=str(selected_frame_idx),
                text_prompt=str(text_prompt),
            )
            mask_by_camera, erode_debug = _erode_object_masks(
                raw_mask_by_camera,
                erode_pixels=int(mask_erode_pixels),
            )

        camera_clouds_by_threshold = _empty_camera_variant_map(threshold_values)
        per_variant_camera_by_threshold = _empty_camera_variant_map(threshold_values)

        for camera_idx in camera_ids:
            camera_mask = mask_by_camera.get(int(camera_idx)) if use_object_mask else None
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
                object_mask=camera_mask,
            )
            native_camera_stats = {
                "camera_idx": int(camera_idx),
                "serial": str(native_metadata["serial_numbers"][camera_idx]),
                "color_path": str(native_color_path.resolve()),
                "depth_path": native_depth_path,
                "object_mask_enabled": bool(use_object_mask),
                "mask_erode_pixels": int(mask_erode_pixels),
                "mask_pixel_count": int(np.count_nonzero(camera_mask)) if camera_mask is not None else 0,
                "point_count": int(len(native_cloud["points"])),
                **dict(native_cloud["stats"]),
            }

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
                object_mask=camera_mask,
            )
            depth_valid_mask = np.isfinite(depth_color_m) & (depth_color_m > 0.0)
            confidence_stats_mask = depth_valid_mask if camera_mask is None else (depth_valid_mask & np.asarray(camera_mask, dtype=bool))
            ffs_camera_stats = {
                "camera_idx": int(camera_idx),
                "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                "color_path": str(ffs_color_path.resolve()),
                "ir_left_path": str(ir_left_path.resolve()),
                "ir_right_path": str(ir_right_path.resolve()),
                "source": "pyTorch_ffs_rerun_unfiltered",
                "object_mask_enabled": bool(use_object_mask),
                "mask_erode_pixels": int(mask_erode_pixels),
                "mask_pixel_count": int(np.count_nonzero(camera_mask)) if camera_mask is not None else 0,
                "point_count": int(len(ffs_cloud["points"])),
                "aligned_valid_pixel_count": int(np.count_nonzero(depth_valid_mask)),
                "masked_valid_pixel_count": int(np.count_nonzero(confidence_stats_mask)),
                **dict(ffs_cloud["stats"]),
            }

            confidence_color_by_mode: dict[str, np.ndarray] = {}
            for mode in CONFIDENCE_FILTER_MODES:
                confidence_ir = np.asarray(ffs_output[f"confidence_{mode}_ir_left"], dtype=np.float32)
                confidence_color_by_mode[mode] = align_ir_scalar_to_color(
                    depth_ir_left_m,
                    confidence_ir,
                    K_ir_left_used,
                    T_ir_left_to_color,
                    K_color,
                    output_shape=output_shape,
                    invalid_value=0.0,
                )

            depth_scale = _depth_scale_for_camera(ffs_metadata, int(camera_idx))
            for threshold in threshold_values:
                threshold_key = _threshold_key(threshold)
                camera_clouds_by_threshold[threshold_key]["native"].append(native_cloud)
                camera_clouds_by_threshold[threshold_key]["ffs_original"].append(ffs_cloud)
                per_variant_camera_by_threshold[threshold_key]["native"].append(dict(native_camera_stats))
                per_variant_camera_by_threshold[threshold_key]["ffs_original"].append(dict(ffs_camera_stats))

                for mode in CONFIDENCE_FILTER_MODES:
                    confidence_color = confidence_color_by_mode[mode]
                    filter_output = build_confidence_filtered_depth_uint16(
                        depth_m=depth_color_m,
                        confidence=confidence_color,
                        confidence_threshold=float(threshold),
                        depth_scale_m_per_unit=float(depth_scale),
                        depth_min_m=float(depth_min_m),
                        depth_max_m=float(depth_max_m),
                        object_mask=camera_mask,
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
                    camera_clouds_by_threshold[threshold_key][variant_key].append(filtered_cloud)
                    per_variant_camera_by_threshold[threshold_key][variant_key].append(
                        {
                            "camera_idx": int(camera_idx),
                            "serial": str(ffs_metadata["serial_numbers"][camera_idx]),
                            "source": "pyTorch_ffs_confidence_filtered",
                            "confidence_mode": str(mode),
                            "confidence_threshold": float(threshold),
                            "depth_scale_m_per_unit": float(depth_scale),
                            "object_mask_enabled": bool(use_object_mask),
                            "mask_erode_pixels": int(mask_erode_pixels),
                            "mask_pixel_count": int(np.count_nonzero(camera_mask)) if camera_mask is not None else 0,
                            "point_count": int(len(filtered_cloud["points"])),
                            "confidence": _confidence_stats(confidence_color, valid_depth=confidence_stats_mask),
                            "filter_stats": dict(filter_output["stats"]),
                            **dict(filtered_cloud["stats"]),
                        }
                    )

        threshold_summaries: list[dict[str, Any]] = []
        for threshold in threshold_values:
            threshold_key = _threshold_key(threshold)
            variant_rows = _variant_rows(threshold)
            fused_by_variant: dict[str, dict[str, Any]] = {}
            for variant in variant_rows:
                variant_key = str(variant["key"])
                raw_points, raw_colors = _fuse_camera_clouds(camera_clouds_by_threshold[threshold_key][variant_key])
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
                        render_kind="confidence_threshold_sweep_pcd_compare",
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

            board = build_confidence_threshold_sweep_pcd_board(
                round_label=str(round_spec["round_label"]),
                frame_idx=selected_frame_idx,
                confidence_threshold=float(threshold),
                model_config=model_config,
                column_headers=column_headers,
                variant_rows=variant_rows,
                rendered_rows=rendered_rows,
            )
            threshold_output_dir = round_output_dir / threshold_key
            threshold_output_dir.mkdir(parents=True, exist_ok=True)
            board_path = threshold_output_dir / (
                f"confidence_threshold_sweep_pcd_6x3_frame_{selected_frame_idx:04d}_"
                f"{threshold_key}_mask_erode_{int(mask_erode_pixels)}px_phystwin.png"
            )
            write_image(board_path, board)

            threshold_summary = {
                "round_id": str(round_spec["round_id"]),
                "round_label": str(round_spec["round_label"]),
                "frame_idx": int(selected_frame_idx),
                "threshold_key": threshold_key,
                "confidence_threshold": float(threshold),
                "confidence_modes": list(CONFIDENCE_FILTER_MODES),
                "row_headers": [str(item["row_header"]) for item in variant_rows],
                "variant_rows": variant_rows,
                "column_headers": column_headers,
                "board_path": str(board_path.resolve()),
                "model_config": dict(model_config),
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
                    "mask_erode_pixels": int(mask_erode_pixels),
                    "formal_depth_written": False,
                },
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
                "per_variant_camera": per_variant_camera_by_threshold[threshold_key],
                "render_summary": render_summary,
                "output_dir": str(threshold_output_dir.resolve()),
            }
            write_json(threshold_output_dir / "summary.json", threshold_summary)
            threshold_summaries.append(threshold_summary)

        round_summary = {
            "round_id": str(round_spec["round_id"]),
            "round_label": str(round_spec["round_label"]),
            "native_case_ref": str(round_spec["native_case_ref"]),
            "ffs_case_ref": str(round_spec["ffs_case_ref"]),
            "native_case_dir": str(native_case_dir.resolve()),
            "ffs_case_dir": str(ffs_case_dir.resolve()),
            "frame_idx": int(selected_frame_idx),
            "thresholds": [float(item) for item in threshold_values],
            "confidence_modes": list(CONFIDENCE_FILTER_MODES),
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
            "postprocess": {
                "enabled": bool(phystwin_like_postprocess),
                "mode": "phystwin_like_radius_neighbor_filter" if phystwin_like_postprocess else "none",
                "radius_m": float(phystwin_radius_m),
                "nb_points": int(phystwin_nb_points),
                "applied_to": "fused_object_pcd_rows_before_rendering",
                "reference_contract": dict(PHYSTWIN_DATA_PROCESS_MASK_CONTRACT),
            },
            "column_views": [_serialize_view_config(view_config) for view_config in view_configs],
            "threshold_summaries": threshold_summaries,
            "output_dir": str(round_output_dir.resolve()),
        }
        write_json(round_output_dir / "summary.json", round_summary)
        rounds_summary.append(round_summary)

    manifest = {
        "output_dir": str(output_root.resolve()),
        "frame_idx": int(frame_idx),
        "thresholds": [float(item) for item in threshold_values],
        "confidence_modes": list(CONFIDENCE_FILTER_MODES),
        "model_config": dict(model_config),
        "rounds": rounds_summary,
    }
    write_json(output_root / "summary.json", manifest)
    return manifest
