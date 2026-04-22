from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from data_process.depth_backends import (
    FastFoundationStereoRunner,
    align_depth_to_color,
    align_ir_scalar_to_color,
)
from ..calibration_io import load_calibration_transforms
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import depth_to_camera_points, load_case_metadata, resolve_case_dir, transform_points
from ..layouts import compose_registration_matrix_board, overlay_scalar_colorbar
from ..roi import crop_points_to_bounds
from ..views import build_original_camera_view_configs
from .ffs_confidence_panels import (
    DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    build_static_confidence_round_specs,
)
from .masked_camera_view_compare import _image_size_from_color_path, _mask_rgb_image, _scale_intrinsic_matrix
from .masked_pointcloud_compare import MIN_MASKED_POINT_COUNT_FOR_FOCUS, _expand_bounds, load_union_masks_for_camera_clouds


DEFAULT_CONFIDENCE_PCD_METRICS: tuple[str, ...] = ("margin", "max_softmax")


def build_static_confidence_pcd_round_specs(
    *,
    aligned_root: Path,
) -> list[dict[str, Any]]:
    return build_static_confidence_round_specs(aligned_root=Path(aligned_root).resolve())


def _resolve_metric_names(metrics: str) -> tuple[str, ...]:
    normalized = str(metrics).strip().lower()
    if normalized == "both":
        return DEFAULT_CONFIDENCE_PCD_METRICS
    if normalized in DEFAULT_CONFIDENCE_PCD_METRICS:
        return (normalized,)
    raise ValueError(f"Unsupported confidence metrics selection: {metrics!r}")


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


def _mask_image(image: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    masked = np.zeros_like(np.asarray(image, dtype=np.uint8))
    bool_mask = np.asarray(mask, dtype=bool)
    masked[bool_mask] = np.asarray(image, dtype=np.uint8)[bool_mask]
    return masked


def _colorize_confidence_map(confidence: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float32)
    normalized = np.clip(confidence, 0.0, 1.0)
    colorized = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    if valid_mask is not None:
        colorized = np.asarray(colorized, dtype=np.uint8).copy()
        colorized[~np.asarray(valid_mask, dtype=bool)] = 0
    return colorized


def _metric_label(metric_name: str) -> str:
    if metric_name == "margin":
        return "Confidence (margin)"
    if metric_name == "max_softmax":
        return "Confidence (max_softmax)"
    raise ValueError(f"Unsupported metric_name={metric_name!r}.")


def _fuse_camera_clouds(camera_clouds: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [np.asarray(item["points"], dtype=np.float32) for item in camera_clouds if len(item["points"]) > 0]
    color_sets = [np.asarray(item["colors"], dtype=np.uint8) for item in camera_clouds if len(item["points"]) > 0]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def _render_open3d_offscreen_pinhole(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    width: int,
    height: int,
    center: np.ndarray,
    eye: np.ndarray,
    up: np.ndarray,
    zoom: float,
    point_size: float,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
    try:
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors[:, ::-1], dtype=np.float64) / 255.0)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = float(point_size)
        renderer.scene.add_geometry("pcd", pcd, material)
        renderer.setup_camera(
            np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4),
            int(width),
            int(height),
        )
        image = np.asarray(renderer.render_to_image())
    finally:
        renderer.scene.clear_geometry()

    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def _compute_focus_bounds(
    *,
    masked_points: np.ndarray,
    unmasked_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    masked_points = np.asarray(masked_points, dtype=np.float32)
    unmasked_points = np.asarray(unmasked_points, dtype=np.float32)
    if len(masked_points) >= MIN_MASKED_POINT_COUNT_FOR_FOCUS:
        return (
            masked_points.min(axis=0).astype(np.float32),
            masked_points.max(axis=0).astype(np.float32),
            "masked_ffs_fused",
            False,
        )
    if len(unmasked_points) == 0:
        raise RuntimeError("Static confidence PCD workflow could not build any FFS points.")
    return (
        unmasked_points.min(axis=0).astype(np.float32),
        unmasked_points.max(axis=0).astype(np.float32),
        "unmasked_fallback",
        True,
    )


def build_confidence_pcd_board(
    *,
    round_label: str,
    frame_idx: int,
    metric_name: str,
    model_config: dict[str, Any],
    column_headers: list[str],
    rgb_images: list[np.ndarray],
    pcd_images: list[np.ndarray],
    confidence_images: list[np.ndarray],
) -> np.ndarray:
    if len(rgb_images) != 3 or len(pcd_images) != 3 or len(confidence_images) != 3:
        raise ValueError("Confidence PCD board requires exactly 3 images per row.")

    image_rows = [
        [label_tile(image, "RGB", (image.shape[1], image.shape[0])) for image in rgb_images],
        [label_tile(image, "Masked FFS PCD", (image.shape[1], image.shape[0])) for image in pcd_images],
        [
            label_tile(
                overlay_scalar_colorbar(
                    image,
                    label="conf",
                    min_text="0.0",
                    max_text="1.0",
                    colormap=cv2.COLORMAP_VIRIDIS,
                ),
                _metric_label(metric_name),
                (image.shape[1], image.shape[0]),
            )
            for image in confidence_images
        ],
    ]
    return compose_registration_matrix_board(
        title_lines=[
            f"FFS Static Confidence PCD | {round_label} | frame {int(frame_idx)}",
            (
                f"metric={metric_name} | scale={float(model_config['scale']):.2f} | "
                f"iters={int(model_config['valid_iters'])} | disp={int(model_config['max_disp'])}"
            ),
        ],
        row_headers=["RGB", "Masked FFS PCD", "Confidence"],
        column_headers=column_headers,
        image_rows=image_rows,
    )


def run_ffs_static_confidence_pcd_panels_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    ffs_repo: str | Path,
    model_path: str | Path,
    scale: float = 1.0,
    valid_iters: int = 8,
    max_disp: int = 192,
    frame_idx: int = 0,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    depth_min_m: float = 0.0,
    depth_max_m: float = 1.5,
    metrics: str = "both",
    point_size: float = 2.0,
    look_distance: float = 1.0,
    round_specs: list[dict[str, Any]] | None = None,
    runner_factory: Callable[..., Any] | None = None,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    round_specs = (
        build_static_confidence_pcd_round_specs(aligned_root=aligned_root)
        if round_specs is None
        else list(round_specs)
    )
    metric_names = _resolve_metric_names(metrics)
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
    }
    runner_factory = FastFoundationStereoRunner if runner_factory is None else runner_factory
    runner = runner_factory(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=float(scale),
        valid_iters=int(valid_iters),
        max_disp=int(max_disp),
    )
    render_frame_fn = render_frame_fn or _render_open3d_offscreen_pinhole

    rounds_summary: list[dict[str, Any]] = []
    for round_spec in round_specs:
        case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(round_spec["case_ref"]))
        metadata = load_case_metadata(case_dir)
        selected_frame_idx = int(frame_idx)
        if selected_frame_idx < 0 or selected_frame_idx >= int(metadata["frame_num"]):
            raise ValueError(
                f"frame_idx={selected_frame_idx} is out of range for case {case_dir.name} "
                f"with frame_num={int(metadata['frame_num'])}."
            )

        camera_ids = list(range(len(metadata["serial_numbers"])))
        if len(camera_ids) != 3:
            raise ValueError(f"Static confidence PCD workflow expects exactly 3 cameras. Got {camera_ids}.")

        round_output_dir = output_root / str(round_spec["round_id"])
        round_output_dir.mkdir(parents=True, exist_ok=True)
        mask_root = Path(round_spec["mask_root"]).resolve()
        if not mask_root.is_dir():
            raise FileNotFoundError(f"Missing static mask root for {round_spec['round_id']}: {mask_root}")

        c2w_list = load_calibration_transforms(
            case_dir / "calibrate.pkl",
            serial_numbers=metadata["serial_numbers"],
            calibration_reference_serials=metadata.get("calibration_reference_serials", metadata["serial_numbers"]),
        )
        minimal_clouds = [
            {
                "camera_idx": int(camera_idx),
                "serial": str(metadata["serial_numbers"][camera_idx]),
                "color_path": str(case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"),
            }
            for camera_idx in camera_ids
        ]
        mask_by_camera, mask_debug = load_union_masks_for_camera_clouds(
            mask_root=mask_root,
            camera_clouds=minimal_clouds,
            frame_token=str(selected_frame_idx),
            text_prompt=text_prompt,
        )

        camera_clouds_masked: list[dict[str, Any]] = []
        camera_clouds_unmasked: list[dict[str, Any]] = []
        rgb_images: list[np.ndarray] = []
        confidence_images_by_metric: dict[str, list[np.ndarray]] = {metric_name: [] for metric_name in metric_names}
        per_camera_summary: list[dict[str, Any]] = []

        for camera_idx in camera_ids:
            serial = str(metadata["serial_numbers"][camera_idx])
            color_path = case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            ir_left_path = case_dir / "ir_left" / str(camera_idx) / f"{selected_frame_idx}.png"
            ir_right_path = case_dir / "ir_right" / str(camera_idx) / f"{selected_frame_idx}.png"
            color_image = _load_color_image(color_path)
            ir_left = _load_ir_image(ir_left_path)
            ir_right = _load_ir_image(ir_right_path)
            mask = np.asarray(mask_by_camera[int(camera_idx)], dtype=bool)

            run_output = runner.run_pair_with_confidence(
                ir_left,
                ir_right,
                K_ir_left=np.asarray(metadata["K_ir_left"][camera_idx], dtype=np.float32),
                baseline_m=float(metadata["ir_baseline_m"][camera_idx]),
                audit_mode=False,
            )
            depth_ir_left_m = np.asarray(run_output["depth_ir_left_m"], dtype=np.float32)
            k_ir_left_used = np.asarray(run_output["K_ir_left_used"], dtype=np.float32)
            t_ir_left_to_color = np.asarray(metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
            k_color = np.asarray(metadata["K_color"][camera_idx], dtype=np.float32)
            c2w = np.asarray(c2w_list[camera_idx], dtype=np.float32)
            output_shape = (int(color_image.shape[0]), int(color_image.shape[1]))
            depth_color_m = align_depth_to_color(
                depth_ir_left_m,
                k_ir_left_used,
                t_ir_left_to_color,
                k_color,
                output_shape=output_shape,
                invalid_value=0.0,
            )
            depth_valid_mask = np.isfinite(depth_color_m) & (depth_color_m > 0)
            camera_points, camera_colors, source_pixel_uv, stats = depth_to_camera_points(
                depth_color_m,
                k_color,
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                color_image=color_image,
                pixel_roi=None,
                max_points_per_camera=None,
            )
            world_points = transform_points(camera_points, c2w)
            point_keep_mask = mask[source_pixel_uv[:, 1], source_pixel_uv[:, 0]] if len(source_pixel_uv) > 0 else np.zeros((0,), dtype=bool)
            camera_clouds_unmasked.append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "color_path": str(color_path),
                    "K_color": k_color,
                    "c2w": c2w,
                    "points": world_points,
                    "colors": camera_colors,
                    "source_pixel_uv": source_pixel_uv,
                }
            )
            camera_clouds_masked.append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "color_path": str(color_path),
                    "K_color": k_color,
                    "c2w": c2w,
                    "points": world_points[point_keep_mask],
                    "colors": camera_colors[point_keep_mask],
                    "source_pixel_uv": source_pixel_uv[point_keep_mask],
                }
            )
            rgb_images.append(_mask_rgb_image(color_path, mask=mask))

            confidence_summary: dict[str, Any] = {}
            for metric_name in metric_names:
                confidence_ir = np.asarray(run_output[f"confidence_{metric_name}_ir_left"], dtype=np.float32)
                confidence_color = align_ir_scalar_to_color(
                    depth_ir_left_m,
                    confidence_ir,
                    k_ir_left_used,
                    t_ir_left_to_color,
                    k_color,
                    output_shape=output_shape,
                    invalid_value=0.0,
                )
                confidence_color_vis = _colorize_confidence_map(confidence_color, valid_mask=depth_valid_mask)
                confidence_images_by_metric[metric_name].append(_mask_image(confidence_color_vis, mask=mask))
                confidence_summary[metric_name] = {
                    "aligned_valid_pixel_count": int(np.count_nonzero(depth_valid_mask)),
                    "masked_valid_pixel_count": int(np.count_nonzero(mask & depth_valid_mask)),
                    "min": float(np.min(confidence_color[depth_valid_mask])) if np.any(depth_valid_mask) else 0.0,
                    "max": float(np.max(confidence_color[depth_valid_mask])) if np.any(depth_valid_mask) else 0.0,
                }

            per_camera_summary.append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "color_path": str(color_path.resolve()),
                    "ir_left_path": str(ir_left_path.resolve()),
                    "ir_right_path": str(ir_right_path.resolve()),
                    "mask_pixel_count": int(np.count_nonzero(mask)),
                    "depth_valid_pixel_count_aligned": int(np.count_nonzero(depth_valid_mask)),
                    "depth_valid_pixel_count_masked": int(np.count_nonzero(mask & depth_valid_mask)),
                    "unmasked_point_count": int(len(world_points)),
                    "masked_point_count": int(np.count_nonzero(point_keep_mask)),
                    "depth_stats": dict(stats),
                    "confidence": confidence_summary,
                }
            )

        fused_masked_points, fused_masked_colors = _fuse_camera_clouds(camera_clouds_masked)
        fused_unmasked_points, fused_unmasked_colors = _fuse_camera_clouds(camera_clouds_unmasked)
        focus_bounds_min, focus_bounds_max, focus_source, fallback_used = _compute_focus_bounds(
            masked_points=fused_masked_points,
            unmasked_points=fused_unmasked_points,
        )
        crop_bounds = _expand_bounds(focus_bounds_min, focus_bounds_max)
        source_cloud_map = {int(item["camera_idx"]): item for item in camera_clouds_unmasked}
        c2w_selected = [np.asarray(source_cloud_map[camera_idx]["c2w"], dtype=np.float32) for camera_idx in camera_ids]
        serial_numbers = [str(source_cloud_map[camera_idx]["serial"]) for camera_idx in camera_ids]
        view_configs = build_original_camera_view_configs(
            c2w_list=c2w_selected,
            serial_numbers=serial_numbers,
            look_distance=float(look_distance),
            camera_ids=list(range(len(camera_ids))),
        )
        for idx, view_config in enumerate(view_configs):
            actual_camera_idx = int(camera_ids[idx])
            camera_cloud = source_cloud_map[actual_camera_idx]
            source_image_size = _image_size_from_color_path(camera_cloud["color_path"])
            target_image_size = (int(source_image_size[0]), int(source_image_size[1]))
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

        pcd_images: list[np.ndarray] = []
        pcd_render_paths: list[str] = []
        debug_dir = round_output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for view_config in view_configs:
            target_w, target_h = [int(item) for item in view_config["image_size"]]
            pcd_image = render_frame_fn(
                *crop_points_to_bounds(fused_masked_points, fused_masked_colors, crop_bounds),
                width=int(target_w),
                height=int(target_h),
                center=np.asarray(view_config["center"], dtype=np.float32),
                eye=np.asarray(view_config["camera_position"], dtype=np.float32),
                up=np.asarray(view_config["up"], dtype=np.float32),
                zoom=0.55,
                point_size=float(point_size),
                intrinsic_matrix=np.asarray(view_config["intrinsic_matrix"], dtype=np.float32),
                extrinsic_matrix=np.asarray(view_config["extrinsic_matrix"], dtype=np.float32),
            )
            pcd_images.append(pcd_image)
            pcd_render_path = debug_dir / f"masked_ffs_pcd_cam{int(view_config['camera_idx'])}.png"
            write_image(pcd_render_path, pcd_image)
            pcd_render_paths.append(str(pcd_render_path.resolve()))

        column_headers = [str(view_config["label"]) for view_config in view_configs]
        board_paths: dict[str, str] = {}
        for metric_name in metric_names:
            board = build_confidence_pcd_board(
                round_label=str(round_spec["round_label"]),
                frame_idx=selected_frame_idx,
                metric_name=metric_name,
                model_config=model_config,
                column_headers=column_headers,
                rgb_images=rgb_images,
                pcd_images=pcd_images,
                confidence_images=confidence_images_by_metric[metric_name],
            )
            board_path = round_output_dir / f"{metric_name}_board.png"
            write_image(board_path, board)
            board_paths[metric_name] = str(board_path.resolve())

        round_summary = {
            "round_id": str(round_spec["round_id"]),
            "round_label": str(round_spec["round_label"]),
            "case_ref": str(round_spec["case_ref"]),
            "case_dir": str(case_dir.resolve()),
            "frame_idx": int(selected_frame_idx),
            "mask_root": str(mask_root.resolve()),
            "text_prompt": str(text_prompt),
            "metrics": list(metric_names),
            "model_config": dict(model_config),
            "row_headers": ["RGB", "Masked FFS PCD", "Confidence"],
            "column_headers": column_headers,
            "board_paths": board_paths,
            "pcd_render_paths": pcd_render_paths,
            "pcd_render_contract": {
                "renderer": "open3d_hidden_visualizer",
                "projection_mode": "original_camera_pinhole",
                "view_mode": "original_camera_extrinsics",
                "shared_crop_across_panels": True,
                "row2_source": "fused_masked_ffs",
            },
            "column_views": [
                {
                    "camera_idx": int(view_config["camera_idx"]),
                    "label": str(view_config["label"]),
                    "camera_position": [float(value) for value in np.asarray(view_config["camera_position"], dtype=np.float32)],
                    "center": [float(value) for value in np.asarray(view_config["center"], dtype=np.float32)],
                    "up": [float(value) for value in np.asarray(view_config["up"], dtype=np.float32)],
                    "image_size": [int(item) for item in view_config["image_size"]],
                    "intrinsic_matrix": [float(item) for item in np.asarray(view_config["intrinsic_matrix"], dtype=np.float32).reshape(-1)],
                    "extrinsic_matrix": [float(item) for item in np.asarray(view_config["extrinsic_matrix"], dtype=np.float32).reshape(-1)],
                }
                for view_config in view_configs
            ],
            "crop_bounds": {
                "mode": str(crop_bounds["mode"]),
                "min": [float(value) for value in crop_bounds["min"]],
                "max": [float(value) for value in crop_bounds["max"]],
            },
            "focus_source": str(focus_source),
            "empty_mask_fallback_used": bool(fallback_used),
            "fused_unmasked_point_count": int(len(fused_unmasked_points)),
            "fused_masked_point_count": int(len(fused_masked_points)),
            "per_camera": per_camera_summary,
            "mask_debug": {str(key): value for key, value in mask_debug.items()},
            "output_dir": str(round_output_dir.resolve()),
        }
        write_json(round_output_dir / "summary.json", round_summary)
        rounds_summary.append(round_summary)

    manifest = {
        "output_dir": str(output_root.resolve()),
        "frame_idx": int(frame_idx),
        "text_prompt": str(text_prompt),
        "metrics": list(metric_names),
        "model_config": dict(model_config),
        "rounds": rounds_summary,
    }
    write_json(output_root / "summary.json", manifest)
    return manifest
