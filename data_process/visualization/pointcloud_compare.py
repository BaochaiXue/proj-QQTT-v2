from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration_io import load_calibration_transforms
from .io_artifacts import write_gif, write_json, write_ply_ascii, write_video
from .io_case import (
    choose_depth_stream,
    decode_depth_to_meters,
    depth_to_camera_points,
    get_case_intrinsics,
    get_depth_scale_list,
    get_frame_count,
    load_case_frame_camera_clouds,
    load_case_frame_cloud,
    load_case_frame_cloud_with_sources,
    load_case_metadata,
    resolve_case_dirs,
    select_frame_indices,
    transform_points,
    voxel_downsample,
)
from .layouts import compose_grid_2x3, overlay_panel_label
from .renderers.fallback import (
    apply_image_flip,
    estimate_ortho_scale,
    look_at_view_matrix,
    project_world_points_to_image,
    rasterize_point_cloud_view,
    render_point_cloud,
    render_point_cloud_fallback,
)
from .roi import compute_scene_crop_bounds, crop_points_to_bounds, estimate_focus_point
from .views import build_camera_pose_view_configs, compute_view_config, normalize_vector as _normalize_vector


RENDER_MODES = (
    "color_by_rgb",
    "color_by_depth",
    "color_by_height",
    "color_by_normals",
    "neutral_gray_shaded",
)
VIEW_NAMES = ("oblique", "top", "side")
VIEW_MODES = ("fixed", "camera_poses_table_focus")
FOCUS_MODES = ("none", "table")
LAYOUT_MODES = ("pair", "grid_2x3")
SCENE_CROP_MODES = ("none", "auto_table_bbox", "auto_object_bbox", "manual_xyz_roi")
PROJECTION_MODES = ("perspective", "orthographic")
IMAGE_FLIP_MODES = ("none", "vertical", "horizontal", "both")



def compose_panel(native_image: np.ndarray, ffs_image: np.ndarray, *, layout: str) -> np.ndarray:
    if layout == "stacked":
        return np.vstack([native_image, ffs_image])
    return np.hstack([native_image, ffs_image])


def run_depth_comparison_workflow(
    *,
    aligned_root: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    output_dir: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = None,
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
    renderer: str = "auto",
    write_ply: bool = False,
    write_mp4: bool = False,
    fps: int = 30,
    panel_layout: str = "side_by_side",
    use_float_ffs_depth_when_available: bool = False,
    render_mode: str = "neutral_gray_shaded",
    views: list[str] | None = None,
    zoom_scale: float = 1.0,
    view_mode: str = "fixed",
    focus_mode: str = "none",
    layout_mode: str = "pair",
    scene_crop_mode: str = "none",
    crop_margin_xy: float = 0.12,
    crop_min_z: float = -0.15,
    crop_max_z: float = 0.35,
    roi_x_min: float | None = None,
    roi_x_max: float | None = None,
    roi_y_min: float | None = None,
    roi_y_max: float | None = None,
    roi_z_min: float | None = None,
    roi_z_max: float | None = None,
    view_distance_scale: float = 1.0,
    projection_mode: str = "perspective",
    ortho_scale: float | None = None,
    point_radius_px: int = 2,
    supersample_scale: int = 2,
    image_flip: str = "none",
) -> dict[str, Any]:
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported render_mode: {render_mode}")
    if view_mode not in VIEW_MODES:
        raise ValueError(f"Unsupported view_mode: {view_mode}")
    if focus_mode not in FOCUS_MODES:
        raise ValueError(f"Unsupported focus_mode: {focus_mode}")
    if layout_mode not in LAYOUT_MODES:
        raise ValueError(f"Unsupported layout_mode: {layout_mode}")
    if scene_crop_mode not in SCENE_CROP_MODES:
        raise ValueError(f"Unsupported scene_crop_mode: {scene_crop_mode}")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")
    if image_flip not in IMAGE_FLIP_MODES:
        raise ValueError(f"Unsupported image_flip: {image_flip}")
    aligned_root = Path(aligned_root).resolve()
    output_dir = Path(output_dir).resolve()
    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=aligned_root,
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)

    if same_case_mode and not (native_case_dir / "depth_ffs").exists():
        raise ValueError("Same-case comparison requires an aligned case containing depth_ffs/.")

    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras for comparison.")

    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for comparison.")

    manual_xyz_roi = None
    if scene_crop_mode == "manual_xyz_roi":
        roi_values = [roi_x_min, roi_x_max, roi_y_min, roi_y_max, roi_z_min, roi_z_max]
        if any(value is None for value in roi_values):
            raise ValueError("manual_xyz_roi requires all roi_x/y/z min/max arguments.")
        manual_xyz_roi = {
            "x_min": float(roi_x_min),
            "x_max": float(roi_x_max),
            "y_min": float(roi_y_min),
            "y_max": float(roi_y_max),
            "z_min": float(roi_z_min),
            "z_max": float(roi_z_max),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    raw_bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    raw_bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    all_point_sets: list[np.ndarray] = []
    for native_frame_idx, ffs_frame_idx in frame_pairs:
        for source, case_dir, metadata, frame_idx in (
            ("native", native_case_dir, native_metadata, native_frame_idx),
            ("ffs", ffs_case_dir, ffs_metadata, ffs_frame_idx),
        ):
            points, colors, stats = load_case_frame_cloud(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=frame_idx,
                depth_source="realsense" if source == "native" else "ffs",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            cache[(source, frame_idx)] = (points, colors, stats)
            if len(points) > 0:
                all_point_sets.append(points)
                raw_bounds_min = np.minimum(raw_bounds_min, points.min(axis=0))
                raw_bounds_max = np.maximum(raw_bounds_max, points.max(axis=0))

    if not np.isfinite(raw_bounds_min).all():
        raw_bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        raw_bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    metrics = []
    renderer_used_by_view: dict[str, str] = {}
    focus_point = estimate_focus_point(
        all_point_sets,
        bounds_min=raw_bounds_min,
        bounds_max=raw_bounds_max,
        focus_mode=focus_mode,
    )
    crop_bounds = compute_scene_crop_bounds(
        all_point_sets,
        focus_point=focus_point,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=float(crop_margin_xy),
        crop_min_z=float(crop_min_z),
        crop_max_z=float(crop_max_z),
        manual_xyz_roi=manual_xyz_roi,
    )

    cropped_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    cropped_point_sets: list[np.ndarray] = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for key, (points, colors, stats) in cache.items():
        cropped_points, cropped_colors = crop_points_to_bounds(points, colors, crop_bounds)
        cropped_stats = dict(stats)
        cropped_stats["fused_point_count_before_crop"] = int(len(points))
        cropped_stats["fused_point_count_after_crop"] = int(len(cropped_points))
        cropped_cache[key] = (cropped_points, cropped_colors, cropped_stats)
        if len(cropped_points) > 0:
            cropped_point_sets.append(cropped_points)
            bounds_min = np.minimum(bounds_min, cropped_points.min(axis=0))
            bounds_max = np.maximum(bounds_max, cropped_points.max(axis=0))

    if not np.isfinite(bounds_min).all():
        bounds_min = raw_bounds_min.copy()
        bounds_max = raw_bounds_max.copy()

    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(float(np.linalg.norm(bounds_max - bounds_min)) * 2.0, 1.0)),
    }
    focus_point = estimate_focus_point(
        cropped_point_sets if cropped_point_sets else all_point_sets,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        focus_mode=focus_mode,
    )

    if view_mode == "fixed":
        selected_view_names = views or ["oblique"]
        for view_name in selected_view_names:
            if view_name not in VIEW_NAMES:
                raise ValueError(f"Unsupported view: {view_name}")
        view_configs = []
        for view_name in selected_view_names:
            config = compute_view_config(bounds_min, bounds_max, view_name=view_name)
            direction = _normalize_vector(config["camera_position"] - focus_point, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            distance = max(1e-3, float(np.linalg.norm(config["camera_position"] - focus_point)) * float(view_distance_scale))
            config["camera_position"] = np.asarray(focus_point, dtype=np.float32) + direction * distance
            config["center"] = np.asarray(focus_point, dtype=np.float32)
            config["radius"] = distance
            view_configs.append(config)
    else:
        if len(native_metadata["serial_numbers"]) != 3:
            raise ValueError("camera_poses_table_focus currently requires exactly 3 cameras.")
        calibration_reference_serials = native_metadata.get("calibration_reference_serials", native_metadata["serial_numbers"])
        native_c2w = load_calibration_transforms(
            native_case_dir / "calibrate.pkl",
            serial_numbers=native_metadata["serial_numbers"],
            calibration_reference_serials=calibration_reference_serials,
        )
        target_distance = max(
            0.6,
            float(np.linalg.norm(bounds_max - bounds_min)),
        )
        view_configs = build_camera_pose_view_configs(
            c2w_list=native_c2w,
            serial_numbers=native_metadata["serial_numbers"],
            focus_point=focus_point,
            view_distance_scale=float(view_distance_scale),
            target_distance=target_distance,
        )

    if layout_mode == "grid_2x3" and len(view_configs) != 3:
        raise ValueError("grid_2x3 layout requires exactly 3 view configs.")

    per_view_outputs: dict[str, dict[str, Any]] = {}
    grid_frame_paths: list[Path] = []
    if layout_mode == "grid_2x3":
        grid_frames_dir = output_dir / "grid_2x3_frames"
        grid_frames_dir.mkdir(parents=True, exist_ok=True)

    for view_config in view_configs:
        view_name = str(view_config["view_name"])
        view_output_dir = output_dir if len(view_configs) == 1 and layout_mode == "pair" else output_dir / f"view_{view_name}"
        view_output_dir.mkdir(parents=True, exist_ok=True)
        native_frames_dir = view_output_dir / "native_frames"
        ffs_frames_dir = view_output_dir / "ffs_frames"
        side_frames_dir = view_output_dir / "side_by_side_frames"
        for directory in (native_frames_dir, ffs_frames_dir, side_frames_dir):
            directory.mkdir(parents=True, exist_ok=True)
        native_clouds_dir = view_output_dir / "native_clouds"
        ffs_clouds_dir = view_output_dir / "ffs_clouds"
        if write_ply:
            native_clouds_dir.mkdir(parents=True, exist_ok=True)
            ffs_clouds_dir.mkdir(parents=True, exist_ok=True)
        per_view_outputs[view_name] = {
            "view_output_dir": view_output_dir,
            "native_frames_dir": native_frames_dir,
            "ffs_frames_dir": ffs_frames_dir,
            "side_frames_dir": side_frames_dir,
            "native_clouds_dir": native_clouds_dir,
            "ffs_clouds_dir": ffs_clouds_dir,
            "native_frame_paths": [],
            "ffs_frame_paths": [],
            "side_frame_paths": [],
            "renderer_used": None,
            "view_config": view_config,
            "ortho_scale": None,
        }

        if projection_mode == "orthographic":
            per_view_outputs[view_name]["ortho_scale"] = float(
                ortho_scale
                if ortho_scale is not None
                else estimate_ortho_scale(
                    cropped_point_sets if cropped_point_sets else all_point_sets,
                    view_config=view_config,
                )
            )

    for panel_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
        native_points, native_colors, native_stats = cropped_cache[("native", native_frame_idx)]
        ffs_points, ffs_colors, ffs_stats = cropped_cache[("ffs", ffs_frame_idx)]
        grid_native_images: list[np.ndarray] = []
        grid_ffs_images: list[np.ndarray] = []

        for view_config in view_configs:
            view_name = str(view_config["view_name"])
            view_state = per_view_outputs[view_name]
            renderer_used = view_state["renderer_used"]
            native_render, renderer_used = render_point_cloud(
                native_points,
                native_colors,
                renderer=renderer,
                view_config=view_config,
                render_mode=render_mode,
                scalar_bounds=scalar_bounds,
                zoom_scale=zoom_scale,
                point_radius_px=point_radius_px,
                supersample_scale=supersample_scale,
                projection_mode=projection_mode,
                ortho_scale=view_state["ortho_scale"],
            )
            ffs_render, renderer_used = render_point_cloud(
                ffs_points,
                ffs_colors,
                renderer=renderer if renderer != "auto" else renderer_used or "auto",
                view_config=view_config,
                render_mode=render_mode,
                scalar_bounds=scalar_bounds,
                zoom_scale=zoom_scale,
                point_radius_px=point_radius_px,
                supersample_scale=supersample_scale,
                projection_mode=projection_mode,
                ortho_scale=view_state["ortho_scale"],
            )
            view_state["renderer_used"] = renderer_used

            native_render = apply_image_flip(native_render, image_flip)
            ffs_render = apply_image_flip(ffs_render, image_flip)
            native_labeled = overlay_panel_label(native_render, label=f"Native | {view_config['label']}")
            ffs_labeled = overlay_panel_label(ffs_render, label=f"FFS | {view_config['label']}")
            side_render = compose_panel(native_labeled, ffs_labeled, layout=panel_layout)

            native_frame_path = view_state["native_frames_dir"] / f"{panel_idx:06d}.png"
            ffs_frame_path = view_state["ffs_frames_dir"] / f"{panel_idx:06d}.png"
            side_frame_path = view_state["side_frames_dir"] / f"{panel_idx:06d}.png"
            cv2.imwrite(str(native_frame_path), native_labeled)
            cv2.imwrite(str(ffs_frame_path), ffs_labeled)
            cv2.imwrite(str(side_frame_path), side_render)
            view_state["native_frame_paths"].append(native_frame_path)
            view_state["ffs_frame_paths"].append(ffs_frame_path)
            view_state["side_frame_paths"].append(side_frame_path)

            if write_ply:
                write_ply_ascii(view_state["native_clouds_dir"] / f"{panel_idx:06d}.ply", native_points, native_colors)
                write_ply_ascii(view_state["ffs_clouds_dir"] / f"{panel_idx:06d}.ply", ffs_points, ffs_colors)

            if layout_mode == "grid_2x3":
                grid_native_images.append(native_labeled)
                grid_ffs_images.append(ffs_labeled)

            metrics.append(
                {
                    "view_name": view_name,
                    "panel_frame_idx": panel_idx,
                    "native_frame_idx": native_frame_idx,
                    "ffs_frame_idx": ffs_frame_idx,
                    "native": native_stats,
                    "ffs": ffs_stats,
                }
            )

        if layout_mode == "grid_2x3":
            title = (
                f"{native_case_dir.name} vs {ffs_case_dir.name} | frame {panel_idx:06d} | "
                f"{render_mode} | crop={scene_crop_mode} | proj={projection_mode} | "
                f"dist={view_distance_scale:.2f} | depth=[{depth_min_m:.2f}, {depth_max_m:.2f}] m"
            )
            grid_image = compose_grid_2x3(
                title=title,
                column_headers=[f"View from Cam{i}" for i in range(3)],
                row_headers=["Native depth", "FFS depth"],
                native_images=grid_native_images,
                ffs_images=grid_ffs_images,
            )
            grid_frame_path = grid_frames_dir / f"{panel_idx:06d}.png"
            cv2.imwrite(str(grid_frame_path), grid_image)
            grid_frame_paths.append(grid_frame_path)

    for view_name, view_state in per_view_outputs.items():
        renderer_used_by_view[view_name] = view_state["renderer_used"] or "fallback"
        videos_dir = Path(view_state["view_output_dir"]) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if write_mp4:
            write_video(videos_dir / "native.mp4", view_state["native_frame_paths"], fps)
            write_video(videos_dir / "ffs.mp4", view_state["ffs_frame_paths"], fps)
            write_video(videos_dir / "side_by_side.mp4", view_state["side_frame_paths"], fps)

    if layout_mode == "grid_2x3":
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if write_mp4:
            write_video(videos_dir / "grid_2x3.mp4", grid_frame_paths, fps)

    comparison_metadata = {
        "same_case_mode": same_case_mode,
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": frame_pairs,
        "renderer_requested": renderer,
        "renderer_used": renderer_used_by_view,
        "views": [str(view_config["view_name"]) for view_config in view_configs],
        "view_labels": [str(view_config["label"]) for view_config in view_configs],
        "view_mode": view_mode,
        "focus_mode": focus_mode,
        "layout_mode": layout_mode,
        "render_mode": render_mode,
        "panel_layout": panel_layout,
        "scene_crop_mode": scene_crop_mode,
        "crop_bounds": {"min": crop_bounds["min"].tolist(), "max": crop_bounds["max"].tolist()},
        "crop_margin_xy": float(crop_margin_xy),
        "crop_min_z": float(crop_min_z),
        "crop_max_z": float(crop_max_z),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "voxel_size": voxel_size,
        "max_points_per_camera": max_points_per_camera,
        "use_float_ffs_depth_when_available": use_float_ffs_depth_when_available,
        "zoom_scale": float(zoom_scale),
        "view_distance_scale": float(view_distance_scale),
        "projection_mode": projection_mode,
        "ortho_scale": ortho_scale,
        "ortho_scale_by_view": {
            view_name: per_view_outputs[view_name]["ortho_scale"]
            for view_name in per_view_outputs
        },
        "point_radius_px": int(point_radius_px),
        "supersample_scale": int(supersample_scale),
        "image_flip": image_flip,
        "scalar_bounds": scalar_bounds,
        "focus_point": focus_point.tolist(),
    }
    write_json(output_dir / "comparison_metadata.json", comparison_metadata)
    write_json(output_dir / "metrics.json", metrics)
    return {
        "output_dir": str(output_dir),
        "comparison_metadata": comparison_metadata,
        "metrics": metrics,
    }
