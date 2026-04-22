from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .io_artifacts import write_json, write_video
from .io_case import get_frame_count, load_case_frame_cloud_with_sources, load_case_metadata, resolve_case_dirs, select_frame_indices
from .pointcloud_defaults import DEFAULT_POINTCLOUD_DEPTH_MAX_M, DEFAULT_POINTCLOUD_DEPTH_MIN_M
from .renderers import apply_image_flip
from .roi import compute_scene_crop_bounds, crop_points_to_bounds, estimate_focus_point
from .triplet_ply_compare import _aggregate_postprocess_origin, _case_has_ffs_raw_depth
from .views import compute_view_config, normalize_vector


TRIPLET_VIDEO_RENDER_CONTRACT = {
    "renderer": "open3d_hidden_visualizer",
    "render_mode": "color_by_rgb",
    "image_flip": "vertical",
    "view_name": "oblique",
    "scene_crop_mode": "auto_table_bbox",
}
FOCUS_SAMPLE_TARGET = 12


def _resolve_triplet_frame_pairs(
    *,
    native_metadata: dict[str, Any],
    ffs_metadata: dict[str, Any],
    frame_start: int | None,
    frame_end: int | None,
    frame_stride: int,
) -> list[tuple[int, int]]:
    return select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )


def _render_open3d_hidden_window(
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
    intrinsic_matrix: np.ndarray | None = None,
    extrinsic_matrix: np.ndarray | None = None,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    # loader colors are BGR uint8; Open3D expects RGB float in [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors[:, ::-1], dtype=np.float64) / 255.0)

    vis = o3d.visualization.Visualizer()
    window_created = vis.create_window(window_name="triplet_video", width=int(width), height=int(height), visible=False)
    if not window_created:
        raise RuntimeError("Failed to create hidden Open3D window for triplet video rendering.")

    try:
        vis.add_geometry(pcd)
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        render_option.point_size = float(point_size)
        view_control = vis.get_view_control()
        if intrinsic_matrix is not None or extrinsic_matrix is not None:
            if intrinsic_matrix is None or extrinsic_matrix is None:
                raise ValueError("intrinsic_matrix and extrinsic_matrix must be passed together.")
            intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
            extrinsic = np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4)
            camera_parameters = o3d.camera.PinholeCameraParameters()
            camera_parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                int(width),
                int(height),
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
            )
            camera_parameters.extrinsic = extrinsic
            if not view_control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True):
                raise RuntimeError("Failed to apply Open3D pinhole camera parameters.")
        else:
            view_control.set_lookat(np.asarray(center, dtype=np.float64))
            view_control.set_front(normalize_vector(np.asarray(center, dtype=np.float32) - np.asarray(eye, dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float64))
            view_control.set_up(np.asarray(up, dtype=np.float64))
            view_control.set_zoom(float(zoom))
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    finally:
        vis.destroy_window()

    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)


def _variant_specs(
    *,
    native_case_dir: Path,
    native_metadata: dict[str, Any],
    ffs_case_dir: Path,
    ffs_metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "native": {
            "case_dir": native_case_dir,
            "metadata": native_metadata,
            "depth_source": "realsense",
            "ffs_native_like_postprocess": False,
            "frame_pair_index": 0,
        },
        "ffs_raw": {
            "case_dir": ffs_case_dir,
            "metadata": ffs_metadata,
            "depth_source": "ffs_raw",
            "ffs_native_like_postprocess": False,
            "frame_pair_index": 1,
        },
        "ffs_postprocess": {
            "case_dir": ffs_case_dir,
            "metadata": ffs_metadata,
            "depth_source": "ffs",
            "ffs_native_like_postprocess": True,
            "frame_pair_index": 1,
        },
    }


def _load_triplet_variant_frame(
    *,
    spec: dict[str, Any],
    frame_pair: tuple[int, int],
    use_float_ffs_depth_when_available: bool,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], int]:
    variant_frame_idx = int(frame_pair[int(spec["frame_pair_index"])])
    points, colors, stats, per_camera_clouds = load_case_frame_cloud_with_sources(
        case_dir=spec["case_dir"],
        metadata=spec["metadata"],
        frame_idx=variant_frame_idx,
        depth_source=str(spec["depth_source"]),
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        ffs_native_like_postprocess=bool(spec["ffs_native_like_postprocess"]),
    )
    return points, colors, stats, per_camera_clouds, variant_frame_idx


def _build_variant_video_summary(
    *,
    video_path: Path,
    frames_dir: Path,
    point_counts: list[int],
    stats_by_frame: list[dict[str, Any]],
    per_camera_clouds_by_frame: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    per_camera_entries: list[dict[str, Any]] = []
    for stats, clouds in zip(stats_by_frame, per_camera_clouds_by_frame, strict=False):
        for camera_stats, camera_cloud in zip(stats["per_camera"], clouds, strict=False):
            per_camera_entries.append(
                {
                    "camera_idx": int(camera_stats["camera_idx"]),
                    "serial": str(camera_stats["serial"]),
                    "point_count": int(len(camera_cloud["points"])),
                    "valid_depth_pixels": int(camera_stats["valid_depth_pixels"]),
                    "depth_dir_used": str(camera_stats["depth_dir_used"]),
                    "source_depth_dir_used": str(camera_stats["source_depth_dir_used"]),
                    "used_float_depth": bool(camera_stats["used_float_depth"]),
                    "depth_path": str(camera_stats["depth_path"]),
                    "ffs_native_like_postprocess_enabled": bool(camera_stats["ffs_native_like_postprocess_enabled"]),
                    "ffs_native_like_postprocess_applied": bool(camera_stats["ffs_native_like_postprocess_applied"]),
                    "ffs_native_like_postprocess_origin": str(camera_stats["ffs_native_like_postprocess_origin"]),
                }
            )

    return {
        "video_path": str(video_path.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "frame_count": int(len(point_counts)),
        "point_count_min": int(min(point_counts)) if point_counts else 0,
        "point_count_max": int(max(point_counts)) if point_counts else 0,
        "point_count_mean": float(np.mean(point_counts)) if point_counts else 0.0,
        "depth_dirs_used": sorted({str(item["depth_dir_used"]) for item in per_camera_entries}),
        "source_depth_dirs_used": sorted({str(item["source_depth_dir_used"]) for item in per_camera_entries}),
        "used_float_depth": bool(all(bool(item["used_float_depth"]) for item in per_camera_entries)) if per_camera_entries else False,
        "ffs_native_like_postprocess_enabled": bool(any(bool(item["ffs_native_like_postprocess_enabled"]) for item in per_camera_entries)),
        "ffs_native_like_postprocess_applied": bool(any(bool(item["ffs_native_like_postprocess_applied"]) for item in per_camera_entries)),
        "ffs_native_like_postprocess_origin": _aggregate_postprocess_origin(per_camera_entries),
    }


def run_triplet_video_compare_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = 50000,
    depth_min_m: float = DEFAULT_POINTCLOUD_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_POINTCLOUD_DEPTH_MAX_M,
    use_float_ffs_depth_when_available: bool = True,
    render_width: int = 960,
    render_height: int = 720,
    point_size: float = 2.0,
    zoom: float = 0.55,
    crop_margin_xy: float = 0.12,
    crop_min_z: float = -0.15,
    crop_max_z: float = 0.35,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
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

    if not _case_has_ffs_raw_depth(ffs_case_dir, ffs_metadata):
        raise ValueError("Triplet video compare requires an aligned FFS case containing raw FFS depth or a raw-depth archive.")
    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras for triplet video comparison.")

    frame_pairs = _resolve_triplet_frame_pairs(
        native_metadata=native_metadata,
        ffs_metadata=ffs_metadata,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for triplet video comparison.")

    render_frame_fn = render_frame_fn or _render_open3d_hidden_window
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_specs = _variant_specs(
        native_case_dir=native_case_dir,
        native_metadata=native_metadata,
        ffs_case_dir=ffs_case_dir,
        ffs_metadata=ffs_metadata,
    )
    focus_sample_stride = max(1, len(frame_pairs) // FOCUS_SAMPLE_TARGET)

    raw_bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    raw_bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    sample_point_sets: list[np.ndarray] = []
    for pair_idx, frame_pair in enumerate(frame_pairs):
        for spec in variant_specs.values():
            points, _, _, _, _ = _load_triplet_variant_frame(
                spec=spec,
                frame_pair=frame_pair,
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            if len(points) == 0:
                continue
            raw_bounds_min = np.minimum(raw_bounds_min, points.min(axis=0))
            raw_bounds_max = np.maximum(raw_bounds_max, points.max(axis=0))
            if pair_idx % focus_sample_stride == 0:
                sample_point_sets.append(points)

    if not np.isfinite(raw_bounds_min).all():
        raise RuntimeError("Triplet video compare could not load any valid fused points.")

    focus_point = estimate_focus_point(
        sample_point_sets,
        bounds_min=raw_bounds_min,
        bounds_max=raw_bounds_max,
        focus_mode="table",
    )
    crop_bounds = compute_scene_crop_bounds(
        sample_point_sets,
        focus_point=focus_point,
        scene_crop_mode="auto_table_bbox",
        crop_margin_xy=float(crop_margin_xy),
        crop_min_z=float(crop_min_z),
        crop_max_z=float(crop_max_z),
    )

    cropped_bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    cropped_bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for frame_pair in frame_pairs:
        for spec in variant_specs.values():
            points, colors, _, _, _ = _load_triplet_variant_frame(
                spec=spec,
                frame_pair=frame_pair,
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            points, colors = crop_points_to_bounds(points, colors, crop_bounds)
            if len(points) == 0:
                continue
            cropped_bounds_min = np.minimum(cropped_bounds_min, points.min(axis=0))
            cropped_bounds_max = np.maximum(cropped_bounds_max, points.max(axis=0))
    if not np.isfinite(cropped_bounds_min).all():
        cropped_bounds_min = raw_bounds_min.copy()
        cropped_bounds_max = raw_bounds_max.copy()

    view_config = compute_view_config(cropped_bounds_min, cropped_bounds_max, view_name="oblique")
    direction = normalize_vector(
        np.asarray(view_config["camera_position"], dtype=np.float32) - focus_point,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    distance = max(1e-3, float(np.linalg.norm(np.asarray(view_config["camera_position"], dtype=np.float32) - focus_point)))
    center = np.asarray(focus_point, dtype=np.float32)
    eye = center + direction * distance
    up = np.asarray(view_config["up"], dtype=np.float32)

    video_fps = min(int(native_metadata.get("fps", 30)), int(ffs_metadata.get("fps", 30)))
    variants_summary: dict[str, dict[str, Any]] = {}
    for variant_name, spec in variant_specs.items():
        frames_dir = output_dir / f"{variant_name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        point_counts: list[int] = []
        stats_by_frame: list[dict[str, Any]] = []
        per_camera_clouds_by_frame: list[list[dict[str, Any]]] = []
        frame_indices: list[int] = []

        for output_frame_idx, frame_pair in enumerate(frame_pairs):
            points, colors, stats, per_camera_clouds, variant_frame_idx = _load_triplet_variant_frame(
                spec=spec,
                frame_pair=frame_pair,
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            points, colors = crop_points_to_bounds(points, colors, crop_bounds)
            image = render_frame_fn(
                points,
                colors,
                width=int(render_width),
                height=int(render_height),
                center=center,
                eye=eye,
                up=up,
                zoom=float(zoom),
                point_size=float(point_size),
            )
            image = apply_image_flip(image, "vertical")
            frame_path = frames_dir / f"{output_frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), image)
            frame_paths.append(frame_path)
            point_counts.append(int(len(points)))
            stats_by_frame.append(stats)
            per_camera_clouds_by_frame.append(per_camera_clouds)
            frame_indices.append(int(variant_frame_idx))

        video_path = output_dir / f"{variant_name}_open3d.mp4"
        write_video(video_path, frame_paths, fps=video_fps)
        variant_summary = _build_variant_video_summary(
            video_path=video_path,
            frames_dir=frames_dir,
            point_counts=point_counts,
            stats_by_frame=stats_by_frame,
            per_camera_clouds_by_frame=per_camera_clouds_by_frame,
        )
        variant_summary["frame_indices"] = frame_indices
        variants_summary[variant_name] = variant_summary

    summary = {
        "aligned_root": str(aligned_root),
        "output_dir": str(output_dir),
        "same_case_mode": bool(same_case_mode),
        "case_name": case_name,
        "native_case_name": str(native_case_dir.name),
        "ffs_case_name": str(ffs_case_dir.name),
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": [[int(native_idx), int(ffs_idx)] for native_idx, ffs_idx in frame_pairs],
        "frame_count": int(len(frame_pairs)),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_stride": int(frame_stride),
        "fps": int(video_fps),
        "focus_sample_stride": int(focus_sample_stride),
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "crop_bounds": {
            "min": [float(value) for value in crop_bounds["min"]],
            "max": [float(value) for value in crop_bounds["max"]],
            "mode": str(crop_bounds["mode"]),
        },
        "view": {
            "view_name": str(view_config["view_name"]),
            "center": [float(value) for value in center],
            "eye": [float(value) for value in eye],
            "up": [float(value) for value in up],
            "zoom": float(zoom),
            "point_size": float(point_size),
            "image_size": [int(render_width), int(render_height)],
        },
        "render_contract": dict(TRIPLET_VIDEO_RENDER_CONTRACT),
        "variants": variants_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
