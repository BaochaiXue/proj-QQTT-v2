from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from ..calibration_io import load_calibration_transforms
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_frame_camera_clouds, load_case_metadata
from ..layouts import compose_single_row_board
from ..renderers import estimate_ortho_scale, render_point_cloud
from ..views import normalize_vector, project_vector_to_plane, rotate_vector_around_axis
from ..workflows.masked_pointcloud_compare import (
    filter_camera_clouds_with_pixel_masks,
    load_union_masks_for_camera_clouds,
)


DEFAULT_STILL_OBJECT_CASE = (
    "data/still_object/ffs203048_iter4_trt_level5/"
    "both_30_still_object_round1_20260428"
)
DEFAULT_OUTPUT_DIR = (
    "data/experiments/"
    "still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5"
)
DEFAULT_TEXT_PROMPT = "stuffed animal"


def _concat_clouds(camera_clouds: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [np.asarray(item["points"], dtype=np.float32) for item in camera_clouds if len(item["points"]) > 0]
    color_sets = [np.asarray(item["colors"], dtype=np.uint8) for item in camera_clouds if len(item["points"]) > 0]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def _deterministic_point_cap(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points is None or int(max_points) <= 0 or len(point_array) <= int(max_points):
        return point_array, color_array
    indices = np.linspace(0, len(point_array) - 1, int(max_points), dtype=np.int32)
    return point_array[indices], color_array[indices]


def _robust_bounds(point_sets: list[np.ndarray], *, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    pct = float(percentile)
    for points in point_sets:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(point_array) == 0:
            continue
        if pct > 0.0 and pct < 50.0 and len(point_array) >= 16:
            lows.append(np.percentile(point_array, pct, axis=0).astype(np.float32))
            highs.append(np.percentile(point_array, 100.0 - pct, axis=0).astype(np.float32))
        else:
            lows.append(point_array.min(axis=0).astype(np.float32))
            highs.append(point_array.max(axis=0).astype(np.float32))
    if not lows:
        center = np.zeros((3,), dtype=np.float32)
        return center - 0.5, center + 0.5
    return np.min(np.stack(lows, axis=0), axis=0), np.max(np.stack(highs, axis=0), axis=0)


def _crop_to_expanded_bounds(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(point_array) == 0:
        return point_array, color_array, 0
    lo = np.asarray(bounds_min, dtype=np.float32).reshape(3)
    hi = np.asarray(bounds_max, dtype=np.float32).reshape(3)
    extent = np.maximum(hi - lo, 1e-4)
    margin = extent * max(0.0, float(margin_ratio))
    keep = np.all((point_array >= (lo - margin)) & (point_array <= (hi + margin)), axis=1)
    if int(np.count_nonzero(keep)) < 64:
        return point_array, color_array, 0
    return point_array[keep], color_array[keep], int(len(point_array) - np.count_nonzero(keep))


def _serialize_array(values: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(values, dtype=np.float32).reshape(-1)]


def _load_masked_variant_cloud(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    masks_by_camera: dict[int, np.ndarray],
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_camera: int | None,
    use_float_ffs_depth_when_available: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    camera_clouds, cloud_stats = load_case_frame_camera_clouds(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=int(frame_idx),
        depth_source=str(depth_source),
        use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
        pixel_roi_by_camera=None,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
    )
    masked_clouds, mask_metrics = filter_camera_clouds_with_pixel_masks(
        camera_clouds,
        pixel_mask_by_camera=masks_by_camera,
    )
    points, colors = _concat_clouds(masked_clouds)
    stats = {
        "depth_source": str(depth_source),
        "point_count": int(len(points)),
        "cloud_stats": cloud_stats,
        "mask_metrics": mask_metrics,
    }
    return points, colors, masked_clouds, stats


def _build_keyed_orbit_views(
    *,
    c2w_list: list[np.ndarray],
    serial_numbers: list[str],
    focus_point: np.ndarray,
    start_camera_idx: int,
    num_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    focus = np.asarray(focus_point, dtype=np.float32).reshape(3)
    camera_positions = [np.asarray(c2w, dtype=np.float32).reshape(4, 4)[:3, 3] for c2w in c2w_list]
    camera_ups = [
        normalize_vector(-np.asarray(c2w, dtype=np.float32).reshape(4, 4)[:3, 1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        for c2w in c2w_list
    ]
    orbit_axis = normalize_vector(np.mean(np.stack(camera_ups, axis=0), axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    start_idx = int(start_camera_idx)
    start_offset = camera_positions[start_idx] - focus
    basis_x = project_vector_to_plane(start_offset, orbit_axis)
    basis_x = normalize_vector(basis_x, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis_y = normalize_vector(np.cross(orbit_axis, basis_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    key_nodes: list[dict[str, Any]] = []
    for camera_idx, (position, up, serial) in enumerate(zip(camera_positions, camera_ups, serial_numbers, strict=False)):
        offset = position - focus
        planar_x = float(offset @ basis_x)
        planar_y = float(offset @ basis_y)
        rel_angle = float(np.rad2deg(np.arctan2(planar_y, planar_x)) % 360.0)
        if camera_idx == start_idx:
            rel_angle = 0.0
        key_nodes.append(
            {
                "camera_idx": int(camera_idx),
                "serial": str(serial),
                "relative_angle_deg": rel_angle,
                "radius_m": float(np.hypot(planar_x, planar_y)),
                "height_m": float(offset @ orbit_axis),
                "position": _serialize_array(position),
                "up": _serialize_array(up),
            }
        )

    key_nodes_sorted = sorted(key_nodes, key=lambda item: float(item["relative_angle_deg"]))
    if key_nodes_sorted[0]["camera_idx"] != start_idx:
        key_nodes_sorted = [item for item in key_nodes_sorted if item["camera_idx"] == start_idx] + [
            item for item in key_nodes_sorted if item["camera_idx"] != start_idx
        ]
    interp_nodes = key_nodes_sorted + [dict(key_nodes_sorted[0], relative_angle_deg=360.0)]
    node_angles = np.asarray([float(item["relative_angle_deg"]) for item in interp_nodes], dtype=np.float32)
    node_radii = np.asarray([float(item["radius_m"]) for item in interp_nodes], dtype=np.float32)
    node_heights = np.asarray([float(item["height_m"]) for item in interp_nodes], dtype=np.float32)

    views: list[dict[str, Any]] = []
    total = max(1, int(num_frames))
    for frame_idx, angle_deg in enumerate(np.linspace(0.0, 360.0, total, endpoint=False)):
        radius = float(np.interp(angle_deg, node_angles, node_radii))
        height = float(np.interp(angle_deg, node_angles, node_heights))
        theta = np.deg2rad(float(angle_deg))
        planar = basis_x * (np.cos(theta) * radius) + basis_y * (np.sin(theta) * radius)
        camera_position = (focus + planar + orbit_axis * height).astype(np.float32)
        up = rotate_vector_around_axis(camera_ups[start_idx], orbit_axis, float(angle_deg))
        views.append(
            {
                "view_name": f"cam{start_idx}_orbit_{frame_idx:03d}",
                "label": f"Cam{start_idx} orbit {angle_deg:06.2f} deg",
                "camera_idx": int(start_idx),
                "serial": str(serial_numbers[start_idx]),
                "center": focus.copy(),
                "camera_position": camera_position,
                "up": normalize_vector(up, orbit_axis),
                "orbit_angle_deg": float(angle_deg),
                "radius": float(np.linalg.norm(camera_position - focus)),
            }
        )

    summary = {
        "focus_point": _serialize_array(focus),
        "orbit_axis": _serialize_array(orbit_axis),
        "orbit_basis_x": _serialize_array(basis_x),
        "orbit_basis_y": _serialize_array(basis_y),
        "start_camera_idx": int(start_idx),
        "key_nodes": key_nodes_sorted,
    }
    return views, summary


def _estimate_global_ortho_scale(
    *,
    point_sets: list[np.ndarray],
    views: list[dict[str, Any]],
    margin: float,
) -> float:
    if not views:
        return 1.0
    sample_count = min(36, len(views))
    sample_indices = np.linspace(0, len(views) - 1, sample_count, dtype=np.int32)
    scales = [
        estimate_ortho_scale(point_sets, view_config=views[int(idx)], margin=float(margin))
        for idx in sample_indices
    ]
    return float(max(scales) if scales else 1.0)


def _render_variant_tile(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    view_config: dict[str, Any],
    render_mode: str,
    scalar_bounds: dict[str, tuple[float, float]],
    tile_width: int,
    tile_height: int,
    point_radius_px: int,
    supersample_scale: int,
    projection_mode: str,
    ortho_scale: float | None,
    label: str,
) -> np.ndarray:
    rendered, _renderer_used = render_point_cloud(
        points,
        colors,
        renderer="fallback",
        view_config=view_config,
        render_mode=str(render_mode),
        scalar_bounds=scalar_bounds,
        width=int(tile_width),
        height=int(tile_height),
        point_radius_px=int(point_radius_px),
        supersample_scale=int(supersample_scale),
        projection_mode=str(projection_mode),
        ortho_scale=ortho_scale,
    )
    return label_tile(rendered, label, (int(tile_width), int(tile_height)))


def run_still_object_orbit_gif_workflow(
    *,
    case_dir: Path,
    output_dir: Path,
    frame_idx: int = 0,
    mask_root: Path | None = None,
    text_prompt: str = DEFAULT_TEXT_PROMPT,
    start_camera_idx: int = 0,
    num_frames: int = 360,
    fps: int = 30,
    tile_width: int = 480,
    tile_height: int = 360,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_variant: int | None = 160_000,
    robust_bounds_percentile: float = 1.0,
    crop_to_robust_bounds: bool = True,
    crop_margin_ratio: float = 0.25,
    render_mode: str = "neutral_gray_shaded",
    projection_mode: str = "orthographic",
    ortho_margin: float = 1.28,
    point_radius_px: int = 2,
    supersample_scale: int = 1,
) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_mask_root = (case_dir / "sam31_masks") if mask_root is None else Path(mask_root)
    selected_mask_root = selected_mask_root.resolve()

    metadata = load_case_metadata(case_dir)
    frame_count = int(metadata["frame_num"])
    selected_frame = int(frame_idx)
    if selected_frame < 0 or selected_frame >= frame_count:
        raise ValueError(f"frame_idx={selected_frame} is outside frame_num={frame_count}.")
    serial_numbers = [str(item) for item in metadata["serial_numbers"]]
    if int(start_camera_idx) < 0 or int(start_camera_idx) >= len(serial_numbers):
        raise ValueError(f"start_camera_idx={start_camera_idx} is outside camera count {len(serial_numbers)}.")

    minimal_clouds = [
        {
            "camera_idx": int(camera_idx),
            "serial": serial,
            "color_path": str(case_dir / "color" / str(camera_idx) / f"{selected_frame}.png"),
        }
        for camera_idx, serial in enumerate(serial_numbers)
    ]
    masks_by_camera, mask_debug = load_union_masks_for_camera_clouds(
        mask_root=selected_mask_root,
        camera_clouds=minimal_clouds,
        frame_token=str(selected_frame),
        text_prompt=str(text_prompt),
    )

    native_points, native_colors, _native_clouds, native_stats = _load_masked_variant_cloud(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=selected_frame,
        depth_source="realsense",
        masks_by_camera=masks_by_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=False,
    )
    ffs_points, ffs_colors, _ffs_clouds, ffs_stats = _load_masked_variant_cloud(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=selected_frame,
        depth_source="ffs_raw",
        masks_by_camera=masks_by_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        use_float_ffs_depth_when_available=True,
    )
    initial_bounds_min, initial_bounds_max = _robust_bounds(
        [native_points, ffs_points],
        percentile=float(robust_bounds_percentile),
    )
    native_cropped_point_count = 0
    ffs_cropped_point_count = 0
    if bool(crop_to_robust_bounds):
        native_points, native_colors, native_cropped_point_count = _crop_to_expanded_bounds(
            native_points,
            native_colors,
            bounds_min=initial_bounds_min,
            bounds_max=initial_bounds_max,
            margin_ratio=float(crop_margin_ratio),
        )
        ffs_points, ffs_colors, ffs_cropped_point_count = _crop_to_expanded_bounds(
            ffs_points,
            ffs_colors,
            bounds_min=initial_bounds_min,
            bounds_max=initial_bounds_max,
            margin_ratio=float(crop_margin_ratio),
        )

    native_points, native_colors = _deterministic_point_cap(
        native_points,
        native_colors,
        max_points=max_points_per_variant,
    )
    ffs_points, ffs_colors = _deterministic_point_cap(
        ffs_points,
        ffs_colors,
        max_points=max_points_per_variant,
    )

    c2w_list = load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=serial_numbers,
        calibration_reference_serials=metadata.get("calibration_reference_serials", serial_numbers),
    )
    bounds_min, bounds_max = _robust_bounds(
        [native_points, ffs_points],
        percentile=float(robust_bounds_percentile),
    )
    focus_point = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    views, orbit_summary = _build_keyed_orbit_views(
        c2w_list=c2w_list,
        serial_numbers=serial_numbers,
        focus_point=focus_point,
        start_camera_idx=int(start_camera_idx),
        num_frames=int(num_frames),
    )

    selected_projection_mode = str(projection_mode)
    ortho_scale = None
    if selected_projection_mode == "orthographic":
        ortho_scale = _estimate_global_ortho_scale(
            point_sets=[native_points, ffs_points],
            views=views,
            margin=float(ortho_margin),
        )
    elif selected_projection_mode != "perspective":
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")

    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(0.25, float(np.linalg.norm(bounds_max - bounds_min) * 2.5))),
    }
    ffs_config = dict(metadata.get("ffs_config", {}))
    ffs_model_name = str(ffs_config.get("model_name", "20-30-48"))
    ffs_valid_iters = int(ffs_config.get("valid_iters", 4))
    ffs_builder_level = int(ffs_config.get("builder_optimization_level", 5))
    ffs_label = f"FFS {ffs_model_name} iter{ffs_valid_iters} L{ffs_builder_level}"
    title_lines = [
        f"Still object round 1 | frame {selected_frame} | Cam{int(start_camera_idx)} keyed orbit",
        (
            f"FFS={ffs_model_name} iter{ffs_valid_iters} pad864 TRT L{ffs_builder_level} | "
            "key poses Cam0/Cam1/Cam2 | 360deg look-at object"
        ),
    ]

    gif_path = output_dir / f"still_object_round1_frame{selected_frame:04d}_cam{int(start_camera_idx)}_orbit_1x2.gif"
    first_frame_path = output_dir / f"still_object_round1_frame{selected_frame:04d}_cam{int(start_camera_idx)}_orbit_first.png"
    summary_path = output_dir / "summary.json"

    with imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(fps)), loop=0) as writer:
        for idx, view_config in enumerate(views):
            angle = float(view_config["orbit_angle_deg"])
            native_tile = _render_variant_tile(
                points=native_points,
                colors=native_colors,
                view_config=view_config,
                render_mode=str(render_mode),
                scalar_bounds=scalar_bounds,
                tile_width=int(tile_width),
                tile_height=int(tile_height),
                point_radius_px=int(point_radius_px),
                supersample_scale=int(supersample_scale),
                projection_mode=selected_projection_mode,
                ortho_scale=ortho_scale,
                label=f"Native Depth | {len(native_points)} pts | {angle:05.1f} deg",
            )
            ffs_tile = _render_variant_tile(
                points=ffs_points,
                colors=ffs_colors,
                view_config=view_config,
                render_mode=str(render_mode),
                scalar_bounds=scalar_bounds,
                tile_width=int(tile_width),
                tile_height=int(tile_height),
                point_radius_px=int(point_radius_px),
                supersample_scale=int(supersample_scale),
                projection_mode=selected_projection_mode,
                ortho_scale=ortho_scale,
                label=f"FFS pad864 L{ffs_builder_level} | {len(ffs_points)} pts | {angle:05.1f} deg",
            )
            board = compose_single_row_board(
                title_lines=title_lines,
                column_headers=["Native Depth", ffs_label],
                images=[native_tile, ffs_tile],
            )
            if idx == 0:
                write_image(first_frame_path, board)
            writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            if idx == 0 or (idx + 1) % 30 == 0 or idx + 1 == len(views):
                print(f"rendered {idx + 1}/{len(views)} frames", flush=True)

    summary = {
        "case_dir": str(case_dir),
        "mask_root": str(selected_mask_root),
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "frame_idx": selected_frame,
        "text_prompt": str(text_prompt),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "crop_to_robust_bounds": bool(crop_to_robust_bounds),
        "crop_margin_ratio": float(crop_margin_ratio),
        "initial_bounds_min": _serialize_array(initial_bounds_min),
        "initial_bounds_max": _serialize_array(initial_bounds_max),
        "projection_mode": selected_projection_mode,
        "ortho_scale": None if ortho_scale is None else float(ortho_scale),
        "render_mode": str(render_mode),
        "point_radius_px": int(point_radius_px),
        "serial_numbers": serial_numbers,
        "ffs_config": ffs_config,
        "mask_debug": mask_debug,
        "native_stats": native_stats,
        "ffs_stats": ffs_stats,
        "native_cropped_point_count": int(native_cropped_point_count),
        "ffs_cropped_point_count": int(ffs_cropped_point_count),
        "native_render_point_count": int(len(native_points)),
        "ffs_render_point_count": int(len(ffs_points)),
        "bounds_min": _serialize_array(bounds_min),
        "bounds_max": _serialize_array(bounds_max),
        "orbit": orbit_summary,
    }
    write_json(summary_path, summary)
    return summary
