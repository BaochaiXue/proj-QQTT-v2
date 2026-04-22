from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_process.aligned_case_metadata import load_aligned_metadata
from data_process.depth_backends import (
    FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND,
    FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND,
    FFS_FLOAT_ARCHIVE_DIR,
)
from .calibration_io import load_calibration_transforms
from .io_artifacts import write_ply_ascii


def load_case_metadata(case_dir: Path) -> dict[str, Any]:
    _, _, merged_metadata = load_aligned_metadata(case_dir)
    return merged_metadata


def decode_depth_to_meters(depth: np.ndarray, depth_scale_m_per_unit: float | None) -> np.ndarray:
    depth = np.asarray(depth)
    if np.issubdtype(depth.dtype, np.floating):
        depth_m = depth.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0] = 0.0
        return depth_m
    if depth_scale_m_per_unit is None:
        raise ValueError("depth_scale_m_per_unit is required to decode integer depth.")
    depth_m = depth.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_m[depth == 0] = 0.0
    return depth_m


def depth_to_camera_points(
    depth_m: np.ndarray,
    K_color: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
    color_image: np.ndarray,
    pixel_roi: tuple[int, int, int, int] | None = None,
    max_points_per_camera: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    color = np.asarray(color_image)
    if color.ndim == 2:
        color = np.tile(color[..., None], (1, 1, 3))

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    valid &= depth >= float(depth_min_m)
    valid &= depth <= float(depth_max_m)
    if pixel_roi is not None:
        x_min, y_min, x_max, y_max = [int(item) for item in pixel_roi]
        roi_mask = (
            (xx >= float(x_min))
            & (xx <= float(x_max))
            & (yy >= float(y_min))
            & (yy <= float(y_max))
        )
        valid &= roi_mask
    valid_count = int(valid.sum())
    if valid_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int32),
            {"valid_depth_pixels": 0, "points_after_sampling": 0},
        )

    z = depth[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = color[valid].astype(np.uint8)
    source_pixel_uv = np.stack([xx[valid], yy[valid]], axis=1).astype(np.int32)

    if max_points_per_camera is not None and len(points) > int(max_points_per_camera):
        idx = np.linspace(0, len(points) - 1, int(max_points_per_camera), dtype=np.int32)
        points = points[idx]
        colors = colors[idx]
        source_pixel_uv = source_pixel_uv[idx]

    return points, colors, source_pixel_uv, {
        "valid_depth_pixels": valid_count,
        "points_after_sampling": int(len(points)),
    }


def transform_points(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float | None) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices], colors[unique_indices]


def resolve_case_dirs(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
) -> tuple[Path, Path, bool]:
    aligned_root = Path(aligned_root).resolve()
    if case_name:
        case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=case_name)
        return case_dir, case_dir, True
    if realsense_case and ffs_case:
        return (
            resolve_case_dir(aligned_root=aligned_root, case_ref=realsense_case),
            resolve_case_dir(aligned_root=aligned_root, case_ref=ffs_case),
            False,
        )
    raise ValueError("Use either --case_name or both --realsense_case and --ffs_case.")


def resolve_case_dir(*, aligned_root: Path, case_ref: str) -> Path:
    root = Path(aligned_root).resolve()
    candidate = (root / case_ref).resolve()
    if candidate.is_dir():
        return candidate

    case_name = Path(str(case_ref)).name
    matches = sorted(
        {
            metadata_path.parent.resolve()
            for metadata_path in root.rglob("metadata.json")
            if metadata_path.parent.name == case_name
        },
        key=lambda path: str(path),
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        relative_matches = ", ".join(str(path.relative_to(root)) for path in matches)
        raise ValueError(
            f"Ambiguous case reference {case_ref!r} under {root}. "
            f"Use a relative subpath such as one of: {relative_matches}"
        )
    raise FileNotFoundError(f"Could not resolve case {case_ref!r} under {root}")


def get_frame_count(metadata: dict[str, Any]) -> int:
    return int(metadata["frame_num"])


def select_frame_indices(
    *,
    native_count: int,
    ffs_count: int,
    frame_start: int | None,
    frame_end: int | None,
    frame_stride: int,
) -> list[tuple[int, int]]:
    max_count = min(native_count, ffs_count)
    start = 0 if frame_start is None else max(0, int(frame_start))
    end = max_count - 1 if frame_end is None else min(int(frame_end), max_count - 1)
    if start > end:
        return []
    return [(idx, idx) for idx in range(start, end + 1, max(1, int(frame_stride)))]


def get_case_intrinsics(metadata: dict[str, Any]) -> list[np.ndarray]:
    matrices = metadata.get("K_color", metadata.get("intrinsics"))
    return [np.asarray(matrix, dtype=np.float32) for matrix in matrices]


def get_depth_scale_list(metadata: dict[str, Any], num_cameras: int) -> list[float | None]:
    scales = metadata.get("depth_scale_m_per_unit", [None] * num_cameras)
    if not isinstance(scales, list):
        scales = [scales for _ in range(num_cameras)]
    return scales


def choose_depth_stream(
    case_dir: Path,
    metadata: dict[str, Any],
    source: str,
    use_float_ffs_depth_when_available: bool,
    ffs_native_like_postprocess: bool = False,
) -> tuple[str, bool]:
    if source == "realsense":
        return "depth", False
    if source == "ffs_raw":
        if use_float_ffs_depth_when_available and (case_dir / FFS_FLOAT_ARCHIVE_DIR).is_dir():
            return FFS_FLOAT_ARCHIVE_DIR, True
        if use_float_ffs_depth_when_available and (case_dir / "depth_ffs_float_m").is_dir():
            return "depth_ffs_float_m", True

        primary_ffs_depth_dir = (
            "depth"
            if str(metadata.get("depth_source_for_depth_dir", "")) == "ffs" or str(metadata.get("depth_backend_used", "")) == "ffs"
            else "depth_ffs"
        )
        archive_ffs_depth_dir = (
            FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND
            if primary_ffs_depth_dir == "depth"
            else FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND
        )
        if (case_dir / archive_ffs_depth_dir).is_dir():
            return archive_ffs_depth_dir, False
        if (case_dir / primary_ffs_depth_dir).is_dir():
            return primary_ffs_depth_dir, False

    if ffs_native_like_postprocess:
        if use_float_ffs_depth_when_available and (case_dir / "depth_ffs_native_like_postprocess_float_m").is_dir():
            return "depth_ffs_native_like_postprocess_float_m", True
        if (case_dir / "depth_ffs_native_like_postprocess").is_dir():
            return "depth_ffs_native_like_postprocess", False

    if use_float_ffs_depth_when_available and (case_dir / "depth_ffs_float_m").is_dir():
        return "depth_ffs_float_m", True
    if (case_dir / "depth_ffs").is_dir():
        return "depth_ffs", False
    return "depth", False


def load_depth_frame(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    camera_idx: int,
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    native_depth_postprocess: bool = False,
    ffs_native_like_postprocess: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth_dir_name, use_float = choose_depth_stream(
        case_dir,
        metadata,
        depth_source,
        use_float_ffs_depth_when_available,
        ffs_native_like_postprocess=ffs_native_like_postprocess,
    )
    depth_path = case_dir / depth_dir_name / str(camera_idx) / f"{frame_idx}.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth frame: {depth_path}")

    depth_raw = np.load(depth_path)
    depth_scales = get_depth_scale_list(metadata, len(metadata["serial_numbers"]))
    depth_m = decode_depth_to_meters(depth_raw, None if use_float else depth_scales[camera_idx])

    native_postprocess_applied = False
    native_postprocess_origin = "none"
    postprocess_applied = False
    postprocess_origin = "none"
    raw_depth_dir_name = depth_dir_name
    if depth_source == "realsense" and native_depth_postprocess:
        from qqtt.env.camera.realsense.depth_postprocess import (
            NATIVE_DEPTH_POSTPROCESS_CONTRACT,
            apply_ffs_native_like_depth_postprocess_float_m,
        )

        filtered_u16, filtered_m = apply_ffs_native_like_depth_postprocess_float_m(
            depth_m,
            depth_scale_m_per_unit=float(depth_scales[camera_idx]),
            fps=int(metadata.get("fps", 30)),
            frame_number=int(frame_idx) + 1,
        )
        depth_raw = filtered_m if use_float else filtered_u16
        depth_m = filtered_m
        depth_dir_name = f"{raw_depth_dir_name}+{str(NATIVE_DEPTH_POSTPROCESS_CONTRACT['mode'])}"
        native_postprocess_applied = True
        native_postprocess_origin = "on_the_fly"
    elif depth_source == "ffs" and ffs_native_like_postprocess and not depth_dir_name.startswith("depth_ffs_native_like_postprocess"):
        from qqtt.env.camera.realsense.depth_postprocess import (
            FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_ON_THE_FLY_SUFFIX,
            apply_ffs_native_like_depth_postprocess_float_m,
        )

        filtered_u16, filtered_m = apply_ffs_native_like_depth_postprocess_float_m(
            depth_m,
            depth_scale_m_per_unit=float(depth_scales[camera_idx]),
            fps=int(metadata.get("fps", 30)),
            frame_number=int(frame_idx) + 1,
        )
        depth_raw = filtered_m if use_float else filtered_u16
        depth_m = filtered_m
        depth_dir_name = f"{raw_depth_dir_name}+{FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_ON_THE_FLY_SUFFIX}"
        postprocess_applied = True
        postprocess_origin = "on_the_fly"
    elif depth_source == "ffs" and ffs_native_like_postprocess:
        postprocess_applied = True
        postprocess_origin = "aligned_auxiliary"

    return depth_raw, depth_m, {
        "depth_dir_used": depth_dir_name,
        "source_depth_dir_used": raw_depth_dir_name,
        "used_float_depth": bool(use_float),
        "depth_path": str(depth_path),
        "native_depth_postprocess_enabled": bool(native_depth_postprocess if depth_source == "realsense" else False),
        "native_depth_postprocess_applied": bool(native_postprocess_applied),
        "native_depth_postprocess_origin": native_postprocess_origin,
        "ffs_native_like_postprocess_enabled": bool(ffs_native_like_postprocess if depth_source == "ffs" else False),
        "ffs_native_like_postprocess_applied": bool(postprocess_applied),
        "ffs_native_like_postprocess_origin": postprocess_origin,
    }


def load_case_frame_camera_clouds(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    pixel_roi_by_camera: dict[int, tuple[int, int, int, int]] | None = None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
    native_depth_postprocess: bool = False,
    ffs_native_like_postprocess: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    serials = metadata["serial_numbers"]
    intrinsics = get_case_intrinsics(metadata)
    calibration_reference_serials = metadata.get("calibration_reference_serials", metadata["serial_numbers"])
    c2w_list = load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=serials,
        calibration_reference_serials=calibration_reference_serials,
    )
    per_camera_clouds: list[dict[str, Any]] = []
    per_camera_stats = []
    for camera_idx, serial in enumerate(serials):
        color_path = case_dir / "color" / str(camera_idx) / f"{frame_idx}.png"
        color_image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_image is None:
            raise FileNotFoundError(f"Missing color frame: {color_path}")
        _, depth_m, depth_info = load_depth_frame(
            case_dir=case_dir,
            metadata=metadata,
            camera_idx=camera_idx,
            frame_idx=frame_idx,
            depth_source=depth_source,
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            native_depth_postprocess=native_depth_postprocess,
            ffs_native_like_postprocess=ffs_native_like_postprocess,
        )
        camera_points, camera_colors, source_pixel_uv, stats = depth_to_camera_points(
            depth_m,
            intrinsics[camera_idx],
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            color_image=color_image,
            pixel_roi=None if pixel_roi_by_camera is None else pixel_roi_by_camera.get(int(camera_idx)),
            max_points_per_camera=max_points_per_camera,
        )
        world_points = transform_points(camera_points, c2w_list[camera_idx])
        per_camera_clouds.append(
            {
                "camera_idx": int(camera_idx),
                "serial": serial,
                "K_color": intrinsics[camera_idx],
                "c2w": c2w_list[camera_idx],
                "color_path": str(color_path),
                "points": world_points,
                "colors": camera_colors,
                "source_pixel_uv": source_pixel_uv,
                "source_camera_idx": np.full((len(world_points),), int(camera_idx), dtype=np.int16),
                "source_serial": np.full((len(world_points),), serial, dtype=object),
                **depth_info,
            }
        )
        per_camera_stats.append(
            {
                "camera_idx": camera_idx,
                "serial": serial,
                **depth_info,
                **stats,
            }
        )
    return per_camera_clouds, {"per_camera": per_camera_stats}


def load_case_frame_cloud(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    voxel_size: float | None,
    pixel_roi_by_camera: dict[int, tuple[int, int, int, int]] | None = None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
    native_depth_postprocess: bool = False,
    ffs_native_like_postprocess: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    per_camera_clouds, camera_stats = load_case_frame_camera_clouds(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=frame_idx,
        depth_source=depth_source,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        pixel_roi_by_camera=pixel_roi_by_camera,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        native_depth_postprocess=native_depth_postprocess,
        ffs_native_like_postprocess=ffs_native_like_postprocess,
    )
    fused_points = [item["points"] for item in per_camera_clouds]
    fused_colors = [item["colors"] for item in per_camera_clouds]
    if fused_points:
        points = np.concatenate(fused_points, axis=0) if len(fused_points) > 1 else fused_points[0]
        colors = np.concatenate(fused_colors, axis=0) if len(fused_colors) > 1 else fused_colors[0]
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    points, colors = voxel_downsample(points, colors, voxel_size)
    stats = {
        "per_camera": camera_stats["per_camera"],
        "fused_point_count": int(len(points)),
    }
    return points, colors, stats


def load_case_frame_cloud_with_sources(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    voxel_size: float | None,
    pixel_roi_by_camera: dict[int, tuple[int, int, int, int]] | None = None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
    native_depth_postprocess: bool = False,
    ffs_native_like_postprocess: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    per_camera_clouds, camera_stats = load_case_frame_camera_clouds(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=frame_idx,
        depth_source=depth_source,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        pixel_roi_by_camera=pixel_roi_by_camera,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        native_depth_postprocess=native_depth_postprocess,
        ffs_native_like_postprocess=ffs_native_like_postprocess,
    )
    fused_points = [item["points"] for item in per_camera_clouds]
    fused_colors = [item["colors"] for item in per_camera_clouds]
    if fused_points:
        points = np.concatenate(fused_points, axis=0) if len(fused_points) > 1 else fused_points[0]
        colors = np.concatenate(fused_colors, axis=0) if len(fused_colors) > 1 else fused_colors[0]
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    points, colors = voxel_downsample(points, colors, voxel_size)
    stats = {
        "per_camera": camera_stats["per_camera"],
        "fused_point_count": int(len(points)),
    }
    return points, colors, stats, per_camera_clouds
