from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
import json
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

DEFAULT_FUTUREPHYSTWIN_ROOT = Path("/home/xinjie/FuturePhysTwin")
DEFAULT_FUTUREPHYSTWIN_PYTHON = sys.executable
DEFAULT_SAM3D_ROOT = Path("/home/xinjie/external/sam-3d-objects")
DEFAULT_SHAPE_PRIOR_CAMERA_IDX = 0
DEFAULT_SHAPE_PRIOR_RENDER_COLOR_RGB = (150, 150, 150)
DEFAULT_TRACK_PROCESS_CONTROLLER_POINTS = 30
DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME = "qqtt_world_c2w"
DEFAULT_SHAPE_PRIOR_UNITS = "meters"
DEFAULT_SHAPE_PRIOR_GROUND_POLICY = "preserve"
DEFAULT_SHAPE_PRIOR_GROUND_Z = 0.0
SHAPE_PRIOR_VALIDATION_GROUND_Z_EPS_M = 1e-4
SHAPE_PRIOR_VALIDATION_POSITIVE_Z_EPS_M = 0.01
SHAPE_PRIOR_VALIDATION_MAX_GROUND_Z_FRACTION = 0.25
SHAPE_PRIOR_VALIDATION_MAX_CENTROID_DRIFT_M = 0.15
SHAPE_PRIOR_VALIDATION_MIN_Z_EXTENT_RATIO = 0.25


@dataclass(frozen=True)
class ShapePriorWarmupConfig:
    enabled: bool
    output_root: Path
    run_id: str
    futurephystwin_root: Path = DEFAULT_FUTUREPHYSTWIN_ROOT
    futurephystwin_python: str = DEFAULT_FUTUREPHYSTWIN_PYTHON
    sam3d_root: Path = DEFAULT_SAM3D_ROOT
    camera_idx: int = DEFAULT_SHAPE_PRIOR_CAMERA_IDX
    force: bool = False
    object_label: str = "stuffed animal"
    controller_label: str = "towel"
    max_controller_points: int = DEFAULT_TRACK_PROCESS_CONTROLLER_POINTS
    ground_policy: str = DEFAULT_SHAPE_PRIOR_GROUND_POLICY
    ground_z: float = DEFAULT_SHAPE_PRIOR_GROUND_Z
    cuda_visible_devices: str | None = None
    cuda_allocator_config: str | None = None
    skip_route_visualizations: bool = False

    @property
    def run_root(self) -> Path:
        output_root = Path(self.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = output_root.resolve()
        return output_root / "demo33_shape_prior_warmup" / self.run_id

    @property
    def case_dir(self) -> Path:
        return self.run_root / "case"


@dataclass(frozen=True)
class ShapePriorWarmupResult:
    status: str
    case_dir: Path
    object_points0: np.ndarray
    surface_points: np.ndarray
    interior_points: np.ndarray
    structure_points: np.ndarray
    structure_colors_rgb: np.ndarray
    profile: dict[str, Any]


SubprocessRunner = Callable[..., Any]


def _as_bool_mask(mask: Any, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}.")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ValueError(f"Mask shape {arr.shape} does not match image/depth shape {shape}.")
    return np.ascontiguousarray(arr)


def _as_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {arr.shape}.")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_intrinsics(intrinsics: Any) -> np.ndarray:
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics, got shape {arr.shape}.")
    return np.ascontiguousarray(arr)


def _as_c2w(c2w: Any) -> np.ndarray:
    arr = np.asarray(c2w, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 c2w, got shape {arr.shape}.")
    return np.ascontiguousarray(arr)


def validate_original_sam3d_root(root: Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    if "MV-SAM3D" in resolved.parts:
        raise ValueError(
            f"Demo 3.3 shape-prior warmup must use original SAM 3D Objects, not MV-SAM3D: {resolved}"
        )
    required = [resolved / "notebook" / "inference.py", resolved / "sam3d_objects"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Original SAM 3D Objects checkout is missing or incomplete. "
            f"Expected notebook/inference.py and sam3d_objects/ under {resolved}. Missing: {missing}"
        )
    return resolved


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write RGB PNG: {path}")


def _write_mask_png(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), payload):
        raise RuntimeError(f"Failed to write mask PNG: {path}")


def _backproject_world_points(depth_m: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}.")
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth
    x = (xx - float(intrinsics[0, 2])) * z / float(intrinsics[0, 0])
    y = (yy - float(intrinsics[1, 2])) * z / float(intrinsics[1, 1])
    camera_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    return np.ascontiguousarray(world_points.reshape(height, width, 3), dtype=np.float32)


def _cap_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    sample_idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return np.ascontiguousarray(points[sample_idx]), np.ascontiguousarray(colors[sample_idx])


def _valid_masked_points(
    points_hw: np.ndarray,
    colors_hw: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_m) & (np.asarray(depth_m, dtype=np.float32) > 0.0)
    points = np.asarray(points_hw, dtype=np.float32)[valid]
    colors = np.asarray(colors_hw, dtype=np.float32)[valid] / 255.0
    finite = np.isfinite(points).all(axis=1)
    return np.ascontiguousarray(points[finite]), np.ascontiguousarray(colors[finite])


def _finite_xyz(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    finite = np.isfinite(arr).all(axis=1)
    return np.ascontiguousarray(arr[finite], dtype=np.float32)


def _point_stats(points: np.ndarray) -> dict[str, Any]:
    pts = _finite_xyz(points)
    if len(pts) == 0:
        return {
            "count": 0,
            "centroid": [0.0, 0.0, 0.0],
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
            "extent": [0.0, 0.0, 0.0],
        }
    min_xyz = np.min(pts, axis=0)
    max_xyz = np.max(pts, axis=0)
    return {
        "count": int(len(pts)),
        "centroid": np.mean(pts, axis=0).astype(float).tolist(),
        "min": min_xyz.astype(float).tolist(),
        "max": max_xyz.astype(float).tolist(),
        "extent": (max_xyz - min_xyz).astype(float).tolist(),
    }


def _load_source_object_points(case_dir: Path) -> np.ndarray | None:
    track_path = Path(case_dir) / "track_process_data.pkl"
    if not track_path.is_file():
        return None
    with track_path.open("rb") as handle:
        track_data = pickle.load(handle)
    source = np.asarray(track_data.get("object_points"), dtype=np.float32)
    if source.ndim != 3 or source.shape[0] < 1 or source.shape[2] != 3:
        return None
    return _finite_xyz(source[0])


def _load_case_metadata(case_dir: Path) -> dict[str, Any]:
    metadata_path = Path(case_dir) / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_shape_prior_coordinates(
    *,
    case_dir: Path,
    object_points0: np.ndarray,
    ground_z: float,
) -> dict[str, Any]:
    source_points = _load_source_object_points(case_dir)
    final_points = _finite_xyz(object_points0)
    profile: dict[str, Any] = {
        "shape_prior_coordinate_validation_status": "unavailable",
        "shape_prior_coordinate_validation_reason": "missing_source_object_points",
        "shape_prior_coordinate_validation_source_points": int(0 if source_points is None else len(source_points)),
        "shape_prior_coordinate_validation_final_points": int(len(final_points)),
        "shape_prior_coordinate_validation_centroid_drift_m": 0.0,
        "shape_prior_coordinate_validation_z_extent_ratio": 1.0,
        "shape_prior_coordinate_validation_ground_z_fraction": 0.0,
    }
    if source_points is None or len(source_points) == 0 or len(final_points) == 0:
        return profile

    source_stats = _point_stats(source_points)
    final_stats = _point_stats(final_points)
    source_min = np.asarray(source_stats["min"], dtype=np.float32)
    source_max = np.asarray(source_stats["max"], dtype=np.float32)
    source_centroid = np.asarray(source_stats["centroid"], dtype=np.float32)
    final_min = np.asarray(final_stats["min"], dtype=np.float32)
    final_max = np.asarray(final_stats["max"], dtype=np.float32)
    final_centroid = np.asarray(final_stats["centroid"], dtype=np.float32)
    source_z_extent = float(max(source_max[2] - source_min[2], 0.0))
    final_z_extent = float(max(final_max[2] - final_min[2], 0.0))
    z_extent_ratio = float(final_z_extent / source_z_extent) if source_z_extent > 1e-9 else 1.0
    centroid_drift = float(np.linalg.norm(final_centroid - source_centroid))
    ground_fraction = float(np.mean(np.abs(final_points[:, 2] - float(ground_z)) <= SHAPE_PRIOR_VALIDATION_GROUND_Z_EPS_M))
    source_has_positive_z = bool(source_max[2] > float(ground_z) + SHAPE_PRIOR_VALIDATION_POSITIVE_Z_EPS_M)
    clamp_suspected = bool(
        source_has_positive_z
        and final_max[2] <= float(ground_z) + SHAPE_PRIOR_VALIDATION_GROUND_Z_EPS_M
        and ground_fraction > SHAPE_PRIOR_VALIDATION_MAX_GROUND_Z_FRACTION
    )
    z_collapsed = bool(
        source_z_extent > SHAPE_PRIOR_VALIDATION_POSITIVE_Z_EPS_M
        and z_extent_ratio < SHAPE_PRIOR_VALIDATION_MIN_Z_EXTENT_RATIO
        and ground_fraction > SHAPE_PRIOR_VALIDATION_MAX_GROUND_Z_FRACTION
    )
    centroid_drifted = bool(centroid_drift > SHAPE_PRIOR_VALIDATION_MAX_CENTROID_DRIFT_M)
    invalid_reasons = []
    if clamp_suspected:
        invalid_reasons.append("positive_z_clamped_to_ground")
    if z_collapsed:
        invalid_reasons.append("z_extent_collapsed")
    if centroid_drifted:
        invalid_reasons.append("centroid_drift_exceeds_threshold")

    profile.update(
        {
            "shape_prior_coordinate_validation_status": "invalid" if invalid_reasons else "valid",
            "shape_prior_coordinate_validation_reason": ",".join(invalid_reasons) if invalid_reasons else "ok",
            "shape_prior_coordinate_validation_source_points": int(len(source_points)),
            "shape_prior_coordinate_validation_final_points": int(len(final_points)),
            "shape_prior_coordinate_validation_centroid_drift_m": centroid_drift,
            "shape_prior_coordinate_validation_z_extent_ratio": z_extent_ratio,
            "shape_prior_coordinate_validation_ground_z_fraction": ground_fraction,
            "shape_prior_source_z_min_m": float(source_min[2]),
            "shape_prior_source_z_max_m": float(source_max[2]),
            "shape_prior_source_z_extent_m": source_z_extent,
            "shape_prior_final_z_min_m": float(final_min[2]),
            "shape_prior_final_z_max_m": float(final_max[2]),
            "shape_prior_final_z_extent_m": final_z_extent,
        }
    )
    return profile


def write_futurephystwin_warmup_case(
    *,
    config: ShapePriorWarmupConfig,
    rgb_by_camera: Mapping[int, Any],
    depth_by_camera: Mapping[int, Any],
    object_mask_by_camera: Mapping[int, Any],
    controller_mask_by_camera: Mapping[int, Any],
    intrinsics_by_camera: Mapping[int, Any],
    c2w_by_camera: Mapping[int, Any],
    camera_ids: Sequence[int],
    source_group_id: int,
) -> dict[str, Any]:
    camera_order = [int(item) for item in camera_ids]
    if not camera_order:
        raise ValueError("Shape-prior warmup requires at least one camera.")
    if camera_order != [0, 1, 2]:
        raise ValueError(
            "Shape-prior warmup uses FuturePhysTwin align.py, which expects camera ids in exact order 0,1,2."
        )
    if int(config.camera_idx) not in camera_order:
        raise ValueError(
            f"Shape-prior camera {int(config.camera_idx)} is not in active camera ids {camera_order}."
        )
    missing = [
        name
        for name, mapping in (
            ("rgb", rgb_by_camera),
            ("depth", depth_by_camera),
            ("object_mask", object_mask_by_camera),
            ("controller_mask", controller_mask_by_camera),
            ("intrinsics", intrinsics_by_camera),
            ("c2w", c2w_by_camera),
        )
        for camera_idx in camera_order
        if int(camera_idx) not in mapping
    ]
    if missing:
        raise ValueError(f"Shape-prior warmup missing complete first-frame inputs: {sorted(set(missing))}")

    case_dir = config.case_dir
    if config.force and case_dir.exists():
        shutil.rmtree(case_dir)
    for relative in ("color", "mask", "pcd", "shape"):
        (case_dir / relative).mkdir(parents=True, exist_ok=True)

    points_by_camera: list[np.ndarray] = []
    colors_by_camera: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    c2ws: list[np.ndarray] = []
    processed_masks: dict[int, dict[str, np.ndarray]] = {}
    object_points: list[np.ndarray] = []
    object_colors: list[np.ndarray] = []
    controller_points: list[np.ndarray] = []
    controller_colors: list[np.ndarray] = []
    object_pixel_counts: dict[int, int] = {}
    controller_pixel_counts: dict[int, int] = {}

    reference_shape: tuple[int, int] | None = None
    for camera_idx in camera_order:
        idx = int(camera_idx)
        rgb = _as_rgb_uint8(rgb_by_camera[idx])
        depth = np.ascontiguousarray(np.asarray(depth_by_camera[idx], dtype=np.float32))
        if depth.ndim != 2:
            raise ValueError(f"Expected depth for camera {idx} to be 2D, got shape {depth.shape}.")
        if tuple(rgb.shape[:2]) != tuple(depth.shape):
            raise ValueError(
                f"RGB/depth shape mismatch for camera {idx}: {rgb.shape[:2]} vs {depth.shape}."
            )
        if reference_shape is None:
            reference_shape = tuple(depth.shape)
        elif tuple(depth.shape) != reference_shape:
            raise ValueError(
                f"FuturePhysTwin warmup case expects same HxW across cameras; camera {idx} has {depth.shape}, "
                f"expected {reference_shape}."
            )
        object_mask = _as_bool_mask(object_mask_by_camera[idx], tuple(depth.shape))
        controller_mask = _as_bool_mask(controller_mask_by_camera[idx], tuple(depth.shape))
        intr = _as_intrinsics(intrinsics_by_camera[idx])
        c2w = _as_c2w(c2w_by_camera[idx])
        world_points = _backproject_world_points(depth, intr, c2w)

        _write_rgb_png(case_dir / "color" / str(idx) / "0.png", rgb)
        _write_mask_png(case_dir / "mask" / str(idx) / "0" / "0.png", object_mask)
        _write_mask_png(case_dir / "mask" / str(idx) / "1" / "0.png", controller_mask)
        (case_dir / "mask" / f"mask_info_{idx}.json").write_text(
            json.dumps({"0": config.object_label, "1": config.controller_label}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        rgb_float = rgb.astype(np.float32)
        points_by_camera.append(world_points)
        colors_by_camera.append(rgb_float / 255.0)
        intrinsics.append(intr)
        c2ws.append(c2w)
        processed_masks[idx] = {
            "object": np.ascontiguousarray(object_mask, dtype=bool),
            "controller": np.ascontiguousarray(controller_mask, dtype=bool),
        }
        obj_pts, obj_cols = _valid_masked_points(world_points, rgb, depth, object_mask)
        ctrl_pts, ctrl_cols = _valid_masked_points(world_points, rgb, depth, controller_mask)
        object_points.append(obj_pts)
        object_colors.append(obj_cols)
        controller_points.append(ctrl_pts)
        controller_colors.append(ctrl_cols)
        object_pixel_counts[idx] = int(np.count_nonzero(object_mask))
        controller_pixel_counts[idx] = int(np.count_nonzero(controller_mask))

    object_points0 = (
        np.concatenate(object_points, axis=0).astype(np.float32, copy=False)
        if object_points
        else np.zeros((0, 3), dtype=np.float32)
    )
    object_colors0 = (
        np.concatenate(object_colors, axis=0).astype(np.float32, copy=False)
        if object_colors
        else np.zeros((0, 3), dtype=np.float32)
    )
    controller_points0 = (
        np.concatenate(controller_points, axis=0).astype(np.float32, copy=False)
        if controller_points
        else np.zeros((0, 3), dtype=np.float32)
    )
    controller_colors0 = (
        np.concatenate(controller_colors, axis=0).astype(np.float32, copy=False)
        if controller_colors
        else np.zeros((0, 3), dtype=np.float32)
    )
    controller_points0, controller_colors0 = _cap_points(
        controller_points0,
        controller_colors0,
        max_points=int(config.max_controller_points),
    )
    if len(object_points0) == 0:
        raise ValueError("Shape-prior warmup object mask/depth produced zero observed object points.")
    if len(controller_points0) == 0:
        controller_points0 = np.zeros((1, 3), dtype=np.float32)
        controller_colors0 = np.zeros((1, 3), dtype=np.float32)

    metadata = {
        "schema": "qqtt_demo33_futurephystwin_single_view_warmup_v1",
        "source": "demo3.3_shape_prior_warmup",
        "source_group_id": int(source_group_id),
        "camera_ids": camera_order,
        "intrinsics": [item.astype(float).tolist() for item in intrinsics],
        "shape_prior_camera_idx": int(config.camera_idx),
        "shape_prior_coordinate_frame": DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME,
        "shape_prior_units": DEFAULT_SHAPE_PRIOR_UNITS,
        "shape_prior_ground_policy": str(config.ground_policy),
        "shape_prior_ground_z": float(config.ground_z),
        "object_label": config.object_label,
        "controller_label": config.controller_label,
        "frame_count": 1,
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    with (case_dir / "calibrate.pkl").open("wb") as handle:
        pickle.dump(c2ws, handle)
    np.savez_compressed(
        case_dir / "pcd" / "0.npz",
        points=np.stack(points_by_camera, axis=0).astype(np.float32),
        colors=np.stack(colors_by_camera, axis=0).astype(np.float32),
    )
    with (case_dir / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump([processed_masks], handle)
    track_data = {
        "object_points": object_points0[None, ...],
        "object_colors": object_colors0[None, ...],
        "object_visibilities": np.ones((1, len(object_points0)), dtype=bool),
        "object_motions_valid": np.ones((1, len(object_points0)), dtype=bool),
        "controller_points": controller_points0[None, ...],
        "controller_colors": controller_colors0[None, ...],
        "controller_visibilities": np.ones((1, len(controller_points0)), dtype=bool),
        "controller_motions_valid": np.ones((1, len(controller_points0)), dtype=bool),
        "source_group_id": int(source_group_id),
        "shape_prior_coordinate_frame": DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME,
        "shape_prior_units": DEFAULT_SHAPE_PRIOR_UNITS,
        "shape_prior_ground_policy": str(config.ground_policy),
        "shape_prior_ground_z": float(config.ground_z),
    }
    with (case_dir / "track_process_data.pkl").open("wb") as handle:
        pickle.dump(track_data, handle)

    return {
        "shape_prior_case_dir": str(case_dir),
        "shape_prior_source_group_id": int(source_group_id),
        "shape_prior_camera_idx": int(config.camera_idx),
        "shape_prior_coordinate_frame": DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME,
        "shape_prior_units": DEFAULT_SHAPE_PRIOR_UNITS,
        "shape_prior_ground_policy": str(config.ground_policy),
        "shape_prior_ground_z": float(config.ground_z),
        "shape_prior_object_pixels_by_camera": object_pixel_counts,
        "shape_prior_controller_pixels_by_camera": controller_pixel_counts,
        "shape_prior_object_points0": int(len(object_points0)),
        "shape_prior_controller_points_capped": int(len(controller_points0)),
        "shape_prior_affects_tracker_input": False,
        "shape_prior_affects_live_observation_pcd": False,
    }


def futurephystwin_single_view_commands(config: ShapePriorWarmupConfig) -> list[tuple[str, list[str]]]:
    case_dir = config.case_dir
    base_path = str(config.run_root)
    case_name = case_dir.name
    root = Path(config.futurephystwin_root)
    sam3d_root = validate_original_sam3d_root(Path(config.sam3d_root))
    python_cmd = shlex.split(str(config.futurephystwin_python))
    shape_dir = case_dir / "shape"
    high_resolution = shape_dir / "high_resolution.png"
    masked_image = shape_dir / "masked_image.png"
    object_mask = case_dir / "mask" / str(int(config.camera_idx)) / "0" / "0.png"
    commands: list[tuple[str, list[str]]] = [
        (
            "image_upscale",
            [
                *python_cmd,
                str(root / "data_process" / "image_upscale.py"),
                "--img_path",
                str(case_dir / "color" / str(int(config.camera_idx)) / "0.png"),
                "--mask_path",
                str(object_mask),
                "--output_path",
                str(high_resolution),
                "--category",
                str(config.object_label),
            ],
        ),
        (
            "segment_util_image",
            [
                *python_cmd,
                str(root / "data_process" / "segment_util_image.py"),
                "--img_path",
                str(high_resolution),
                "--TEXT_PROMPT",
                str(config.object_label),
                "--output_path",
                str(masked_image),
            ],
        ),
        (
            "shape_prior_sam3d",
            [
                *python_cmd,
                str(root / "data_process_sam3d" / "shape_prior.py"),
                "--img_path",
                str(masked_image),
                "--output_dir",
                str(shape_dir),
                "--sam3d_root",
                str(sam3d_root),
            ],
        ),
        (
            "align",
            [
                *python_cmd,
                str(root / "data_process" / "align.py"),
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--controller_name",
                str(config.controller_label),
            ],
        ),
        (
            "data_process_sample",
            [
                *python_cmd,
                str(root / "data_process_sam3d" / "data_process_sample.py"),
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--shape_prior",
                "--ground-policy",
                str(config.ground_policy),
                "--ground-z",
                str(float(config.ground_z)),
            ],
        ),
    ]
    if config.force:
        for stage, command in commands:
            if stage == "align":
                command.append("--force_rematch")
    if config.skip_route_visualizations:
        for stage, command in commands:
            if stage in {"shape_prior_sam3d", "align", "data_process_sample"}:
                command.append("--skip_visualization")
    return commands


def run_futurephystwin_single_view_route(
    *,
    config: ShapePriorWarmupConfig,
    runner: SubprocessRunner = subprocess.run,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    subprocess_env: dict[str, str] | None = None
    cuda_visible_devices = config.cuda_visible_devices
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        subprocess_env = dict(os.environ)
        subprocess_env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    if config.cuda_allocator_config is not None and str(config.cuda_allocator_config).strip():
        if subprocess_env is None:
            subprocess_env = dict(os.environ)
        subprocess_env["PYTORCH_CUDA_ALLOC_CONF"] = str(config.cuda_allocator_config).strip()
    for stage, command in futurephystwin_single_view_commands(config):
        start_s = time.perf_counter()
        runner_kwargs: dict[str, Any] = {"cwd": str(config.futurephystwin_root), "check": True}
        if subprocess_env is not None:
            runner_kwargs["env"] = subprocess_env
        runner(command, **runner_kwargs)
        records.append(
            {
                "stage": stage,
                "command": list(command),
                "elapsed_ms": float((time.perf_counter() - start_s) * 1000.0),
                "cuda_visible_devices": str(cuda_visible_devices or ""),
                "cuda_allocator_config": str(config.cuda_allocator_config or ""),
            }
        )
    return records


def load_shape_prior_final_data(case_dir: Path) -> ShapePriorWarmupResult:
    final_path = Path(case_dir) / "final_data.pkl"
    if not final_path.is_file():
        raise FileNotFoundError(f"Missing FuturePhysTwin final_data.pkl: {final_path}")
    with final_path.open("rb") as handle:
        final_data = pickle.load(handle)
    object_points = np.asarray(final_data.get("object_points"), dtype=np.float32)
    if object_points.ndim != 3 or object_points.shape[0] < 1 or object_points.shape[2] != 3:
        raise ValueError(f"final_data.pkl object_points must be FxNx3, got {object_points.shape}.")
    object_points0 = np.ascontiguousarray(object_points[0], dtype=np.float32)
    surface_points = np.ascontiguousarray(
        np.asarray(final_data.get("surface_points", np.zeros((0, 3))), dtype=np.float32).reshape(-1, 3)
    )
    interior_points = np.ascontiguousarray(
        np.asarray(final_data.get("interior_points", np.zeros((0, 3))), dtype=np.float32).reshape(-1, 3)
    )
    structure_points = np.ascontiguousarray(
        np.concatenate([object_points0, surface_points, interior_points], axis=0),
        dtype=np.float32,
    )
    metadata = _load_case_metadata(Path(case_dir))
    coordinate_frame = str(metadata.get("shape_prior_coordinate_frame", DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME))
    units = str(metadata.get("shape_prior_units", DEFAULT_SHAPE_PRIOR_UNITS))
    ground_policy = str(
        final_data.get(
            "shape_prior_ground_policy",
            metadata.get("shape_prior_ground_policy", DEFAULT_SHAPE_PRIOR_GROUND_POLICY),
        )
    )
    ground_z = float(
        final_data.get(
            "shape_prior_ground_z",
            metadata.get("shape_prior_ground_z", DEFAULT_SHAPE_PRIOR_GROUND_Z),
        )
    )
    validation_profile = _validate_shape_prior_coordinates(
        case_dir=Path(case_dir),
        object_points0=object_points0,
        ground_z=ground_z,
    )
    validation_status = str(validation_profile.get("shape_prior_coordinate_validation_status", "unavailable"))
    render_enabled = validation_status != "invalid"
    raw_structure_count = int(len(structure_points))
    if not render_enabled:
        structure_points = np.zeros((0, 3), dtype=np.float32)
    colors = np.tile(
        np.asarray(DEFAULT_SHAPE_PRIOR_RENDER_COLOR_RGB, dtype=np.uint8).reshape(1, 3),
        (len(structure_points), 1),
    )
    profile = {
        "shape_prior_status": "ready" if render_enabled else "invalid_coordinate_policy",
        "shape_prior_case_dir": str(case_dir),
        "shape_prior_source_group_id": int(metadata.get("source_group_id", -1)),
        "shape_prior_coordinate_frame": coordinate_frame,
        "shape_prior_units": units,
        "shape_prior_ground_policy": ground_policy,
        "shape_prior_ground_z": ground_z,
        "shape_prior_object_points0": int(len(object_points0)),
        "shape_prior_surface_points": int(len(surface_points)),
        "shape_prior_interior_points": int(len(interior_points)),
        "shape_prior_structure_points": int(len(structure_points)),
        "shape_prior_raw_structure_points": raw_structure_count,
        "shape_prior_render_layer_enabled": bool(render_enabled),
        "shape_prior_affects_tracker_input": False,
        "shape_prior_affects_live_observation_pcd": False,
    }
    profile.update(validation_profile)
    return ShapePriorWarmupResult(
        status=str(profile["shape_prior_status"]),
        case_dir=Path(case_dir),
        object_points0=object_points0,
        surface_points=surface_points,
        interior_points=interior_points,
        structure_points=structure_points,
        structure_colors_rgb=np.ascontiguousarray(colors, dtype=np.uint8),
        profile=profile,
    )


def run_shape_prior_warmup(
    *,
    config: ShapePriorWarmupConfig,
    rgb_by_camera: Mapping[int, Any],
    depth_by_camera: Mapping[int, Any],
    object_mask_by_camera: Mapping[int, Any],
    controller_mask_by_camera: Mapping[int, Any],
    intrinsics_by_camera: Mapping[int, Any],
    c2w_by_camera: Mapping[int, Any],
    camera_ids: Sequence[int],
    source_group_id: int,
    runner: SubprocessRunner = subprocess.run,
) -> ShapePriorWarmupResult:
    profile: dict[str, Any] = {
        "shape_prior_warmup_enabled": bool(config.enabled),
        "shape_prior_status": "disabled" if not config.enabled else "starting",
        "shape_prior_case_dir": str(config.case_dir),
        "shape_prior_source_group_id": int(source_group_id),
        "shape_prior_coordinate_frame": DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME,
        "shape_prior_units": DEFAULT_SHAPE_PRIOR_UNITS,
        "shape_prior_ground_policy": str(config.ground_policy),
        "shape_prior_ground_z": float(config.ground_z),
        "shape_prior_coordinate_validation_status": "disabled" if not config.enabled else "pending",
        "shape_prior_coordinate_validation_reason": "",
        "shape_prior_raw_structure_points": 0,
        "shape_prior_render_layer_enabled": False,
        "shape_prior_affects_tracker_input": False,
        "shape_prior_affects_live_observation_pcd": False,
    }
    if not config.enabled:
        return ShapePriorWarmupResult(
            status="disabled",
            case_dir=config.case_dir,
            object_points0=np.zeros((0, 3), dtype=np.float32),
            surface_points=np.zeros((0, 3), dtype=np.float32),
            interior_points=np.zeros((0, 3), dtype=np.float32),
            structure_points=np.zeros((0, 3), dtype=np.float32),
            structure_colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            profile=profile,
        )

    case_profile = write_futurephystwin_warmup_case(
        config=config,
        rgb_by_camera=rgb_by_camera,
        depth_by_camera=depth_by_camera,
        object_mask_by_camera=object_mask_by_camera,
        controller_mask_by_camera=controller_mask_by_camera,
        intrinsics_by_camera=intrinsics_by_camera,
        c2w_by_camera=c2w_by_camera,
        camera_ids=camera_ids,
        source_group_id=source_group_id,
    )
    command_records = run_futurephystwin_single_view_route(config=config, runner=runner)
    result = load_shape_prior_final_data(config.case_dir)
    profile.update(case_profile)
    profile.update(result.profile)
    profile.update(
        {
            "shape_prior_warmup_enabled": True,
            "shape_prior_command_records": command_records,
            "shape_prior_command_order": [record["stage"] for record in command_records],
        }
    )
    return ShapePriorWarmupResult(
        status=result.status,
        case_dir=result.case_dir,
        object_points0=result.object_points0,
        surface_points=result.surface_points,
        interior_points=result.interior_points,
        structure_points=result.structure_points,
        structure_colors_rgb=result.structure_colors_rgb,
        profile=profile,
    )


__all__ = [
    "DEFAULT_FUTUREPHYSTWIN_PYTHON",
    "DEFAULT_FUTUREPHYSTWIN_ROOT",
    "DEFAULT_SAM3D_ROOT",
    "DEFAULT_SHAPE_PRIOR_CAMERA_IDX",
    "ShapePriorWarmupConfig",
    "ShapePriorWarmupResult",
    "futurephystwin_single_view_commands",
    "load_shape_prior_final_data",
    "run_futurephystwin_single_view_route",
    "run_shape_prior_warmup",
    "validate_original_sam3d_root",
    "write_futurephystwin_warmup_case",
]
