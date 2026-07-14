"""Frame-0 shape-prior case serialization (offline-style case directory)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ShapePriorFrame0Request:
    """Frame-0 capture snapshot needed to build an offline-style case dir."""

    seq: int
    source_timestamp_s: float | None
    input_source: str
    depth_backend: str
    depth_source_internal: str
    rgb_u8: np.ndarray
    object_mask: np.ndarray
    controller_mask: np.ndarray
    depth_color_m: np.ndarray
    depth_valid_mask: np.ndarray
    points_world_m: np.ndarray
    k_color: np.ndarray
    camera_to_world_c2w: np.ndarray
    warmup_runtime_start_perf_s: float | None = None
    frame_receive_perf_s: float | None = None
    frame_mask_ready_perf_s: float | None = None
    frame_pcd_ready_perf_s: float | None = None
    frame0_pipeline_timing_ms: dict[str, float] = field(default_factory=dict)
    frame0_perception_profile: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input validation and small IO helpers
# ---------------------------------------------------------------------------


def _as_mask(value: np.ndarray, *, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return the as mask."""
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match RGB shape {shape}")
    return np.ascontiguousarray(mask)


def require_name(value: str, *, field_name: str) -> str:
    """Return validated name."""
    name = str(value).strip()
    if not name:
        raise ValueError(f"shape prior {field_name} must be non-empty")
    return name


def _write_mask(mask: np.ndarray, path: Path) -> None:
    """Write a boolean mask image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def points_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return the points array."""
    points = np.asarray(value, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3")
    return np.ascontiguousarray(points, dtype=np.float32)


def write_shape_prior_points_npz(
    path: str | Path,
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
) -> Path:
    """Write shape prior points NPZ."""
    output_path = Path(path)
    surface = points_array(surface_points, name="surface_points")
    interior = points_array(interior_points, name="interior_points")
    points = np.concatenate([surface, interior], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        surface_points=surface,
        interior_points=interior,
        points=np.ascontiguousarray(points, dtype=np.float32),
    )
    return output_path


def write_shape_prior_case(
    frame0: ShapePriorFrame0Request,
    *,
    case_root: Path,
    case_name: str,
    object_name: str,
    controller_name: str,
) -> dict[str, Path]:
    """Serialize frame 0 as a one-frame, one-camera offline-style case dir.

    The directory layout (color/, mask/, pcd/, calibrate.pkl, metadata.json,
    processed_masks.pkl, track_process_data.pkl) mirrors what the original
    PhysTwin data_process_origin scripts expect, with camera index 0 and
    frame index 0.
    """
    object_name = require_name(object_name, field_name="object_name")
    controller_name = require_name(controller_name, field_name="controller_name")

    rgb = np.asarray(frame0.rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("shape prior rgb_u8 must have shape HxWx3")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    image_shape = tuple(rgb.shape[:2])
    object_mask = _as_mask(frame0.object_mask, shape=image_shape, name="object_mask")
    if not np.any(object_mask):
        raise ValueError("shape prior object_mask is empty")
    controller_mask = _as_mask(
        frame0.controller_mask,
        shape=image_shape,
        name="controller_mask",
    )
    depth_m = np.asarray(frame0.depth_color_m, dtype=np.float32)
    if depth_m.shape != image_shape:
        raise ValueError(
            f"depth shape {depth_m.shape} does not match RGB shape {image_shape}"
        )
    depth_m = np.ascontiguousarray(depth_m)
    depth_valid = _as_mask(
        frame0.depth_valid_mask,
        shape=image_shape,
        name="depth_valid_mask",
    )
    points_world = np.asarray(frame0.points_world_m, dtype=np.float32)
    if points_world.shape != (*image_shape, 3):
        raise ValueError(
            "shape prior points_world_m must have shape "
            f"{(*image_shape, 3)}; got {points_world.shape}"
        )
    points_world = np.ascontiguousarray(points_world)
    # Masks are guaranteed depth-valid subsets upstream by
    # ProcessedFramePacket.__post_init__ (mdp_packets), so no re-check here.
    k_color = np.asarray(frame0.k_color, dtype=np.float32).reshape(3, 3)
    c2w = np.asarray(frame0.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)
    if not np.isfinite(k_color).all():
        raise ValueError("shape prior color intrinsics must be finite")
    if not np.isfinite(c2w).all():
        raise ValueError("shape prior camera-to-world transform must be finite")
    if not np.isfinite(points_world[object_mask | controller_mask]).all():
        raise ValueError("shape prior processed masks contain non-finite 3D points")

    case = Path(case_root) / str(case_name)
    color_path = case / "color" / "0" / "0.png"
    shape_dir = case / "shape"
    object_mask_path = case / "mask" / "0" / "0" / "0.png"
    controller_mask_path = case / "mask" / "0" / "1" / "0.png"
    pcd_path = case / "pcd" / "0.npz"

    shape_dir.mkdir(parents=True, exist_ok=True)
    color_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(color_path)
    _write_mask(object_mask, object_mask_path)
    _write_mask(controller_mask, controller_mask_path)
    _write_json(
        {"0": object_name, "1": controller_name},
        case / "mask" / "mask_info_0.json",
    )
    _write_json(
        {
            "frame_num": 1,
            "intrinsics": [k_color.tolist()],
            "input_source": str(frame0.input_source),
            "depth_backend": str(frame0.depth_backend),
            "depth_source_internal": str(frame0.depth_source_internal),
        },
        case / "metadata.json",
    )
    with (case / "calibrate.pkl").open("wb") as handle:
        pickle.dump([c2w], handle, protocol=pickle.HIGHEST_PROTOCOL)

    object_points = points_world[object_mask]
    object_colors = rgb[object_mask].astype(np.float32) / 255.0
    if object_points.size == 0:
        raise ValueError("shape prior object observation has no valid depth points")

    controller_points = points_world[controller_mask]
    if controller_points.size == 0:
        raise ValueError("shape prior controller observation has no valid points")

    # pcd/0.npz keeps the dense HxW grid (leading axis = camera index) so the
    # align stage can index it with the processed masks.
    pcd_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pcd_path,
        points=np.ascontiguousarray(points_world[None], dtype=np.float32),
        colors=np.ascontiguousarray((rgb[None].astype(np.float32) / 255.0)),
        masks=np.ascontiguousarray(depth_valid[None], dtype=bool),
    )
    with (case / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(
            [[{"object": object_mask, "controller": controller_mask}]],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with (case / "track_process_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "object_points": object_points[None].astype(np.float32),
                "object_colors": object_colors[None].astype(np.float32),
                "object_visibilities": np.ones((1, object_points.shape[0]), dtype=bool),
                "object_motions_valid": np.ones(
                    (1, object_points.shape[0]), dtype=bool
                ),
                "controller_points": controller_points[None].astype(np.float32),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {
        "case": case,
        "color": color_path,
        "object_mask": object_mask_path,
        "shape": shape_dir,
        "pcd": pcd_path,
        "track_process": case / "track_process_data.pkl",
    }


__all__ = [
    "ShapePriorFrame0Request",
    "write_shape_prior_case",
]
