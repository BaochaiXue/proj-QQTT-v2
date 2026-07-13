"""Strict PhysTwin-compatible tracking product for Demo v6.1.

Builds, from a single-camera realtime capture, the same on-disk layout the
offline data_process_origin pipeline produces (processed masks, dense world
PCDs, per-frame tracks, track_process_data/final_data pickles) so PhysTwin
consumes a realtime capture unchanged. Comments tagged "offline parity" map
each step to its origin counterpart; keep them verbatim next to the code they
describe.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2.utils.depth_geometry import transform_points
from demo_v6_2.utils.pcd_postprocess import detect_radius_outlier_indices
from demo_v6_2.utils.projection import build_projection_grid_from_matrix
from demo_v6_2.utils.render import (
    _load_rgb,
    _render_empty_video,
    _render_tracking_2d_video,
    _render_world_track_video,
)


TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY = "realtime-overlay"
TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT = "phystwin-strict-tracking"
TRACKING_PRODUCT_BACKENDS = (
    TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
    TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
)
DEFAULT_TRACKING_PRODUCT_BACKEND = TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY

COMPATIBILITY_TARGET_PHYSTWIN = "PhysTwin"
PHYSTWIN_STRICT_EXECUTION_MODE = "workstation_strict"
PHYSTWIN_COMPATIBILITY_PATH_NAME = "cotracker"
PHYSTWIN_RADIUS_OUTLIER_RADIUS_M = 0.01
PHYSTWIN_RADIUS_OUTLIER_NB_POINTS = 40
PHYSTWIN_DEPTH_MIN_M = 0.2
PHYSTWIN_DEPTH_MAX_M = 1.5


@dataclass(frozen=True)
class PreparedPhysTwinFrame:
    """One fully-processed frame of the strict product.

    Units/layout: `pcd_points` is a world-space grid in meters shaped
    (1, H, W, 3) with invalid pixels zeroed; `pcd_colors` is the matching
    (1, H, W, 3) uint8 RGB grid; `tracks_yx`/`query_points_yx` are pixel
    coordinates in (y, x) order with one row per query. The `source_*`
    fields are optional capture provenance.
    """

    seq: int
    rgb_frame: np.ndarray
    processed_mask_frame: Mapping[str, np.ndarray]
    pcd_points: np.ndarray
    pcd_colors: np.ndarray
    tracks_yx: np.ndarray
    visibility: np.ndarray
    query_points_yx: np.ndarray
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None
    # Color-aligned raw depth, (H, W) uint16 integer millimeters, invalid = 0.
    # Canonical online_data depth format for every backend: RealSense units at
    # the standard 0.001 m/unit scale round-trip bit-exactly, FFS float meters
    # quantize through the same depth_m_to_mm_u16 conversion. None only for
    # legacy npz files written before the online_data color/depth contract.
    depth_mm_u16: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Tracking product backend selection
# ---------------------------------------------------------------------------


def normalize_tracking_product_backend(value: str | None) -> str:
    # Accept hyphen/underscore and shorthand spellings from CLI flags and
    # config files; everything maps onto the two canonical backend names.
    """Normalize tracking product backend."""
    normalized = (
        str(value or DEFAULT_TRACKING_PRODUCT_BACKEND).strip().lower().replace("_", "-")
    )
    aliases = {
        "overlay": TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
        "realtime": TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
        "tapnextpp-overlay": TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
        "phystwin-strict": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "phys-twin-strict": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "phystwin-strict-tracking": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRACKING_PRODUCT_BACKENDS:
        raise ValueError(
            f"unsupported tracking product backend {value!r}; expected one of "
            f"{TRACKING_PRODUCT_BACKENDS}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Processed mask frames (object / controller / optional per-hand)
# ---------------------------------------------------------------------------


def normalize_processed_mask_frame(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
    # Demo v6.1 offline parity: data_process_sam3d/data_process_mask.py:L56-L80
    # keeps object and controller masks independent — origin never subtracts
    # the controller mask from the object mask, so overlap pixels stay valid
    # for both classes.
    """Normalize processed mask frame."""
    if "object" in frame and frame["object"] is not None:
        obj = np.asarray(frame["object"], dtype=bool)
    else:
        raise ValueError("missing required mask key 'object'")
    if "controller" in frame and frame["controller"] is not None:
        ctrl = np.asarray(frame["controller"], dtype=bool)
    else:
        # No explicit controller mask: the controller is the union of the two
        # hand masks (either hand may be absent).
        hand_a = np.asarray(
            frame.get("hand_a", np.zeros_like(obj, dtype=bool)), dtype=bool
        )
        hand_b = np.asarray(
            frame.get("hand_b", np.zeros_like(obj, dtype=bool)), dtype=bool
        )
        ctrl = np.logical_or(hand_a, hand_b)
    if obj.shape != ctrl.shape:
        raise ValueError("object/controller masks must have the same shape")
    out = {
        "object": np.ascontiguousarray(obj, dtype=bool),
        "controller": np.ascontiguousarray(ctrl, dtype=bool),
    }
    # Per-hand identity masks ride along when present so downstream consumers
    # can still tell the hands apart. The canonical PT step later intersects
    # them with the cleaned combined-controller mask.
    if "hand_a" in frame and frame["hand_a"] is not None:
        out["hand_a"] = np.ascontiguousarray(
            np.asarray(frame["hand_a"], dtype=bool), dtype=bool
        )
    if "hand_b" in frame and frame["hand_b"] is not None:
        out["hand_b"] = np.ascontiguousarray(
            np.asarray(frame["hand_b"], dtype=bool), dtype=bool
        )
    return out


def write_processed_masks(
    output_dir: str | Path, frames: Sequence[Mapping[str, Any]]
) -> Path:
    """Write processed masks."""
    root = Path(output_dir)
    mask_dir = root / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    # Origin layout is processed[frame][camera]; this demo is single-camera,
    # hence the one-element inner lists.
    processed = [[normalize_processed_mask_frame(frame)] for frame in frames]
    path = mask_dir / "processed_masks.pkl"
    with path.open("wb") as handle:
        pickle.dump(processed, handle)
    return path


# ---------------------------------------------------------------------------
# Dense world-space point cloud lifting
# ---------------------------------------------------------------------------


def _depth_validity_mask(depth: np.ndarray) -> np.ndarray:
    """Origin depth gate: finite and strictly inside (MIN, MAX) meters."""
    return (
        np.isfinite(depth)
        & (depth > np.float32(PHYSTWIN_DEPTH_MIN_M))
        & (depth < np.float32(PHYSTWIN_DEPTH_MAX_M))
    )


def _intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    """Coerce intrinsics to a 3x3 pinhole K matrix (float32).

    Accepts a mapping with fx/fy/cx/cy, an object exposing those attributes
    (e.g. a pyrealsense2 intrinsics struct), or anything reshapeable to 3x3.
    """
    if isinstance(intrinsics, Mapping):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    if all(hasattr(intrinsics, name) for name in ("fx", "fy", "cx", "cy")):
        return np.array(
            [
                [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                [0.0, float(intrinsics.fy), 0.0 + float(intrinsics.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)


def dense_world_pcd_grid(
    *,
    depth_m: np.ndarray,
    color_rgb_u8: np.ndarray,
    intrinsics: Any,
    c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Demo v6.1 offline parity: data_process_sam3d/data_process_pcd.py:L84-L149
    # lifts RGB-D pixels through intrinsics and camera_to_world into a
    # world-space PCD grid.
    """Build dense world PCD grid."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW; got {depth.shape}")
    color = np.asarray(color_rgb_u8, dtype=np.uint8)
    if color.shape[:2] != depth.shape or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color_rgb_u8 must have shape HxWx3 matching depth_m")
    height, width = depth.shape
    K = _intrinsics_to_matrix(intrinsics)
    c2w_matrix = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    if not np.isfinite(K).all():
        raise ValueError("camera intrinsics must be finite")
    if float(K[0, 0]) == 0.0 or float(K[1, 1]) == 0.0:
        raise ValueError("camera focal lengths must be nonzero")
    if not np.isfinite(c2w_matrix).all():
        raise ValueError("camera-to-world transform must be finite")
    ray_x, ray_y = build_projection_grid_from_matrix(width=width, height=height, K=K)
    valid = _depth_validity_mask(depth)

    # Invalid pixels stay exactly (0, 0, 0) to preserve the origin dense-grid
    # layout. Canonical processed masks exclude them through the depth gate.
    points = np.zeros((height, width, 3), dtype=np.float32)
    if np.any(valid):
        rows, cols = np.nonzero(valid)
        z = depth[rows, cols]
        points_camera = np.stack(
            [
                ray_x[rows, cols].astype(np.float32, copy=False) * z,
                ray_y[rows, cols].astype(np.float32, copy=False) * z,
                z,
            ],
            axis=1,
        ).astype(np.float32)
        points_world = transform_points(points_camera, c2w_matrix).astype(np.float32)
        points[rows, cols] = points_world
    # Leading axis is the camera axis (single camera here), matching the
    # origin per-frame pcd layout of (num_cameras, H, W, 3).
    return points[None].astype(np.float32, copy=False), color[None].astype(
        np.uint8, copy=False
    )


def apply_depth_validity_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    depth_m: np.ndarray,
) -> dict[str, np.ndarray]:
    # Demo v6.1 offline parity: data_process_sam3d/data_process_mask.py:L56-L80
    # intersects semantic masks with valid depth support.
    """Apply depth validity to mask frame."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = _depth_validity_mask(depth)
    normalized = normalize_processed_mask_frame(frame)
    filtered: dict[str, np.ndarray] = {}
    for key, mask in normalized.items():
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != valid.shape:
            raise ValueError(
                f"mask {key!r} shape {arr.shape} does not match depth shape {valid.shape}"
            )
        filtered[key] = np.ascontiguousarray(arr & valid, dtype=bool)
    return normalize_processed_mask_frame(filtered)


def apply_radius_outlier_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    points_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    # Demo v6.1 offline parity: data_process_sam3d/data_process_mask.py:L81-L92
    # and L107-L136 remove isolated 3D mask points before processed mask output.
    """Apply radius outlier to mask frame."""
    normalized = normalize_processed_mask_frame(frame)
    grid = np.asarray(points_grid, dtype=np.float32)
    if grid.ndim == 4:
        grid = grid[0]
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(
            f"points_grid must have shape H,W,3 or 1,H,W,3; got {grid.shape}"
        )

    # Origin filters object and the combined controller independently. It does
    # not resolve pixels shared by both masks; preserving that behavior avoids
    # inventing a class-priority rule, although overlapping pixels make tracker
    # identity ambiguous and should remain visible in diagnostics.
    filtered = {
        key: np.asarray(value, dtype=bool).copy() for key, value in normalized.items()
    }
    for key in ("object", "controller"):
        mask = filtered[key]
        if mask.shape != grid.shape[:2]:
            raise ValueError(
                f"mask {key!r} shape {mask.shape} does not match points grid {grid.shape[:2]}"
            )
        yy, xx = np.nonzero(mask)
        if len(yy) == 0:
            continue
        class_points = grid[yy, xx]
        # Depth-invalid pixels were removed before this function. A legitimate
        # world-space point may be exactly at the world origin, so zero norm is
        # not an invalidity test here.
        finite = np.isfinite(class_points).all(axis=1)
        if not np.all(finite):
            invalid_rows = yy[~finite]
            invalid_cols = xx[~finite]
            filtered[key][invalid_rows, invalid_cols] = False
            yy = yy[finite]
            xx = xx[finite]
            class_points = class_points[finite]
        if len(class_points) == 0:
            continue
        result = detect_radius_outlier_indices(
            class_points,
            radius_m=PHYSTWIN_RADIUS_OUTLIER_RADIUS_M,
            nb_points=PHYSTWIN_RADIUS_OUTLIER_NB_POINTS,
        )
        outlier_indices = np.asarray(result["outlier_indices"], dtype=np.int64)
        if len(outlier_indices):
            filtered[key][yy[outlier_indices], xx[outlier_indices]] = False

    # EdgeTAM tracks each hand separately, while origin/PT treats their union
    # as the controller. Preserve the per-hand identity by intersecting each
    # hand with the one canonical cleaned controller mask.
    for key in ("hand_a", "hand_b"):
        if key not in filtered:
            continue
        if filtered[key].shape != filtered["controller"].shape:
            raise ValueError(
                f"mask {key!r} shape {filtered[key].shape} does not match "
                f"controller mask shape {filtered['controller'].shape}"
            )
        filtered[key] &= filtered["controller"]
    return normalize_processed_mask_frame(filtered)


# ---------------------------------------------------------------------------
# Prepared per-frame product (build + npz round-trip)
# ---------------------------------------------------------------------------


def depth_m_to_mm_u16(depth_m: np.ndarray) -> np.ndarray:
    """Convert metric depth to the canonical (H, W) uint16-millimeter frame.

    Non-finite, non-positive, and uint16-overflowing pixels (> 65.535 m —
    FFS far-field garbage from near-zero disparity; unreachable for RealSense
    raw units) all map to 0, the shared invalid sentinel; everything else
    rounds to the nearest millimeter. RealSense uint16 units at the standard
    0.001 m/unit scale survive the units->meters->millimeters round trip
    bit-exactly, so archiving through this conversion equals a direct copy of
    the aligned raw frame; FFS float meters quantize to the identical
    downstream format.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be (H, W), got shape {depth.shape}")
    mm = np.rint(depth.astype(np.float64) * 1000.0)
    invalid = ~np.isfinite(mm)
    invalid |= mm < 0.0
    invalid |= mm > float(np.iinfo(np.uint16).max)
    mm[invalid] = 0.0
    return np.ascontiguousarray(mm.astype(np.uint16))


def prepare_phystwin_frame(
    *,
    seq: int,
    rgb_frame: np.ndarray,
    depth_m: np.ndarray,
    processed_mask_frame: Mapping[str, np.ndarray],
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_points_yx: np.ndarray,
    source_timestamp_s: float | None = None,
    source_frame_index: int | None = None,
    source_step: int | None = None,
) -> PreparedPhysTwinFrame:
    """Validate and package one already-processed canonical frame."""
    rgb = np.ascontiguousarray(np.asarray(rgb_frame, dtype=np.uint8))
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have shape HxW; got {depth.shape}")
    if rgb.shape != (*depth.shape, 3):
        raise ValueError(
            f"rgb_frame shape {rgb.shape} does not match depth shape {depth.shape}"
        )
    points = np.ascontiguousarray(np.asarray(pcd_points, dtype=np.float32))
    colors = np.ascontiguousarray(np.asarray(pcd_colors, dtype=np.uint8))
    expected_grid_shape = (1, *depth.shape, 3)
    if points.shape != expected_grid_shape:
        raise ValueError(
            f"pcd_points must have shape {expected_grid_shape}; got {points.shape}"
        )
    if colors.shape != expected_grid_shape:
        raise ValueError(
            f"pcd_colors must have shape {expected_grid_shape}; got {colors.shape}"
        )
    processed = normalize_processed_mask_frame(processed_mask_frame)
    depth_valid = _depth_validity_mask(depth)
    points_grid = points[0]
    for key, mask in processed.items():
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != depth.shape:
            raise ValueError(
                f"processed mask {key!r} shape {mask_bool.shape} "
                f"does not match depth shape {depth.shape}"
            )
        if np.any(mask_bool & ~depth_valid):
            raise ValueError(f"processed mask {key!r} contains depth-invalid pixels")
        if not np.isfinite(points_grid[mask_bool]).all():
            raise ValueError(f"processed mask {key!r} contains non-finite 3D points")
    if not np.any(processed["object"]):
        raise ValueError("processed object mask is empty")
    if not np.any(processed["controller"]):
        raise ValueError("processed controller mask is empty")
    tracks = np.ascontiguousarray(
        np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    )
    vis = np.ascontiguousarray(np.asarray(visibility, dtype=bool).reshape(-1))
    queries = np.ascontiguousarray(
        np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    )
    if tracks.shape[0] != queries.shape[0] or vis.shape[0] != queries.shape[0]:
        raise ValueError(
            "prepared PhysTwin frame requires full tracks/visibility matching query_points_yx; "
            f"tracks={tracks.shape[0]} visibility={vis.shape[0]} queries={queries.shape[0]}"
        )
    return PreparedPhysTwinFrame(
        seq=int(seq),
        rgb_frame=rgb,
        processed_mask_frame=processed,
        pcd_points=np.ascontiguousarray(points, dtype=np.float32),
        pcd_colors=np.ascontiguousarray(colors, dtype=np.uint8),
        tracks_yx=tracks,
        visibility=vis,
        query_points_yx=queries,
        source_timestamp_s=None
        if source_timestamp_s is None
        else float(source_timestamp_s),
        source_frame_index=None
        if source_frame_index is None
        else int(source_frame_index),
        source_step=None if source_step is None else int(source_step),
        depth_mm_u16=depth_m_to_mm_u16(depth),
    )


def write_prepared_phystwin_frame(
    path: str | Path, frame: PreparedPhysTwinFrame
) -> Path:
    """Write prepared phystwin frame."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp")
    masks = normalize_processed_mask_frame(frame.processed_mask_frame)
    mask_keys = np.asarray(sorted(masks.keys()))
    # The npz is read back with allow_pickle=False, so None provenance fields
    # are stored as NaN / -1 sentinels; load_prepared_phystwin_frame reverses
    # the encoding.
    payload: dict[str, Any] = {
        "seq": np.asarray([int(frame.seq)], dtype=np.int64),
        "rgb_frame": np.ascontiguousarray(frame.rgb_frame, dtype=np.uint8),
        "pcd_points": np.ascontiguousarray(frame.pcd_points, dtype=np.float32),
        "pcd_colors": np.ascontiguousarray(frame.pcd_colors, dtype=np.uint8),
        "tracks_yx": np.ascontiguousarray(frame.tracks_yx, dtype=np.float32),
        "visibility": np.ascontiguousarray(frame.visibility, dtype=bool),
        "query_points_yx": np.ascontiguousarray(
            frame.query_points_yx, dtype=np.float32
        ),
        "mask_keys": mask_keys,
        "source_timestamp_s": np.asarray(
            [
                np.nan
                if frame.source_timestamp_s is None
                else float(frame.source_timestamp_s)
            ],
            dtype=np.float64,
        ),
        "source_frame_index": np.asarray(
            [-1 if frame.source_frame_index is None else int(frame.source_frame_index)],
            dtype=np.int64,
        ),
        "source_step": np.asarray(
            [-1 if frame.source_step is None else int(frame.source_step)],
            dtype=np.int64,
        ),
    }
    if frame.depth_mm_u16 is not None:
        payload["depth_mm_u16"] = np.ascontiguousarray(
            frame.depth_mm_u16, dtype=np.uint16
        )
    for key in mask_keys:
        payload[f"mask_{str(key)}"] = np.ascontiguousarray(masks[str(key)], dtype=bool)
    # Write-then-rename keeps concurrent readers from ever seeing a partial npz.
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(output)
    return output


def _none_if_negative(value: int) -> int | None:
    """Decode the -1 npz sentinel used for absent integer provenance fields."""
    return None if int(value) < 0 else int(value)


def load_prepared_phystwin_frame(path: str | Path) -> PreparedPhysTwinFrame:
    """Load prepared phystwin frame."""
    payload = np.load(Path(path), allow_pickle=False)
    mask_frame: dict[str, np.ndarray] = {}
    for key in payload["mask_keys"]:
        name = str(key)
        mask_frame[name] = np.ascontiguousarray(
            np.asarray(payload[f"mask_{name}"], dtype=bool)
        )
    # NaN timestamp is the "no provenance" sentinel from the writer.
    timestamp = float(payload["source_timestamp_s"][0])
    return PreparedPhysTwinFrame(
        seq=int(payload["seq"][0]),
        rgb_frame=np.ascontiguousarray(
            np.asarray(payload["rgb_frame"], dtype=np.uint8)
        ),
        processed_mask_frame=normalize_processed_mask_frame(mask_frame),
        pcd_points=np.ascontiguousarray(
            np.asarray(payload["pcd_points"], dtype=np.float32)
        ),
        pcd_colors=np.ascontiguousarray(
            np.asarray(payload["pcd_colors"], dtype=np.uint8)
        ),
        tracks_yx=np.ascontiguousarray(
            np.asarray(payload["tracks_yx"], dtype=np.float32).reshape(-1, 2)
        ),
        visibility=np.ascontiguousarray(
            np.asarray(payload["visibility"], dtype=bool).reshape(-1)
        ),
        query_points_yx=np.ascontiguousarray(
            np.asarray(payload["query_points_yx"], dtype=np.float32).reshape(-1, 2)
        ),
        source_timestamp_s=None if not np.isfinite(timestamp) else timestamp,
        source_frame_index=_none_if_negative(int(payload["source_frame_index"][0])),
        source_step=_none_if_negative(int(payload["source_step"][0])),
        depth_mm_u16=(
            np.ascontiguousarray(np.asarray(payload["depth_mm_u16"], dtype=np.uint16))
            if "depth_mm_u16" in payload.files
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Headless capture finalize: inputs and PhysTwin-layout outputs
# ---------------------------------------------------------------------------


def _load_frame_masks(path: Path) -> dict[str, np.ndarray]:
    """Map a headless capture mask npz (``*_mask`` keys) to a mask frame dict."""
    payload = np.load(path, allow_pickle=False)
    frame = {
        "object": np.asarray(payload["object_mask"], dtype=bool),
        "controller": np.asarray(payload["controller_mask"], dtype=bool),
    }
    if "hand_a_mask" in payload:
        frame["hand_a"] = np.asarray(payload["hand_a_mask"], dtype=bool)
    if "hand_b_mask" in payload:
        frame["hand_b"] = np.asarray(payload["hand_b_mask"], dtype=bool)
    return frame


def _write_tracking_npz(
    output_dir: Path,
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_txy: np.ndarray,
) -> None:
    # The identical npz lands in both tracking/ and cotracker/: PhysTwin
    # loaders read cotracker/0.npz by path convention even though the tracks
    # come from tapnextpp (manifest sets not_actual_cotracker=True).
    """Write tracking NPZ."""
    tracking_dir = output_dir / "tracking"
    compat_dir = output_dir / PHYSTWIN_COMPATIBILITY_PATH_NAME
    tracking_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=True)
    for directory in (tracking_dir, compat_dir):
        np.savez(
            directory / "0.npz",
            tracks=np.ascontiguousarray(tracks_yx, dtype=np.float32),
            visibility=np.ascontiguousarray(visibility, dtype=bool),
            queries_txy=np.ascontiguousarray(query_txy, dtype=np.float32),
        )


def _manifest_shared_fields(
    metadata: Mapping[str, Any], *, frame_count: int, query_count: int
) -> dict[str, Any]:
    """Manifest keys shared by the full and prepared-only finalize paths."""
    return {
        "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
        "tracking_product_backend": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "tracker_backend": "tapnextpp",
        "mask_backend": "edgetam",
        "depth_backend": str(
            metadata.get("depth_backend") or metadata.get("depth_source", "")
        ),
        "depth_source_internal": str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
        "compatibility_path_name": PHYSTWIN_COMPATIBILITY_PATH_NAME,
        "not_actual_cotracker": True,
        "camera_count": 1,
        "frame_count": int(frame_count),
        "query_count": int(query_count),
    }


def _finalize_prepared_only_headless_capture(
    capture: Path,
    out: Path,
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # Prepared-only captures already hold one npz per frame; finalize just
    # validates the rows and writes a manifest. The pickle/mp4 products are
    # materialized later from the prepared frames, hence the None paths below.
    """Finalize prepared only headless capture."""
    prepared_paths = [row.get("prepared_phystwin_frame_path") for row in rows]
    missing = [idx for idx, value in enumerate(prepared_paths) if value is None]
    if missing:
        raise KeyError(
            "prepared-only headless capture rows must contain prepared_phystwin_frame_path; "
            f"missing row indices={missing[:5]}"
        )
    first_frame = load_prepared_phystwin_frame(capture / str(prepared_paths[0]))
    manifest = {
        **_manifest_shared_fields(
            metadata,
            frame_count=len(rows),
            query_count=first_frame.query_points_yx.shape[0],
        ),
        "headless_prepared_only": True,
        "chunk_materialization_source": "prepared_phystwin_frame",
        "prepared_frame_count": int(len(prepared_paths)),
        "prepared_frames_dir": "prepared_phystwin",
        "processed_masks_path": None,
        "track_process_data_path": None,
        "final_data_path": None,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_pcd_frames(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_dir: Path,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Write PCD frames."""
    pcd_dir = output_dir / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    c2w_payload = metadata.get("camera_to_world_c2w")
    if c2w_payload is None:
        raise RuntimeError("strict product requires camera_to_world_c2w metadata")
    c2w = np.asarray(c2w_payload, dtype=np.float32).reshape(4, 4)
    if not np.isfinite(c2w).all():
        raise RuntimeError("camera_to_world_c2w metadata must be finite")
    intrinsics = metadata["intrinsics"]
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for idx, row in enumerate(rows):
        if "depth_color_m_path" in row:
            depth_path = capture_dir / str(row["depth_color_m_path"])
        elif "ffs_depth_path" in row:
            # Older captures recorded the depth npy under the ffs-specific key.
            depth_path = capture_dir / str(row["ffs_depth_path"])
        else:
            raise KeyError(
                "headless capture row must contain depth_color_m_path or legacy ffs_depth_path"
            )
        depth = np.load(depth_path)
        rgb = _load_rgb(capture_dir / str(row["rgb_path"]))
        points, colors = dense_world_pcd_grid(
            depth_m=depth,
            color_rgb_u8=rgb,
            intrinsics=intrinsics,
            c2w=c2w,
        )
        np.savez(pcd_dir / f"{idx}.npz", points=points, colors=colors)
        all_points.append(points)
        all_colors.append(colors)
    # rows is never empty here: finalize_headless_capture raises before this
    # call when the capture holds no frames.
    return np.stack(all_points, axis=0), np.stack(all_colors, axis=0)


# ---------------------------------------------------------------------------
# Headless capture finalize: entry point
# ---------------------------------------------------------------------------


def finalize_headless_capture(
    capture_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Finalize headless capture."""
    capture = Path(capture_dir)
    out = Path(output_dir) if output_dir is not None else capture / "phystwin_like"
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = capture / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames_path = capture / "frames.jsonl"
    rows: list[dict[str, Any]] = []
    if frames_path.is_file():
        with frames_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no headless frames found in {capture / 'frames.jsonl'}")
    # Prepared-only captures (per-frame prepared npz, no raw mask files) skip
    # product materialization; the row-shape check covers captures written
    # before the metadata flag existed.
    if bool(metadata.get("headless_prepared_only")) or (
        "prepared_phystwin_frame_path" in rows[0] and "mask_path" not in rows[0]
    ):
        return _finalize_prepared_only_headless_capture(
            capture, out, metadata=metadata, rows=rows
        )

    mask_frames = [_load_frame_masks(capture / str(row["mask_path"])) for row in rows]
    processed_mask_path = write_processed_masks(out, mask_frames)

    trajectory_payloads = [
        np.load(capture / str(row["query_trajectory_path"]), allow_pickle=False)
        for row in rows
    ]
    # All queries are seeded on frame 0, so queries_txy is (t=0, x, y) — note
    # the (y, x) -> (x, y) swap relative to query_points_yx.
    query_points_yx = np.asarray(
        trajectory_payloads[0]["query_points_yx"], dtype=np.float32
    )
    query_txy = np.zeros((len(query_points_yx), 3), dtype=np.float32)
    query_txy[:, 1] = query_points_yx[:, 1]
    query_txy[:, 2] = query_points_yx[:, 0]
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    for payload in trajectory_payloads:
        if "all_tracks_yx" not in payload.files:
            raise RuntimeError("strict product requires full per-query tracks")
        if "all_observation_visibility" not in payload.files:
            raise RuntimeError(
                "strict product requires processed-mask/depth-gated visibility"
            )
        current_tracks = np.asarray(payload["all_tracks_yx"], dtype=np.float32).reshape(
            -1, 2
        )
        current_vis = np.asarray(
            payload["all_observation_visibility"], dtype=bool
        ).reshape(-1)
        if current_tracks.shape[0] != len(query_points_yx):
            raise RuntimeError(
                "strict PhysTwin product requires full per-query tracks; "
                f"got {current_tracks.shape[0]} tracks for "
                f"{len(query_points_yx)} queries at seq={int(payload['seq'][0])}"
            )
        tracks.append(current_tracks)
        visibility.append(current_vis)
    tracks_yx = np.stack(tracks, axis=0)
    tracker_visibility = np.stack(visibility, axis=0)
    _write_tracking_npz(
        out, tracks_yx=tracks_yx, visibility=tracker_visibility, query_txy=query_txy
    )
    pcd_points, pcd_colors = _write_pcd_frames(
        out, rows, capture_dir=capture, metadata=metadata
    )

    from demo_v6_2 import tracking as tracking_module  # noqa: PLC0415

    normalized_masks = [normalize_processed_mask_frame(frame) for frame in mask_frames]
    track_input = tracking_module.build_window_observations(
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        mask_frames=normalized_masks,
        pcd_points=pcd_points,
        pcd_colors=pcd_colors,
    )
    # One-shot capture finalize equals the chunk-0 semantics of the
    # design_spec.md state machine: origin motion filtering, whole-window
    # controller selection, and first-frame object volume sampling.
    track_process = tracking_module.TrackingRuntime().process_window(track_input)
    # Origin parity: track_process_data.pkl keeps the FULL motion-filtered
    # object set (data_process_track.py:L127-L135); only final_data is
    # volume-sampled.
    object_motions_valid_full, _ = tracking_module.motion_consistency(
        np.asarray(track_input["object_points"], dtype=np.float32),
        np.asarray(track_input["object_visibilities"], dtype=bool),
        once_false_mask=False,
    )
    track_process_path = out / "track_process_data.pkl"
    with track_process_path.open("wb") as handle:
        pickle.dump(
            {
                "object_points": track_input["object_points"],
                "object_colors": track_input["object_colors"],
                "object_visibilities": track_input["object_visibilities"],
                "object_motions_valid": object_motions_valid_full,
                "controller_points": track_process["controller_points"],
            },
            handle,
        )
    final_data_path = out / "final_data.pkl"
    with final_data_path.open("wb") as handle:
        pickle.dump(
            {
                "object_points": track_process["object_points"],
                "object_colors": track_process["object_colors"],
                "object_visibilities": track_process["object_visibilities"],
                "object_motions_valid": track_process["object_motions_valid"],
                "controller_points": track_process["controller_points"],
            },
            handle,
        )

    _render_tracking_2d_video(
        out / "tracking_2d.mp4",
        capture_dir=capture,
        rows=rows,
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        query_is_object=track_input["query_is_object"],
        query_is_controller=track_input["query_is_controller"],
    )
    _render_world_track_video(
        out / "track_process_data.mp4",
        object_points=track_process["object_points"],
        object_valid=track_process["object_motions_valid"],
        controller_points=track_process["controller_points"],
        title="track_process_data object motion valid + controller FPS30",
    )
    _render_world_track_video(
        out / "final_data.mp4",
        object_points=track_process["object_points"],
        object_valid=track_process["object_visibilities"],
        controller_points=track_process["controller_points"],
        title="final_data object 5mm sample + controller FPS30",
    )
    if (
        track_process["object_points"].shape[1]
        or track_process["controller_points"].shape[1]
    ):
        # final_pcd treats every sampled object point as valid for the whole clip.
        _render_world_track_video(
            out / "final_pcd.mp4",
            object_points=track_process["object_points"],
            object_valid=np.ones(
                np.asarray(track_process["object_points"]).shape[:2], dtype=bool
            ),
            controller_points=track_process["controller_points"],
            title="final_pcd 5mm object sample + controller FPS30",
        )
    else:
        _render_empty_video(
            out / "final_pcd.mp4", frame_count=len(rows), label="final_pcd empty"
        )

    manifest = {
        **_manifest_shared_fields(
            metadata, frame_count=len(rows), query_count=len(query_points_yx)
        ),
        "processed_masks_path": str(processed_mask_path.relative_to(out)),
        "track_process_data_path": str(track_process_path.relative_to(out)),
        "final_data_path": str(final_data_path.relative_to(out)),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
