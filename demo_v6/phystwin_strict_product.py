"""Strict PhysTwin-compatible tracking product for Demo v6.

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

from data_process.depth_backends.geometry import transform_points
from demo_v6.utils.pcd_postprocess import detect_radius_outlier_indices
from demo_v6.utils.projection import build_projection_grid_from_matrix


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
QUERY_SEMANTIC_NONE = np.int8(0)
QUERY_SEMANTIC_OBJECT = np.int8(1)
QUERY_SEMANTIC_CONTROLLER = np.int8(2)


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


# ---------------------------------------------------------------------------
# Tracking product backend selection
# ---------------------------------------------------------------------------


def normalize_tracking_product_backend(value: str | None) -> str:
    # Accept hyphen/underscore and shorthand spellings from CLI flags and
    # config files; everything maps onto the two canonical backend names.
    """Normalize tracking product backend."""
    normalized = str(value or DEFAULT_TRACKING_PRODUCT_BACKEND).strip().lower().replace("_", "-")
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
            f"unsupported tracking product backend {value!r}; expected one of {TRACKING_PRODUCT_BACKENDS}"
        )
    return normalized


def tracking_product_backend_is_strict(value: str | None) -> bool:
    """Return the tracking product backend is strict."""
    return normalize_tracking_product_backend(value) == TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT


# ---------------------------------------------------------------------------
# Processed mask frames (object / controller / optional per-hand)
# ---------------------------------------------------------------------------


def normalize_processed_mask_frame(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
    # Demo v6 offline parity: data_process_sam3d/data_process_mask.py:L56-L80
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
        hand_a = np.asarray(frame.get("hand_a", np.zeros_like(obj, dtype=bool)), dtype=bool)
        hand_b = np.asarray(frame.get("hand_b", np.zeros_like(obj, dtype=bool)), dtype=bool)
        ctrl = np.logical_or(hand_a, hand_b)
    if obj.shape != ctrl.shape:
        raise ValueError("object/controller masks must have the same shape")
    out = {
        "object": np.ascontiguousarray(obj, dtype=bool),
        "controller": np.ascontiguousarray(ctrl, dtype=bool),
    }
    # Raw per-hand masks ride along when present so downstream consumers can
    # still tell the hands apart; they stay optional in the processed frame.
    if "hand_a" in frame and frame["hand_a"] is not None:
        out["hand_a"] = np.ascontiguousarray(np.asarray(frame["hand_a"], dtype=bool), dtype=bool)
    if "hand_b" in frame and frame["hand_b"] is not None:
        out["hand_b"] = np.ascontiguousarray(np.asarray(frame["hand_b"], dtype=bool), dtype=bool)
    return out


def write_processed_masks(output_dir: str | Path, frames: Sequence[Mapping[str, Any]]) -> Path:
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
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
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
    depth_min_m: float = 0.0,
    depth_max_m: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    # Demo v6 offline parity: data_process_sam3d/data_process_pcd.py:L84-L149
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
    ray_x, ray_y = build_projection_grid_from_matrix(width=width, height=height, K=K)
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0) & (depth >= np.float32(float(depth_min_m)))
    if np.isfinite(float(depth_max_m)):
        valid &= depth <= np.float32(float(depth_max_m))

    # Invalid pixels stay exactly (0, 0, 0); downstream filters use that zero
    # norm as the "no depth" sentinel.
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
        points_world = transform_points(points_camera, np.asarray(c2w, dtype=np.float32).reshape(4, 4)).astype(np.float32)
        points[rows, cols] = points_world
    # Leading axis is the camera axis (single camera here), matching the
    # origin per-frame pcd layout of (num_cameras, H, W, 3).
    return points[None].astype(np.float32, copy=False), color[None].astype(np.uint8, copy=False)


def apply_depth_validity_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    depth_m: np.ndarray,
) -> dict[str, np.ndarray]:
    # Demo v6 offline parity: data_process_sam3d/data_process_mask.py:L56-L80
    # intersects semantic masks with valid depth support.
    """Apply depth validity to mask frame."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = normalize_processed_mask_frame(frame)
    filtered: dict[str, np.ndarray] = {}
    for key, mask in normalized.items():
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != valid.shape:
            raise ValueError(f"mask {key!r} shape {arr.shape} does not match depth shape {valid.shape}")
        filtered[key] = np.ascontiguousarray(arr & valid, dtype=bool)
    return normalize_processed_mask_frame(filtered)


def apply_radius_outlier_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    points_grid: np.ndarray,
    *,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    # Demo v6 offline parity: data_process_sam3d/data_process_mask.py:L81-L92
    # and L107-L136 remove isolated 3D mask points before processed mask output.
    """Apply radius outlier to mask frame."""
    normalized = normalize_processed_mask_frame(frame)
    if not bool(enabled):
        return normalized
    grid = np.asarray(points_grid, dtype=np.float32)
    if grid.ndim == 4:
        grid = grid[0]
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(f"points_grid must have shape H,W,3 or 1,H,W,3; got {grid.shape}")

    # Only the object/controller classes are filtered; optional per-hand masks
    # pass through unchanged.
    filtered = {key: np.asarray(value, dtype=bool).copy() for key, value in normalized.items()}
    for key in ("object", "controller"):
        mask = filtered[key]
        if mask.shape != grid.shape[:2]:
            raise ValueError(f"mask {key!r} shape {mask.shape} does not match points grid {grid.shape[:2]}")
        yy, xx = np.nonzero(mask)
        if len(yy) == 0:
            continue
        class_points = grid[yy, xx]
        # Zero-norm points are the "no depth" sentinel from dense_world_pcd_grid.
        finite = np.isfinite(class_points).all(axis=1) & (np.linalg.norm(class_points, axis=1) > 1e-9)
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
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        outlier_indices = np.asarray(result["outlier_indices"], dtype=np.int64)
        if len(outlier_indices):
            filtered[key][yy[outlier_indices], xx[outlier_indices]] = False
    return normalize_processed_mask_frame(filtered)


# ---------------------------------------------------------------------------
# Prepared per-frame product (build + npz round-trip)
# ---------------------------------------------------------------------------


def prepare_phystwin_frame(
    *,
    seq: int,
    rgb_frame: np.ndarray,
    depth_m: np.ndarray,
    mask_frame: Mapping[str, np.ndarray],
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_points_yx: np.ndarray,
    intrinsics: Any,
    c2w: np.ndarray,
    mask_radius_outlier_filter: bool = True,
    mask_radius_outlier_radius_m: float = 0.01,
    mask_radius_outlier_nb_points: int = 40,
    source_timestamp_s: float | None = None,
    source_frame_index: int | None = None,
    source_step: int | None = None,
) -> PreparedPhysTwinFrame:
    """Prepare phystwin frame."""
    rgb = np.ascontiguousarray(np.asarray(rgb_frame, dtype=np.uint8))
    depth = np.asarray(depth_m, dtype=np.float32)
    points, colors = dense_world_pcd_grid(
        depth_m=depth,
        color_rgb_u8=rgb,
        intrinsics=intrinsics,
        c2w=c2w,
    )
    # Mask post-processing runs in origin order: depth-validity intersection
    # first, then the 3D radius-outlier filter on the lifted grid.
    depth_valid_masks = apply_depth_validity_to_mask_frame(mask_frame, depth)
    processed = apply_radius_outlier_to_mask_frame(
        depth_valid_masks,
        points,
        enabled=bool(mask_radius_outlier_filter),
        radius_m=float(mask_radius_outlier_radius_m),
        nb_points=int(mask_radius_outlier_nb_points),
    )
    tracks = np.ascontiguousarray(np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2))
    vis = np.ascontiguousarray(np.asarray(visibility, dtype=bool).reshape(-1))
    queries = np.ascontiguousarray(np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2))
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
        source_timestamp_s=None if source_timestamp_s is None else float(source_timestamp_s),
        source_frame_index=None if source_frame_index is None else int(source_frame_index),
        source_step=None if source_step is None else int(source_step),
    )


def write_prepared_phystwin_frame(path: str | Path, frame: PreparedPhysTwinFrame) -> Path:
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
        "query_points_yx": np.ascontiguousarray(frame.query_points_yx, dtype=np.float32),
        "mask_keys": mask_keys,
        "source_timestamp_s": np.asarray(
            [np.nan if frame.source_timestamp_s is None else float(frame.source_timestamp_s)],
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
        mask_frame[name] = np.ascontiguousarray(np.asarray(payload[f"mask_{name}"], dtype=bool))
    # NaN timestamp is the "no provenance" sentinel from the writer.
    timestamp = float(payload["source_timestamp_s"][0])
    return PreparedPhysTwinFrame(
        seq=int(payload["seq"][0]),
        rgb_frame=np.ascontiguousarray(np.asarray(payload["rgb_frame"], dtype=np.uint8)),
        processed_mask_frame=normalize_processed_mask_frame(mask_frame),
        pcd_points=np.ascontiguousarray(np.asarray(payload["pcd_points"], dtype=np.float32)),
        pcd_colors=np.ascontiguousarray(np.asarray(payload["pcd_colors"], dtype=np.uint8)),
        tracks_yx=np.ascontiguousarray(np.asarray(payload["tracks_yx"], dtype=np.float32).reshape(-1, 2)),
        visibility=np.ascontiguousarray(np.asarray(payload["visibility"], dtype=bool).reshape(-1)),
        query_points_yx=np.ascontiguousarray(np.asarray(payload["query_points_yx"], dtype=np.float32).reshape(-1, 2)),
        source_timestamp_s=None if not np.isfinite(timestamp) else timestamp,
        source_frame_index=_none_if_negative(int(payload["source_frame_index"][0])),
        source_step=_none_if_negative(int(payload["source_step"][0])),
    )


# ---------------------------------------------------------------------------
# Headless capture finalize: inputs and PhysTwin-layout outputs
# ---------------------------------------------------------------------------


def _load_rgb(path: Path) -> np.ndarray:
    """Load RGB."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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
    # Keep the shared keys in sync with the full-capture manifest at the end
    # of finalize_headless_capture.
    manifest = {
        "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
        "tracking_product_backend": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "tracker_backend": "tapnextpp",
        "mask_backend": "edgetam",
        "depth_backend": str(metadata.get("depth_backend") or metadata.get("depth_source", "")),
        "depth_source_internal": str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
        "compatibility_path_name": PHYSTWIN_COMPATIBILITY_PATH_NAME,
        "not_actual_cotracker": True,
        "camera_count": 1,
        "frame_count": int(len(rows)),
        "query_count": int(first_frame.query_points_yx.shape[0]),
        "headless_prepared_only": True,
        "chunk_materialization_source": "prepared_phystwin_frame",
        "prepared_frame_count": int(len(prepared_paths)),
        "prepared_frames_dir": "prepared_phystwin",
        "processed_masks_path": None,
        "track_process_data_path": None,
        "final_data_path": None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    # Captures without table calibration carry no camera_to_world_c2w; identity
    # keeps the product in camera space.
    c2w = np.asarray(metadata.get("camera_to_world_c2w") or np.eye(4, dtype=np.float32), dtype=np.float32).reshape(4, 4)
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
            raise KeyError("headless capture row must contain depth_color_m_path or legacy ffs_depth_path")
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
    if not all_points:
        return np.empty((0, 1, 0, 0, 3), dtype=np.float32), np.empty((0, 1, 0, 0, 3), dtype=np.uint8)
    return np.stack(all_points, axis=0), np.stack(all_colors, axis=0)


# ---------------------------------------------------------------------------
# Headless capture finalize: diagnostic video rendering
# ---------------------------------------------------------------------------


def _open_video_writer(path: Path, *, size: tuple[int, int], fps: float = 30.0):
    """Open video writer."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(size[0]), int(size[1])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer


def _render_tracking_2d_video(
    path: Path,
    *,
    capture_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    size: tuple[int, int] = (848, 480),
) -> None:
    """Render tracking 2d video."""
    import cv2

    writer = _open_video_writer(path, size=size)
    width, height = int(size[0]), int(size[1])
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    for frame_idx, row in enumerate(rows):
        rgb = _load_rgb(capture_dir / str(row["rgb_path"]))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        src_h, src_w = frame.shape[:2]
        if (src_w, src_h) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        # Track coordinates are in source pixels; scale them into the output size.
        sx = float(width) / max(1.0, float(src_w))
        sy = float(height) / max(1.0, float(src_h))
        tracks = np.asarray(tracks_yx[frame_idx], dtype=np.float32)
        vis = np.asarray(visibility[frame_idx], dtype=bool)
        finite = np.isfinite(tracks).all(axis=1)
        visible = np.flatnonzero(vis & finite)
        for idx in visible:
            y = int(round(float(tracks[idx, 0]) * sy))
            x = int(round(float(tracks[idx, 1]) * sx))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            # BGR color code: green = object query, red = controller query,
            # light gray = neither semantic class.
            color = (60, 220, 60) if idx < len(is_object) and is_object[idx] else (40, 80, 255)
            if idx < len(is_controller) and not is_object[idx] and not is_controller[idx]:
                color = (220, 220, 220)
            cv2.circle(frame, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            f"tracking_2d frame={frame_idx} visible={len(visible)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _world_xy_bounds(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Padded world-XY bounds over all finite, non-zero points.

    Zero-norm points (the "no depth" sentinel) are excluded so they do not
    drag the view toward the origin; an 8% margin keeps dots off the frame
    edge, and a unit box is the fallback when nothing is valid.
    """
    chunks: list[np.ndarray] = []
    for arr in arrays:
        pts = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.0)
        if np.any(finite):
            chunks.append(pts[finite, :2])
    if not chunks:
        return np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)
    xy = np.concatenate(chunks, axis=0)
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    span = np.maximum(hi - lo, np.float32(1e-3))
    pad = span * np.float32(0.08)
    return lo - pad, hi + pad


def _draw_world_points(
    frame: np.ndarray,
    points: np.ndarray,
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    color_bgr: tuple[int, int, int],
    radius: int,
) -> int:
    """Scatter world points onto a top-down XY view; returns the drawn count."""
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.0)
    pts = pts[finite]
    if len(pts) == 0:
        return 0
    lo, hi = bounds
    height, width = frame.shape[:2]
    span = np.maximum(hi - lo, np.float32(1e-6))
    # Fixed pixel margins leave room for the HUD text; world +Y points up, so
    # flip the row axis after mapping.
    px = np.clip(((pts[:, 0] - lo[0]) / span[0] * (width - 60) + 30).astype(np.int64), 0, width - 1)
    py = np.clip(((pts[:, 1] - lo[1]) / span[1] * (height - 80) + 50).astype(np.int64), 0, height - 1)
    py = height - 1 - py
    for x, y in zip(px, py):
        cv2.circle(frame, (int(x), int(y)), int(radius), color_bgr, -1, lineType=cv2.LINE_AA)
    return int(len(pts))


def _render_world_track_video(
    path: Path,
    *,
    object_points: np.ndarray,
    object_valid: np.ndarray,
    controller_points: np.ndarray,
    title: str,
    size: tuple[int, int] = (640, 480),
) -> None:
    """Render world track video."""
    import cv2

    writer = _open_video_writer(path, size=size)
    frame_count = max(int(np.asarray(object_points).shape[0]), int(np.asarray(controller_points).shape[0]), 1)
    # Bounds are computed once over the whole clip so the view does not jitter
    # frame to frame.
    bounds = _world_xy_bounds(object_points, controller_points)
    width, height = int(size[0]), int(size[1])
    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Indices clamp to the last frame so a shorter object/controller/valid
        # array simply holds its final state.
        obj = np.asarray(object_points[min(frame_idx, max(0, object_points.shape[0] - 1))], dtype=np.float32).reshape(-1, 3)
        valid = np.asarray(object_valid[min(frame_idx, max(0, object_valid.shape[0] - 1))], dtype=bool).reshape(-1)
        if len(valid) == len(obj):
            obj = obj[valid]
        ctrl = np.asarray(controller_points[min(frame_idx, max(0, controller_points.shape[0] - 1))], dtype=np.float32).reshape(-1, 3)
        obj_count = _draw_world_points(frame, obj, bounds=bounds, color_bgr=(50, 220, 80), radius=2)
        ctrl_count = _draw_world_points(frame, ctrl, bounds=bounds, color_bgr=(40, 40, 255), radius=5)
        cv2.putText(frame, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"frame={frame_idx} object={obj_count} controller={ctrl_count}",
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 230, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _render_empty_video(path: Path, *, frame_count: int, label: str, size: tuple[int, int] = (640, 360)) -> None:
    """Render empty video."""
    import cv2

    writer = _open_video_writer(path, size=size)
    width, height = int(size[0]), int(size[1])
    count = max(1, int(frame_count))
    for frame_idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame={frame_idx}", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


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
        return _finalize_prepared_only_headless_capture(capture, out, metadata=metadata, rows=rows)

    mask_frames = [_load_frame_masks(capture / str(row["mask_path"])) for row in rows]
    processed_mask_path = write_processed_masks(out, mask_frames)

    trajectory_payloads = [np.load(capture / str(row["query_trajectory_path"]), allow_pickle=False) for row in rows]
    # All queries are seeded on frame 0, so queries_txy is (t=0, x, y) — note
    # the (y, x) -> (x, y) swap relative to query_points_yx.
    query_points_yx = np.asarray(trajectory_payloads[0]["query_points_yx"], dtype=np.float32)
    query_txy = np.zeros((len(query_points_yx), 3), dtype=np.float32)
    query_txy[:, 1] = query_points_yx[:, 1]
    query_txy[:, 2] = query_points_yx[:, 0]
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    for payload in trajectory_payloads:
        # Prefer the all_* keys: they carry the full per-query arrays, whereas
        # tracks_yx/visibility may hold only the currently-alive subset. The
        # length check below enforces the full-array requirement either way.
        track_key = "all_tracks_yx" if "all_tracks_yx" in payload.files else "tracks_yx"
        vis_key = "all_tracker_visibility" if "all_tracker_visibility" in payload.files else "visibility"
        current_tracks = np.asarray(payload[track_key], dtype=np.float32).reshape(-1, 2)
        current_vis = np.asarray(payload[vis_key], dtype=bool).reshape(-1)
        if current_tracks.shape[0] != len(query_points_yx):
            raise RuntimeError(
                "strict PhysTwin product requires full per-query tracks; "
                f"got {current_tracks.shape[0]} tracks for {len(query_points_yx)} queries at seq={int(payload['seq'][0])}"
            )
        tracks.append(current_tracks)
        visibility.append(current_vis)
    tracks_yx = np.stack(tracks, axis=0)
    tracker_visibility = np.stack(visibility, axis=0)
    _write_tracking_npz(out, tracks_yx=tracks_yx, visibility=tracker_visibility, query_txy=query_txy)
    pcd_points, pcd_colors = _write_pcd_frames(out, rows, capture_dir=capture, metadata=metadata)

    from demo_v6 import tracking as tracking_module  # noqa: PLC0415

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
    if track_process["object_points"].shape[1] or track_process["controller_points"].shape[1]:
        # final_pcd treats every sampled object point as valid for the whole clip.
        _render_world_track_video(
            out / "final_pcd.mp4",
            object_points=track_process["object_points"],
            object_valid=np.ones(np.asarray(track_process["object_points"]).shape[:2], dtype=bool),
            controller_points=track_process["controller_points"],
            title="final_pcd 5mm object sample + controller FPS30",
        )
    else:
        _render_empty_video(out / "final_pcd.mp4", frame_count=len(rows), label="final_pcd empty")

    # Keep the shared keys in sync with the prepared-only manifest in
    # _finalize_prepared_only_headless_capture.
    manifest = {
        "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
        "tracking_product_backend": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "tracker_backend": "tapnextpp",
        "mask_backend": "edgetam",
        "depth_backend": str(metadata.get("depth_backend") or metadata.get("depth_source", "")),
        "depth_source_internal": str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
        "compatibility_path_name": PHYSTWIN_COMPATIBILITY_PATH_NAME,
        "not_actual_cotracker": True,
        "camera_count": 1,
        "frame_count": int(len(rows)),
        "query_count": int(len(query_points_yx)),
        "processed_masks_path": str(processed_mask_path.relative_to(out)),
        "track_process_data_path": str(track_process_path.relative_to(out)),
        "final_data_path": str(final_data_path.relative_to(out)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
