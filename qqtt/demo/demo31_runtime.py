from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from data_process.depth_backends.ffs_defaults import (
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_VALID_ITERS,
    DEFAULT_FFS_MAX_DISP,
)
from data_process.depth_backends.geometry import transform_points
from qqtt.demo import demo3_runtime
from qqtt.demo.demo31_cotracker_process import (
    CoTrackerProcessConfig,
    PROCESS_MODE_SUBPROCESS,
    PROCESS_MODES,
    start_cotracker_process,
)
from qqtt.demo.demo31_dual_gpu_ipc import (
    LatestMaskCache,
    TrackingInputLitePacket,
    TrackingResultLitePacket,
    should_publish_tracking_input,
)
from qqtt.demo.demo31_profile import build_empty_dual_gpu_profile_summary, event_fps, percentile_summary
from qqtt.demo.trackable_mask_filter import (
    TRACKABLE_MASK_BUILD_POLICIES,
    TRACKABLE_MASK_BUILD_POLICY_DISABLED,
    TRACKABLE_MASK_BUILD_POLICY_INIT_ONLY,
    TRACKABLE_QUERY_INIT_STRATEGIES,
    TRACKABLE_QUERY_INIT_STRATEGY_STANDARD_FILTER_INIT,
    TrackableMaskFilterConfig,
    build_standard_filter_trackable_masks_for_camera,
    summarize_trackable_stats,
)
from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world
from qqtt.tracking.backends.point_tracker_adapter import (
    LITETRACKER_RUNTIME_ONNX_CUDA,
    LITETRACKER_RUNTIME_PYTORCH,
    LITETRACKER_RUNTIMES,
    TRACKER_BACKEND_COTRACKER3,
    TRACKER_BACKEND_LITETRACKER,
    TRACKER_BACKEND_LOCOTRACK,
    TRACKER_BACKENDS,
    TRACKER_BATCH_QUERY_COUNT_POLICIES,
    TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED,
    TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON,
    TRACKER_EXECUTION_MODE_AUTO,
    TRACKER_EXECUTION_MODE_BATCH_VIEWS,
    TRACKER_EXECUTION_MODE_SERIAL,
    TRACKER_EXECUTION_MODES,
    effective_legacy_update_mode,
    normalize_litetracker_runtime,
    normalize_tracker_backend,
    normalize_tracker_batch_query_count_policy,
    normalize_tracker_execution_mode,
    tracker_backend_spec,
)


PRESET_DEMO31_DUAL4090_HIGHFPS = "demo3.1-dual4090-highfps"
PRESET_DEMO32_FFS_LITETRACKER = "demo3.2-ffs-litetracker"
PRESETS = (PRESET_DEMO31_DUAL4090_HIGHFPS, PRESET_DEMO32_FFS_LITETRACKER)

FUSION_MASK_POLICY_STRICT = "strict"
FUSION_MASK_POLICY_LATEST_REUSE = "latest-reuse"
FUSION_MASK_POLICIES = (FUSION_MASK_POLICY_STRICT, FUSION_MASK_POLICY_LATEST_REUSE)

GPU_PLAN_SPLIT_MASK0_TRACK1 = "split-mask0-track1"
GPU_PLANS = (GPU_PLAN_SPLIT_MASK0_TRACK1,)

DEFAULT_OUTPUT_ROOT = Path("result/demo31_dual4090_realsense_cotracker")
DEFAULT_DEMO32_OUTPUT_ROOT = Path("result/demo32_ffs_litetracker")
DEFAULT_RENDER_TARGET_FPS = 60.0
DEFAULT_COTRACKER_INPUT_FPS = 10.0
DEFAULT_COTRACKER_INPUT_MAX_AGE_MS = 250.0
DEFAULT_COTRACKER_RESULT_STALE_TIMEOUT_MS = 1500.0
DEFAULT_MASK_STALE_TIMEOUT_MS = 250.0
DEFAULT_MASK_GPU = "0"
DEFAULT_COTRACKER_GPU = "1"
DEFAULT_DEMO31_COTRACKER_QUERY_COUNT_REQUEST = "4096"
DEFAULT_LOCOTRACK_MODEL_SIZE = "small"
DEFAULT_LOCOTRACK_WINDOW_FRAMES = 8
DEFAULT_LOCOTRACK_RESOLUTION = (256, 256)
DEFAULT_LOCOTRACK_QUERY_CHUNK_SIZE = 256
DEFAULT_LOCOTRACK_AUTOCAST_DTYPE = "bf16"
DEFAULT_SAM31_INIT_QUICK_FAIL_EMPTY_MASKS = True
DEFAULT_SAM31_INIT_MIN_MASK_PIXELS = 1
DEFAULT_DEMO32_LITETRACKER_REPO_DIR = "/home/xinjie/external/lite-tracker"
DEFAULT_DEMO32_LITETRACKER_WEIGHTS = "/home/xinjie/external/weights/cotracker3/scaled_online.pth"
DEFAULT_DEMO32_TRACKABLE_MASK_BUILD_POLICY = TRACKABLE_MASK_BUILD_POLICY_INIT_ONLY
DEFAULT_DEMO32_TRACKABLE_QUERY_INIT_STRATEGY = TRACKABLE_QUERY_INIT_STRATEGY_STANDARD_FILTER_INIT
DEFAULT_DEMO32_CONTROLLER_TRACKABLE_MAX_POINTS_PER_CAMERA = 4999
DEFAULT_CONTROLLER_MASK_ERODE_PX = 0
DEFAULT_DEMO_MODE_CONTROLLER_MASK_ERODE_PX = 1
DEFAULT_CONTROLLER_RENDER_VOXEL_M = 0.003
DEFAULT_CONTROLLER_RENDER_MAX_POINTS = 10_000
DEFAULT_LIFT_INPUT_CACHE_GROUPS = 128
DEFAULT_PENDING_RENDER_PACKET_GROUPS = 128
TRACKING_RENDER_PACKET_MATCH_POLICY = "exact-then-nearest-pending-pcd-by-group-id"
DEFAULT_WAIT_FOR_TRACKING_OVERLAY = True
DEFAULT_DEMO31_OVERLAY_MAX_POINTS_PER_CAMERA = 0
DEFAULT_OVERLAY_REJECT_OUTSIDE_SEMANTIC_BBOX = True
DEFAULT_OVERLAY_MAX_DISTANCE_FROM_CONTROLLER_M = 0.15
DEFAULT_OVERLAY_CONTROL_POINT_MARKERS = True
DEFAULT_OVERLAY_CONTROL_POINT_COUNT = 30
DEFAULT_OVERLAY_CONTROL_POINT_RADIUS_M = 0.01
DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB = (255, 0, 0)
DEFAULT_OVERLAY_RENDER_RAW_TRACK_POINTS = False
TRACKER_VISUALIZATION_MODE_NONE = "none"
TRACKER_VISUALIZATION_MODE_SURFACE_MARKERS = "3d-surface-markers"
TRACKER_VISUALIZATION_MODE_2D_DEBUG = "2d-debug"
TRACKER_VISUALIZATION_MODE_LEGACY_3D_LIFT = "legacy-3d-lift"
TRACKER_VISUALIZATION_MODE_ALL_TRACKS_3D_LIFT = "all-tracks-3d-lift"
TRACKER_VISUALIZATION_MODES = (
    TRACKER_VISUALIZATION_MODE_NONE,
    TRACKER_VISUALIZATION_MODE_SURFACE_MARKERS,
    TRACKER_VISUALIZATION_MODE_2D_DEBUG,
    TRACKER_VISUALIZATION_MODE_LEGACY_3D_LIFT,
    TRACKER_VISUALIZATION_MODE_ALL_TRACKS_3D_LIFT,
)
DEFAULT_TRACKER_VISUALIZATION_MODE = TRACKER_VISUALIZATION_MODE_SURFACE_MARKERS
DEFAULT_DEMO32_TRACKER_VISUALIZATION_MODE = TRACKER_VISUALIZATION_MODE_ALL_TRACKS_3D_LIFT
DEFAULT_TRACKER_3D_SNAP_RADIUS_PX = 4.0
DEFAULT_TRACKER_3D_MARKER_RADIUS_M = 0.006
DEFAULT_TRACKER_CONTROL_POINTS_PER_CAMERA = 16
TRACKER_CONTROL_POINT_SELECTION_VISIBLE_SPREAD = "visible-spread"
TRACKER_CONTROL_POINT_SELECTION_TOP_VISIBLE = "top-visible"
TRACKER_CONTROL_POINT_SELECTION_MASK_STRATIFIED = "mask-stratified"
TRACKER_CONTROL_POINT_SELECTIONS = (
    TRACKER_CONTROL_POINT_SELECTION_VISIBLE_SPREAD,
    TRACKER_CONTROL_POINT_SELECTION_TOP_VISIBLE,
    TRACKER_CONTROL_POINT_SELECTION_MASK_STRATIFIED,
)
DEFAULT_TRACKER_CONTROL_POINT_SELECTION = TRACKER_CONTROL_POINT_SELECTION_VISIBLE_SPREAD
SURFACE_ANCHOR_LABEL_OBJECT = "object"
SURFACE_ANCHOR_LABEL_CONTROLLER = "controller"
SURFACE_ANCHOR_LABEL_UNION = "union"
SURFACE_ANCHOR_LABELS = (
    SURFACE_ANCHOR_LABEL_OBJECT,
    SURFACE_ANCHOR_LABEL_CONTROLLER,
    SURFACE_ANCHOR_LABEL_UNION,
)
TRACKER_MARKER_LABEL_COLORS_RGB = {
    SURFACE_ANCHOR_LABEL_OBJECT: DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB,
    SURFACE_ANCHOR_LABEL_CONTROLLER: DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB,
    SURFACE_ANCHOR_LABEL_UNION: (255, 0, 0),
}
OVERLAY_DEBUG_CAMERA_COLORS_RGB = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}
PCD_COLOR_MODE_RGB = "rgb"
PCD_COLOR_MODE_CLASS = "class"
PCD_COLOR_MODES = (PCD_COLOR_MODE_RGB, PCD_COLOR_MODE_CLASS)
TRACKING_BACKEND_EXECUTION_MODES = TRACKER_EXECUTION_MODES
TRACKING_BACKEND_EXECUTION_MODE_AUTO = TRACKER_EXECUTION_MODE_AUTO
TRACKING_BACKEND_EXECUTION_MODE_SERIAL = TRACKER_EXECUTION_MODE_SERIAL
TRACKING_BACKEND_EXECUTION_MODE_BATCH_VIEWS = TRACKER_EXECUTION_MODE_BATCH_VIEWS
DEFAULT_TRACKING_BACKEND_EXECUTION_MODE = TRACKING_BACKEND_EXECUTION_MODE_BATCH_VIEWS

ConnectedSerialsProvider = Callable[[], Sequence[str]]
CudaDeviceCountProvider = Callable[[], int]
ProcessClientFactory = Callable[[CoTrackerProcessConfig], Any]


def is_demo32_preset(args: argparse.Namespace) -> bool:
    return str(getattr(args, "preset", "")) == PRESET_DEMO32_FFS_LITETRACKER


def demo_label_for_args(args: argparse.Namespace) -> str:
    return "Demo 3.2" if is_demo32_preset(args) else "Demo 3.1"


def demo_name_for_args(args: argparse.Namespace) -> str:
    return "demo3.2" if is_demo32_preset(args) else "demo3.1"


def default_controller_mask_erode_px_for_mode(mode: str) -> int:
    return (
        DEFAULT_DEMO_MODE_CONTROLLER_MASK_ERODE_PX
        if str(mode) == demo3_runtime.MODE_DEMO
        else DEFAULT_CONTROLLER_MASK_ERODE_PX
    )


def resolved_controller_mask_erode_px(args: argparse.Namespace) -> int:
    value = getattr(args, "controller_mask_erode_px", None)
    if value is None:
        return default_controller_mask_erode_px_for_mode(str(getattr(args, "mode", demo3_runtime.DEFAULT_MODE)))
    return int(value)


def _merge_cotracker_process_snapshot_metrics(
    summary: dict[str, Any],
    snapshot: dict[str, Any] | None,
) -> None:
    if not isinstance(snapshot, dict):
        return
    worker = snapshot.get("worker") if isinstance(snapshot.get("worker"), dict) else {}

    def _float_value(primary_key: str, *, worker_key: str | None = None) -> float:
        value = snapshot.get(primary_key)
        if value is None and worker_key:
            value = worker.get(worker_key)
        return float(value or 0.0)

    def _int_value(primary_key: str, *, worker_key: str | None = None) -> int:
        value = snapshot.get(primary_key)
        if value is None and worker_key:
            value = worker.get(worker_key)
        return int(value or 0)

    summary.update(
        {
            "cotracker_input_fps": _float_value("cotracker_input_fps", worker_key="input_fps"),
            "tracker_input_fps": _float_value("cotracker_input_fps", worker_key="input_fps"),
            "cotracker_publish_fps": _float_value("cotracker_publish_fps", worker_key="publish_fps"),
            "tracker_publish_fps": _float_value("cotracker_publish_fps", worker_key="publish_fps"),
            "cotracker_input_count": _int_value("cotracker_input_count", worker_key="input_count"),
            "tracker_input_count": _int_value("cotracker_input_count", worker_key="input_count"),
            "cotracker_result_count": _int_value("cotracker_result_count", worker_key="published_packets"),
            "tracker_result_count": _int_value("cotracker_result_count", worker_key="published_packets"),
            "cotracker_model_ms_median": _float_value("cotracker_model_ms_median", worker_key="model_ms_median"),
            "tracker_model_ms_median": _float_value("cotracker_model_ms_median", worker_key="model_ms_median"),
            "cotracker_model_ms_p95": _float_value("cotracker_model_ms_p95", worker_key="model_ms_p95"),
            "tracker_model_ms_p95": _float_value("cotracker_model_ms_p95", worker_key="model_ms_p95"),
            "cotracker_e2e_ms_median": _float_value("cotracker_e2e_ms_median", worker_key="e2e_ms_median"),
            "tracker_e2e_ms_median": _float_value("cotracker_e2e_ms_median", worker_key="e2e_ms_median"),
            "cotracker_e2e_ms_p95": _float_value("cotracker_e2e_ms_p95", worker_key="e2e_ms_p95"),
            "tracker_e2e_ms_p95": _float_value("cotracker_e2e_ms_p95", worker_key="e2e_ms_p95"),
        }
    )
    trackable = snapshot.get("trackable_mask_stats")
    if isinstance(trackable, dict):
        summary.update(trackable)
    for key in (
        "first_trackable_mask_group_id",
        "first_trackable_mask_s",
        "first_tracking_input_publish_s",
        "trackable_mask_initialized_cameras",
        "controller_mask_erode_px",
        "controller_mask_pixels_before_erode_by_camera",
        "controller_mask_pixels_after_erode_by_camera",
    ):
        if key in snapshot:
            summary[key] = snapshot.get(key)


def _overlay_debug_color_rgb(camera_idx: int) -> tuple[int, int, int]:
    color = OVERLAY_DEBUG_CAMERA_COLORS_RGB.get(int(camera_idx))
    if color is not None:
        return color
    palette = tuple(OVERLAY_DEBUG_CAMERA_COLORS_RGB.values())
    return palette[int(camera_idx) % len(palette)]


def _overlay_color_array(point_count: int, color_rgb: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    if int(point_count) <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    return np.repeat(np.asarray(color_rgb, dtype=np.uint8).reshape(1, 3), int(point_count), axis=0)


def _point_centroid(points: np.ndarray) -> list[float] | None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return None
    centroid = pts.mean(axis=0)
    return [float(item) for item in centroid]


def _packet_points(packet: object, attr: str) -> np.ndarray:
    value = getattr(packet, attr, None)
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(value, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points.reshape(-1, 3)


def _semantic_bbox_reference_points(*, scope: str, render_packet: object) -> np.ndarray:
    controller_points = _packet_points(render_packet, "controller_points_m")
    object_points = _packet_points(render_packet, "object_points_m")
    if str(scope) == demo3_runtime.OVERLAY_DISPLAY_SCOPE_CONTROLLER:
        return controller_points
    if str(scope) == SURFACE_ANCHOR_LABEL_OBJECT:
        return object_points
    candidates = [points for points in (object_points, controller_points) if len(points) > 0]
    if not candidates:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(candidates, axis=0).astype(np.float32, copy=False)


def _semantic_bbox_keep_mask(
    points: np.ndarray,
    reference_points: np.ndarray,
    *,
    margin_m: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    refs = np.asarray(reference_points, dtype=np.float32).reshape(-1, 3)
    finite_refs = refs[np.isfinite(refs).all(axis=1)]
    if len(finite_refs) == 0:
        return np.ones((len(pts),), dtype=bool)
    margin = max(float(margin_m), 0.0)
    lower = finite_refs.min(axis=0) - margin
    upper = finite_refs.max(axis=0) + margin
    return np.isfinite(pts).all(axis=1) & np.all(pts >= lower[None, :], axis=1) & np.all(pts <= upper[None, :], axis=1)


def _farthest_point_sample_indices(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    target = min(max(int(count), 0), len(pts))
    if target == 0:
        return np.empty((0,), dtype=np.int64)
    finite_mask = np.isfinite(pts).all(axis=1)
    finite_indices = np.flatnonzero(finite_mask)
    if len(finite_indices) == 0:
        return np.empty((0,), dtype=np.int64)
    finite_pts = pts[finite_indices]
    selected_local = [0]
    min_dist2 = np.sum((finite_pts - finite_pts[0]) ** 2, axis=1)
    for _ in range(1, min(target, len(finite_pts))):
        next_local = int(np.argmax(min_dist2))
        selected_local.append(next_local)
        dist2 = np.sum((finite_pts - finite_pts[next_local]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return finite_indices[np.asarray(selected_local, dtype=np.int64)]


def _farthest_point_sample_indices_2d(points_yx: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points_yx, dtype=np.float32).reshape(-1, 2)
    target = min(max(int(count), 0), len(pts))
    if target == 0:
        return np.empty((0,), dtype=np.int64)
    finite_mask = np.isfinite(pts).all(axis=1)
    finite_indices = np.flatnonzero(finite_mask)
    if len(finite_indices) == 0:
        return np.empty((0,), dtype=np.int64)
    finite_pts = pts[finite_indices]
    selected_local = [0]
    min_dist2 = np.sum((finite_pts - finite_pts[0]) ** 2, axis=1)
    for _ in range(1, min(target, len(finite_pts))):
        next_local = int(np.argmax(min_dist2))
        selected_local.append(next_local)
        dist2 = np.sum((finite_pts - finite_pts[next_local]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return finite_indices[np.asarray(selected_local, dtype=np.int64)]


def _sphere_marker_offsets() -> np.ndarray:
    offsets = [np.zeros((3,), dtype=np.float32)]
    for x in np.linspace(-1.0, 1.0, 5, dtype=np.float32):
        for y in np.linspace(-1.0, 1.0, 5, dtype=np.float32):
            for z in np.linspace(-1.0, 1.0, 5, dtype=np.float32):
                vec = np.asarray([x, y, z], dtype=np.float32)
                if float(np.linalg.norm(vec)) > 0.0 and float(np.linalg.norm(vec)) <= 1.0001:
                    offsets.append(vec)
    return np.asarray(offsets, dtype=np.float32)


_SPHERE_MARKER_OFFSETS = _sphere_marker_offsets()


def _control_point_marker_cloud(
    centers: np.ndarray,
    colors: np.ndarray,
    *,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    ctrs = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(ctrs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    color_arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(color_arr) != len(ctrs):
        color_arr = _overlay_color_array(len(ctrs), DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB)
    radius = max(float(radius_m), 0.0)
    offsets = _SPHERE_MARKER_OFFSETS * radius
    marker_points = (ctrs[:, None, :] + offsets[None, :, :]).reshape(-1, 3).astype(np.float32)
    marker_colors = np.repeat(color_arr, len(offsets), axis=0).astype(np.uint8)
    return marker_points, marker_colors


def _overlay_scope_to_surface_label(scope: str) -> str:
    if str(scope) == SURFACE_ANCHOR_LABEL_OBJECT:
        return SURFACE_ANCHOR_LABEL_OBJECT
    if str(scope) == demo3_runtime.OVERLAY_DISPLAY_SCOPE_CONTROLLER:
        return SURFACE_ANCHOR_LABEL_CONTROLLER
    return SURFACE_ANCHOR_LABEL_UNION


def _surface_marker_color(label: str, camera_idx: int, *, color_by_camera: bool) -> tuple[int, int, int]:
    if color_by_camera:
        return _overlay_debug_color_rgb(int(camera_idx))
    return TRACKER_MARKER_LABEL_COLORS_RGB.get(str(label), TRACKER_MARKER_LABEL_COLORS_RGB[SURFACE_ANCHOR_LABEL_UNION])


def _select_visible_control_indices(
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    *,
    max_points: int,
    selection: str,
) -> np.ndarray:
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    if vis.shape[0] != tracks.shape[0]:
        raise ValueError("visibility length must match tracks_yx.")
    visible_indices = np.flatnonzero(vis)
    limit = int(max_points)
    if limit < 0 or len(visible_indices) <= limit:
        return visible_indices.astype(np.int64)
    if limit == 0:
        return np.empty((0,), dtype=np.int64)
    mode = str(selection)
    if mode == TRACKER_CONTROL_POINT_SELECTION_TOP_VISIBLE:
        return visible_indices[:limit].astype(np.int64)
    # mask-stratified currently has the same spread behavior after semantic
    # scope filtering; the name is kept for the public contract.
    local = _farthest_point_sample_indices_2d(tracks[visible_indices], limit)
    return visible_indices[local].astype(np.int64)


@dataclass(frozen=True)
class SurfaceAnchorLayer:
    camera_idx: int
    label: str
    yx: np.ndarray
    points_world: np.ndarray


@dataclass(frozen=True)
class SurfaceAnchorIndexSnapshot:
    group_id: int
    timestamp_s: float
    layers: dict[tuple[int, str], SurfaceAnchorLayer]


@dataclass(frozen=True)
class SurfaceSnapResult:
    points_world: np.ndarray
    source_indices: np.ndarray
    tracks_yx: np.ndarray
    pixel_errors: np.ndarray
    accepted: int
    rejected: int


def snap_tracks_to_surface(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    surface_layer: SurfaceAnchorLayer,
    radius_px: float,
    max_points: int = -1,
    selection: str = DEFAULT_TRACKER_CONTROL_POINT_SELECTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = _snap_tracks_to_surface_result(
        tracks_yx=tracks_yx,
        visibility=visibility,
        surface_layer=surface_layer,
        radius_px=radius_px,
        max_points=max_points,
        selection=selection,
    )
    return result.points_world, {
        "accepted": int(result.accepted),
        "rejected": int(result.rejected),
        "pixel_error_median": float(np.median(result.pixel_errors)) if len(result.pixel_errors) else 0.0,
        "pixel_error_p95": float(np.percentile(result.pixel_errors, 95)) if len(result.pixel_errors) else 0.0,
    }


def _snap_tracks_to_surface_result(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    surface_layer: SurfaceAnchorLayer,
    radius_px: float,
    max_points: int,
    selection: str,
) -> SurfaceSnapResult:
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    selected_indices = _select_visible_control_indices(
        tracks,
        vis,
        max_points=int(max_points),
        selection=str(selection),
    )
    anchors_yx = np.asarray(surface_layer.yx, dtype=np.float32).reshape(-1, 2)
    anchors_xyz = np.asarray(surface_layer.points_world, dtype=np.float32).reshape(-1, 3)
    if len(selected_indices) == 0 or len(anchors_yx) == 0 or len(anchors_xyz) == 0:
        return SurfaceSnapResult(
            points_world=np.empty((0, 3), dtype=np.float32),
            source_indices=np.empty((0,), dtype=np.int64),
            tracks_yx=np.empty((0, 2), dtype=np.float32),
            pixel_errors=np.empty((0,), dtype=np.float32),
            accepted=0,
            rejected=int(len(selected_indices)),
        )

    accepted_points: list[np.ndarray] = []
    accepted_indices: list[int] = []
    accepted_tracks: list[np.ndarray] = []
    errors: list[float] = []
    radius = max(float(radius_px), 0.0)
    radius2 = radius * radius
    for source_idx in selected_indices:
        track_yx = tracks[int(source_idx)]
        d2 = np.sum((anchors_yx - track_yx[None, :]) ** 2, axis=1)
        nearest = int(np.argmin(d2))
        if float(d2[nearest]) <= radius2:
            accepted_points.append(anchors_xyz[nearest])
            accepted_indices.append(int(source_idx))
            accepted_tracks.append(track_yx)
            errors.append(float(np.sqrt(float(d2[nearest]))))

    accepted = len(accepted_points)
    rejected = int(len(selected_indices)) - accepted
    return SurfaceSnapResult(
        points_world=(
            np.asarray(accepted_points, dtype=np.float32).reshape(-1, 3)
            if accepted_points
            else np.empty((0, 3), dtype=np.float32)
        ),
        source_indices=np.asarray(accepted_indices, dtype=np.int64),
        tracks_yx=(
            np.asarray(accepted_tracks, dtype=np.float32).reshape(-1, 2)
            if accepted_tracks
            else np.empty((0, 2), dtype=np.float32)
        ),
        pixel_errors=np.asarray(errors, dtype=np.float32),
        accepted=int(accepted),
        rejected=int(rejected),
    )


def _surface_anchor_layer_from_mask(
    *,
    camera_idx: int,
    label: str,
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> SurfaceAnchorLayer:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    if depth.shape[:2] != mask_bool.shape[:2]:
        raise ValueError("surface anchor mask and depth shapes must match.")
    valid = np.isfinite(depth) & (depth > np.float32(depth_min_m)) & mask_bool
    if float(depth_max_m) > 0.0:
        valid &= depth < np.float32(depth_max_m)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return SurfaceAnchorLayer(
            camera_idx=int(camera_idx),
            label=str(label),
            yx=np.empty((0, 2), dtype=np.float32),
            points_world=np.empty((0, 3), dtype=np.float32),
        )
    K = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    z = depth[rows, cols].astype(np.float32, copy=False)
    x = ((cols.astype(np.float32) - cx) / fx) * z
    y = ((rows.astype(np.float32) - cy) / fy) * z
    points_camera = np.stack([x, y, z], axis=1).astype(np.float32)
    points_world = transform_points(points_camera, np.asarray(c2w, dtype=np.float32).reshape(4, 4)).astype(np.float32)
    return SurfaceAnchorLayer(
        camera_idx=int(camera_idx),
        label=str(label),
        yx=np.stack([rows, cols], axis=1).astype(np.float32),
        points_world=np.ascontiguousarray(points_world, dtype=np.float32),
    )


def _lift_mask_for_overlay_scope(
    *,
    scope: str,
    camera_idx: int,
    lift_inputs: "Demo31LiftInputSnapshot",
) -> np.ndarray | None:
    idx = int(camera_idx)
    if str(scope) == demo3_runtime.OVERLAY_DISPLAY_SCOPE_CONTROLLER:
        return lift_inputs.controller_mask_by_camera.get(idx, lift_inputs.mask_by_camera.get(idx))
    if str(scope) == SURFACE_ANCHOR_LABEL_OBJECT:
        return lift_inputs.object_mask_by_camera.get(idx, lift_inputs.mask_by_camera.get(idx))
    return lift_inputs.mask_by_camera.get(idx)


@dataclass(frozen=True)
class Demo31LiftInputSnapshot:
    group_id: int
    timestamp_s: float
    depth_by_camera: dict[int, np.ndarray]
    intrinsics_by_camera: dict[int, np.ndarray]
    c2w_by_camera: dict[int, np.ndarray]
    mask_by_camera: dict[int, np.ndarray]
    object_mask_by_camera: dict[int, np.ndarray]
    controller_mask_by_camera: dict[int, np.ndarray]


@dataclass(frozen=True)
class Demo31RetargetedMaskGroup:
    group_id: int
    mask_packets: dict[int, Any]
    edgetam_stage_wall_ms: float
    edgetam_stage_sum_model_ms: float
    edgetam_stage_mode: str
    source_group_id: int
    mask_age_ms: float
    mask_reused: bool

    @property
    def seq(self) -> int:
        return int(self.group_id)


class Demo31LiftInputCache:
    """Bounded main-process cache for group-aligned 2D-to-world lift inputs."""

    def __init__(self, *, max_groups: int = DEFAULT_LIFT_INPUT_CACHE_GROUPS) -> None:
        self.max_groups = int(max_groups)
        self._snapshots: dict[int, Demo31LiftInputSnapshot] = {}
        self._lock = threading.Lock()
        self.published = 0
        self.evicted = 0
        self.hit_count = 0
        self.miss_count = 0

    def publish(
        self,
        *,
        group_id: int,
        timestamp_s: float,
        depth_by_camera: dict[int, np.ndarray],
        intrinsics_by_camera: dict[int, np.ndarray],
        c2w_by_camera: dict[int, np.ndarray],
        mask_by_camera: dict[int, np.ndarray],
        object_mask_by_camera: dict[int, np.ndarray] | None = None,
        controller_mask_by_camera: dict[int, np.ndarray] | None = None,
    ) -> None:
        object_masks = object_mask_by_camera or {}
        controller_masks = controller_mask_by_camera or {}
        snapshot = Demo31LiftInputSnapshot(
            group_id=int(group_id),
            timestamp_s=float(timestamp_s),
            depth_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(depth, dtype=np.float32)).copy()
                for camera_idx, depth in depth_by_camera.items()
            },
            intrinsics_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)).copy()
                for camera_idx, intrinsics in intrinsics_by_camera.items()
            },
            c2w_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(c2w, dtype=np.float32).reshape(4, 4)).copy()
                for camera_idx, c2w in c2w_by_camera.items()
            },
            mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool)).copy()
                for camera_idx, mask in mask_by_camera.items()
            },
            object_mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool)).copy()
                for camera_idx, mask in object_masks.items()
            },
            controller_mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool)).copy()
                for camera_idx, mask in controller_masks.items()
            },
        )
        with self._lock:
            self._snapshots[int(group_id)] = snapshot
            self.published += 1
            self._prune_locked()

    def get(self, group_id: int) -> Demo31LiftInputSnapshot | None:
        with self._lock:
            snapshot = self._snapshots.get(int(group_id))
            if snapshot is None:
                self.miss_count += 1
                return None
            self.hit_count += 1
            return snapshot

    def cached_group_ids(self) -> set[int]:
        with self._lock:
            return {int(group_id) for group_id in self._snapshots}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_groups": int(self.max_groups),
                "cached_groups": int(len(self._snapshots)),
                "oldest_group_id": int(min(self._snapshots)) if self._snapshots else None,
                "newest_group_id": int(max(self._snapshots)) if self._snapshots else None,
                "published": int(self.published),
                "evicted": int(self.evicted),
                "hit_count": int(self.hit_count),
                "miss_count": int(self.miss_count),
            }

    def _prune_locked(self) -> None:
        while len(self._snapshots) > max(1, int(self.max_groups)):
            oldest = min(self._snapshots)
            self._snapshots.pop(oldest, None)
            self.evicted += 1


class Demo31SurfaceAnchorCache:
    """Bounded main-process cache for surface-snapped tracking markers."""

    def __init__(self, *, max_groups: int = DEFAULT_PENDING_RENDER_PACKET_GROUPS) -> None:
        self.max_groups = int(max_groups)
        self._snapshots: dict[int, SurfaceAnchorIndexSnapshot] = {}
        self._lock = threading.Lock()
        self.published = 0
        self.evicted = 0
        self.hit_count = 0
        self.miss_count = 0

    def publish(self, snapshot: SurfaceAnchorIndexSnapshot) -> None:
        copied_layers: dict[tuple[int, str], SurfaceAnchorLayer] = {}
        for (camera_idx, label), layer in snapshot.layers.items():
            copied_layers[(int(camera_idx), str(label))] = SurfaceAnchorLayer(
                camera_idx=int(layer.camera_idx),
                label=str(layer.label),
                yx=np.ascontiguousarray(np.asarray(layer.yx, dtype=np.float32).reshape(-1, 2)).copy(),
                points_world=np.ascontiguousarray(
                    np.asarray(layer.points_world, dtype=np.float32).reshape(-1, 3)
                ).copy(),
            )
        copied = SurfaceAnchorIndexSnapshot(
            group_id=int(snapshot.group_id),
            timestamp_s=float(snapshot.timestamp_s),
            layers=copied_layers,
        )
        with self._lock:
            self._snapshots[int(copied.group_id)] = copied
            self.published += 1
            self._prune_locked()

    def get(self, group_id: int) -> SurfaceAnchorIndexSnapshot | None:
        with self._lock:
            snapshot = self._snapshots.get(int(group_id))
            if snapshot is None:
                self.miss_count += 1
                return None
            self.hit_count += 1
            return snapshot

    def cached_group_ids(self) -> set[int]:
        with self._lock:
            return {int(group_id) for group_id in self._snapshots}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_groups": int(self.max_groups),
                "cached_groups": int(len(self._snapshots)),
                "oldest_group_id": int(min(self._snapshots)) if self._snapshots else None,
                "newest_group_id": int(max(self._snapshots)) if self._snapshots else None,
                "published": int(self.published),
                "evicted": int(self.evicted),
                "hit_count": int(self.hit_count),
                "miss_count": int(self.miss_count),
            }

    def _prune_locked(self) -> None:
        while len(self._snapshots) > max(1, int(self.max_groups)):
            oldest = min(self._snapshots)
            self._snapshots.pop(oldest, None)
            self.evicted += 1


def _normalize_mask_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized != demo3_runtime.MASK_SOURCE_HF_EDGETAM:
        raise ValueError("Demo 3.1 mask source must be hf-edgetam.")
    return demo3_runtime.MASK_SOURCE_HF_EDGETAM


def parse_locotrack_resolution(value: str | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, str):
        raw = value.strip().lower().replace("x", ",")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) == 1:
            height = width = int(parts[0])
        elif len(parts) == 2:
            height, width = int(parts[0]), int(parts[1])
        else:
            raise argparse.ArgumentTypeError("--locotrack-resolution must be HxW, H,W, or a single square size.")
    else:
        parts = tuple(int(item) for item in value)
        if len(parts) == 1:
            height = width = parts[0]
        elif len(parts) == 2:
            height, width = parts
        else:
            raise argparse.ArgumentTypeError("--locotrack-resolution must contain one or two integers.")
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("--locotrack-resolution dimensions must be positive.")
    if height % 8 != 0 or width % 8 != 0:
        raise argparse.ArgumentTypeError("--locotrack-resolution dimensions must be multiples of 8.")
    return (int(height), int(width))


def _physical_cuda_device_count_from_nvidia_smi() -> int:
    override = os.environ.get("QQTT_DEMO31_TEST_CUDA_COUNT")
    if override:
        return int(override)
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return 0
    if completed.returncode != 0:
        return 0
    return sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("GPU "))


def _cuda_count(provider: CudaDeviceCountProvider | None = None) -> int:
    if provider is not None:
        return int(provider())
    return _physical_cuda_device_count_from_nvidia_smi()


def build_arg_parser(*, default_preset: str = PRESET_DEMO31_DUAL4090_HIGHFPS) -> argparse.ArgumentParser:
    if default_preset not in PRESETS:
        raise ValueError(f"Unsupported Demo 3.x preset default: {default_preset}")
    parser = argparse.ArgumentParser(
        description=(
            "Demo 3.1/3.2 dual-4090 realtime visualization. Demo 3.1 uses "
            "RealSense depth plus a point-tracker child process; Demo 3.2 "
            "uses FFS TensorRT batch=3 opt=5 depth and LiteTracker batch-views by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS, default=default_preset)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved Demo 3.x runtime contract and exit.")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-json-output", type=Path, default=None)
    default_output_root = DEFAULT_DEMO32_OUTPUT_ROOT if default_preset == PRESET_DEMO32_FFS_LITETRACKER else DEFAULT_OUTPUT_ROOT
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--camera-ids", type=demo3_runtime.parse_camera_ids, default=demo3_runtime.DEFAULT_CAMERA_IDS)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--calibrate-path", type=Path, default=Path("calibrate.pkl"))
    parser.add_argument("--width", type=int, default=demo3_runtime.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=demo3_runtime.DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=demo3_runtime.DEFAULT_FPS)
    parser.add_argument("--depth-source", default=demo3_runtime.DEPTH_SOURCE_REALSENSE)
    parser.add_argument("--mask-source", default=demo3_runtime.MASK_SOURCE_HF_EDGETAM_CLI)
    parser.add_argument(
        "--edgetam-live-session-keep-frames",
        type=int,
        default=demo3_runtime.DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
        help=(
            "Maximum recent HF EdgeTAM live-session frames kept per camera in "
            "the shared live runtime. This bounds long-run GPU memory growth."
        ),
    )
    parser.add_argument("--mode", choices=demo3_runtime.MODES, default=demo3_runtime.DEFAULT_MODE)
    parser.add_argument("--object-prompt", default=demo3_runtime.DEFAULT_OBJECT_PROMPT)
    parser.add_argument(
        "--sam31-init-quick-fail-empty-masks",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAM31_INIT_QUICK_FAIL_EMPTY_MASKS,
        help=(
            "Fail during live first-frame warmup if SAM3.1 does not produce every "
            "required object/controller mask."
        ),
    )
    parser.add_argument(
        "--sam31-init-min-mask-pixels",
        type=int,
        default=DEFAULT_SAM31_INIT_MIN_MASK_PIXELS,
        help="Minimum pixels required for each enabled SAM3.1 first-frame mask.",
    )
    parser.add_argument(
        "--cotracker-backend",
        choices=TRACKER_BACKENDS,
        default=TRACKER_BACKEND_COTRACKER3,
        help="Legacy flag name for the Demo 3.1 point-tracker backend.",
    )
    parser.add_argument(
        "--tracking-backend",
        dest="cotracker_backend",
        choices=TRACKER_BACKENDS,
        default=argparse.SUPPRESS,
        help="Alias for --cotracker-backend.",
    )
    parser.add_argument(
        "--tracking-backend-execution-mode",
        choices=TRACKING_BACKEND_EXECUTION_MODES,
        default=DEFAULT_TRACKING_BACKEND_EXECUTION_MODE,
        help="Run tracker views serially, as a camera-view batch, or auto-select the best supported mode.",
    )
    parser.add_argument(
        "--tracker-batch-query-count-policy",
        choices=TRACKER_BATCH_QUERY_COUNT_POLICIES,
        default=TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED,
        help="Policy used by batch-capable tracker adapters when camera query counts differ.",
    )
    parser.add_argument("--trackon2-checkpoint", default=None)
    parser.add_argument("--trackon2-config", default=None)
    parser.add_argument("--trackon2-repo-dir", default=None)
    parser.add_argument("--litetracker-weights", default=None)
    parser.add_argument("--litetracker-repo-dir", default=None)
    parser.add_argument(
        "--litetracker-runtime",
        choices=LITETRACKER_RUNTIMES,
        default=LITETRACKER_RUNTIME_PYTORCH,
        help="LiteTracker runtime. onnx-cuda is serial-only for Demo 3.2 A/B profiling.",
    )
    parser.add_argument("--litetracker-onnx-dir", default=None)
    parser.add_argument("--litetracker-export-onnx", action="store_true")
    parser.add_argument("--litetracker-onnx-opset", type=int, default=17)
    parser.add_argument(
        "--litetracker-onnx-optimization-level",
        type=int,
        default=5,
        help="Requested ONNX Runtime graph optimization level; 5 maps to ORT_ENABLE_ALL when available.",
    )
    parser.add_argument("--locotrack-repo-dir", default=None)
    parser.add_argument("--locotrack-checkpoint", default=None)
    parser.add_argument(
        "--locotrack-model-size",
        choices=("small", "base"),
        default=DEFAULT_LOCOTRACK_MODEL_SIZE,
    )
    parser.add_argument(
        "--locotrack-window-frames",
        type=int,
        default=DEFAULT_LOCOTRACK_WINDOW_FRAMES,
    )
    parser.add_argument(
        "--locotrack-resolution",
        type=parse_locotrack_resolution,
        default=DEFAULT_LOCOTRACK_RESOLUTION,
        help="LocoTrack inference resolution as HxW, H,W, or a square size.",
    )
    parser.add_argument(
        "--locotrack-query-chunk-size",
        type=int,
        default=DEFAULT_LOCOTRACK_QUERY_CHUNK_SIZE,
    )
    parser.add_argument(
        "--locotrack-autocast-dtype",
        choices=("bf16", "fp16", "fp32"),
        default=DEFAULT_LOCOTRACK_AUTOCAST_DTYPE,
    )
    parser.add_argument(
        "--cotracker-query-mode",
        choices=(demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE,),
        default=demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE,
    )
    parser.add_argument(
        "--cotracker-query-count",
        default=DEFAULT_DEMO31_COTRACKER_QUERY_COUNT_REQUEST,
        help=(
            "Raw CoTracker query points per camera. Demo 3.1 defaults to 4096 "
            "because full batch=3 at 5000/view exceeds RTX 4090 24GB memory."
        ),
    )
    parser.add_argument(
        "--controller-pcd-max-points-per-camera",
        type=int,
        default=demo3_runtime.DEFAULT_CONTROLLER_PCD_MAX_POINTS_PER_CAMERA,
        help=(
            "Maximum controller/towel mask pixels kept per camera before CoTracker query "
            "selection and before fused PCD construction. Must be < 5000."
        ),
    )
    parser.add_argument(
        "--controller-mask-erode-px",
        type=int,
        default=None,
        help=(
            "Erode the controller mask by this many pixels before tracking/query/anchor "
            "use. The implicit default is 1 in --mode demo and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--trackable-mask-build-policy",
        choices=TRACKABLE_MASK_BUILD_POLICIES,
        default=DEFAULT_DEMO32_TRACKABLE_MASK_BUILD_POLICY,
        help="Demo 3.2 LiteTracker query-init mask build policy.",
    )
    parser.add_argument(
        "--trackable-query-init-strategy",
        choices=TRACKABLE_QUERY_INIT_STRATEGIES,
        default=DEFAULT_DEMO32_TRACKABLE_QUERY_INIT_STRATEGY,
        help="Demo 3.2 LiteTracker query-init strategy.",
    )
    parser.add_argument(
        "--controller-trackable-max-points-per-camera",
        type=int,
        default=DEFAULT_DEMO32_CONTROLLER_TRACKABLE_MAX_POINTS_PER_CAMERA,
        help="Maximum controller pixels kept after standard-filter trackable query-init filtering.",
    )
    parser.add_argument("--cotracker-seed", type=int, default=demo3_runtime.DEFAULT_COTRACKER_SEED)
    parser.add_argument("--disable-cotracker", action="store_true")
    parser.add_argument("--render-mode", choices=demo3_runtime.RENDER_MODES, default=demo3_runtime.RENDER_MODE_POINTCLOUD)
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--render-layer-mode", default=None)
    parser.add_argument("--render-copy-mode", default=None)
    parser.add_argument(
        "--pcd-color-mode",
        choices=PCD_COLOR_MODES,
        default=PCD_COLOR_MODE_RGB,
        help=(
            "Color mode forwarded to the shared PCD runtime. Demo 3.1 defaults "
            "to live RGB so the rendered point cloud keeps camera color instead "
            "of inheriting the fast-native class-color preset."
        ),
    )
    parser.add_argument("--no-render-async-latest-only", action="store_true")
    parser.add_argument("--render-micro-profile", action="store_true")
    parser.add_argument(
        "--object-point-control",
        choices=demo3_runtime.OBJECT_POINT_CONTROLS,
        default=demo3_runtime.OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME,
    )
    parser.add_argument(
        "--object-volume-voxel-m",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
    )
    parser.add_argument(
        "--object-volume-origin",
        choices=demo3_runtime.PHYSTWIN_VOLUME_ORIGINS,
        default=demo3_runtime.PHYSTWIN_VOLUME_ORIGIN_WORLD,
    )
    parser.add_argument("--object-volume-adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--object-volume-min-voxel-m",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M,
    )
    parser.add_argument(
        "--object-volume-max-voxel-m",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M,
    )
    parser.add_argument(
        "--object-volume-target-ms",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS,
    )
    parser.add_argument(
        "--object-volume-emergency-max-points",
        type=int,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
    )
    parser.add_argument(
        "--object-volume-points-per-voxel",
        type=int,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
    )
    parser.add_argument(
        "--controller-render-voxel-m",
        type=float,
        default=DEFAULT_CONTROLLER_RENDER_VOXEL_M,
        help=(
            "Render-only controller PCD voxel downsample size in meters. "
            "Use 0 to disable; tracker/control markers are rendered separately."
        ),
    )
    parser.add_argument(
        "--controller-render-max-points",
        type=int,
        default=DEFAULT_CONTROLLER_RENDER_MAX_POINTS,
        help=(
            "Render-only maximum controller body PCD points after controller render voxel downsampling. "
            "Use 0 to disable; tracker queries and tracker/control markers are not affected."
        ),
    )
    parser.add_argument("--debug-color-by-camera", action="store_true")
    parser.add_argument("--debug-save-per-camera-pcd", action="store_true")
    parser.add_argument("--debug-save-mask-overlays", action="store_true")
    parser.add_argument("--debug-identity-c2w", action="store_true")
    parser.add_argument("--debug-invert-c2w", action="store_true")
    parser.add_argument("--debug-only-camera-idx", type=int, choices=demo3_runtime.DEFAULT_CAMERA_IDS, default=None)
    parser.add_argument("--debug-fusion-max-saved-groups", type=int, default=None)
    parser.add_argument("--gpu-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-sampling-interval-s", type=float, default=0.5)
    parser.add_argument("--gpu-sampling-backend", choices=demo3_runtime.GPU_SAMPLING_BACKENDS, default="nvml")
    parser.add_argument("--gpu-sampling-device-index", type=int, default=0)
    parser.add_argument("--gpu-sampling-device-indexes", type=demo3_runtime.parse_gpu_sampling_device_indexes, default=None)
    parser.add_argument(
        "--overlay-max-points-per-camera",
        type=int,
        default=DEFAULT_DEMO31_OVERLAY_MAX_POINTS_PER_CAMERA,
        help="Maximum rendered CoTracker overlay points per camera; 0 renders all selected visible tracks.",
    )
    parser.add_argument(
        "--overlay-display-scope",
        choices=demo3_runtime.OVERLAY_DISPLAY_SCOPES,
        default=demo3_runtime.DEFAULT_OVERLAY_DISPLAY_SCOPE,
    )
    parser.add_argument(
        "--overlay-debug-color-by-camera",
        action="store_true",
        help="Color lifted CoTracker overlay points by source camera for live alignment debugging.",
    )
    parser.add_argument(
        "--overlay-reject-outside-semantic-bbox",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAY_REJECT_OUTSIDE_SEMANTIC_BBOX,
        help="Reject lifted overlay points outside the current semantic point-cloud bbox plus margin.",
    )
    parser.add_argument(
        "--overlay-max-distance-from-controller-m",
        type=float,
        default=DEFAULT_OVERLAY_MAX_DISTANCE_FROM_CONTROLLER_M,
        help="Semantic bbox margin, in meters, for controller-scope overlay outlier rejection.",
    )
    parser.add_argument(
        "--overlay-control-point-markers",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAY_CONTROL_POINT_MARKERS,
        help="Legacy-lift debug flag for rendering sampled 3D control point markers.",
    )
    parser.add_argument(
        "--overlay-control-point-count",
        type=int,
        default=DEFAULT_OVERLAY_CONTROL_POINT_COUNT,
        help="Number of rendered tracking control points after lift/filter; defaults to FuturePhysTwin's 30.",
    )
    parser.add_argument(
        "--overlay-control-point-radius-m",
        type=float,
        default=DEFAULT_OVERLAY_CONTROL_POINT_RADIUS_M,
        help="Radius of the 3D control point marker cloud in meters.",
    )
    parser.add_argument(
        "--overlay-render-raw-track-points",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAY_RENDER_RAW_TRACK_POINTS,
        help="Also render every lifted track point after filtering; disabled by default to show only control points.",
    )
    parser.add_argument(
        "--tracker-visualization-mode",
        choices=TRACKER_VISUALIZATION_MODES,
        default=DEFAULT_TRACKER_VISUALIZATION_MODE,
        help="How tracking controls are visualized in the 3D PCD.",
    )
    parser.add_argument(
        "--tracker-3d-snap-radius-px",
        type=float,
        default=DEFAULT_TRACKER_3D_SNAP_RADIUS_PX,
        help="Maximum 2D pixel distance from a track to a same-camera semantic surface anchor.",
    )
    parser.add_argument(
        "--tracker-3d-marker-radius-m",
        type=float,
        default=DEFAULT_TRACKER_3D_MARKER_RADIUS_M,
        help="Radius of the rendered 3D tracking control marker sphere.",
    )
    parser.add_argument(
        "--tracker-control-points-per-camera",
        type=int,
        default=DEFAULT_TRACKER_CONTROL_POINTS_PER_CAMERA,
        help="Maximum sparse tracking control handles to render per camera.",
    )
    parser.add_argument(
        "--tracker-control-point-selection",
        choices=TRACKER_CONTROL_POINT_SELECTIONS,
        default=DEFAULT_TRACKER_CONTROL_POINT_SELECTION,
        help="Policy for choosing sparse visible tracking controls before surface snapping.",
    )
    parser.add_argument("--overlay-trail-len", type=int, default=demo3_runtime.DEFAULT_OVERLAY_TRAIL_LEN)
    parser.add_argument("--overlay-stale-timeout-ms", type=float, default=demo3_runtime.DEFAULT_OVERLAY_STALE_TIMEOUT_MS)
    parser.add_argument("--mask-gpu", default=DEFAULT_MASK_GPU)
    parser.add_argument("--cotracker-gpu", default=DEFAULT_COTRACKER_GPU)
    parser.add_argument("--require-two-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-single-gpu-debug", action="store_true")
    parser.add_argument("--gpu-plan", choices=GPU_PLANS, default=GPU_PLAN_SPLIT_MASK0_TRACK1)
    parser.add_argument("--cotracker-process-mode", choices=PROCESS_MODES, default=PROCESS_MODE_SUBPROCESS)
    parser.add_argument("--cotracker-prewarm-backends", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cotracker-update-mode",
        choices=demo3_runtime.COTRACKER_UPDATE_MODES,
        default=demo3_runtime.DEFAULT_COTRACKER_UPDATE_MODE,
    )
    parser.add_argument("--cotracker-input-fps", type=float, default=DEFAULT_COTRACKER_INPUT_FPS)
    parser.add_argument("--cotracker-input-max-age-ms", type=float, default=DEFAULT_COTRACKER_INPUT_MAX_AGE_MS)
    parser.add_argument("--cotracker-result-stale-timeout-ms", type=float, default=DEFAULT_COTRACKER_RESULT_STALE_TIMEOUT_MS)
    parser.add_argument(
        "--wait-for-tracking-overlay",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WAIT_FOR_TRACKING_OVERLAY,
        help=(
            "Legacy compatibility flag. Demo 3.1 rendered profiling remains "
            "gated on fresh CoTracker results whenever tracking is enabled."
        ),
    )
    parser.add_argument("--fusion-mask-policy", choices=FUSION_MASK_POLICIES, default=FUSION_MASK_POLICY_LATEST_REUSE)
    parser.add_argument("--mask-stale-timeout-ms", type=float, default=DEFAULT_MASK_STALE_TIMEOUT_MS)
    parser.add_argument("--render-target-fps", type=float, default=DEFAULT_RENDER_TARGET_FPS)
    parser.add_argument("--render-resample-latest", action=argparse.BooleanOptionalAction, default=True)
    return parser


def apply_preset_defaults(args: argparse.Namespace, *, explicit_options: set[str] | None = None) -> argparse.Namespace:
    explicit = explicit_options or set()
    if "--output-root" not in explicit:
        args.output_root = DEFAULT_DEMO32_OUTPUT_ROOT if args.preset == PRESET_DEMO32_FFS_LITETRACKER else DEFAULT_OUTPUT_ROOT
    if args.preset == PRESET_DEMO31_DUAL4090_HIGHFPS:
        if "--fusion-mask-policy" not in explicit:
            args.fusion_mask_policy = FUSION_MASK_POLICY_LATEST_REUSE
        if "--cotracker-input-fps" not in explicit:
            args.cotracker_input_fps = DEFAULT_COTRACKER_INPUT_FPS
        if "--render-target-fps" not in explicit:
            args.render_target_fps = DEFAULT_RENDER_TARGET_FPS
    elif args.preset == PRESET_DEMO32_FFS_LITETRACKER:
        if "--depth-source" not in explicit:
            args.depth_source = demo3_runtime.DEPTH_SOURCE_FFS
        if "--fusion-mask-policy" not in explicit:
            args.fusion_mask_policy = FUSION_MASK_POLICY_LATEST_REUSE
        if "--cotracker-backend" not in explicit and "--tracking-backend" not in explicit:
            args.cotracker_backend = TRACKER_BACKEND_LITETRACKER
        if "--tracking-backend-execution-mode" not in explicit:
            args.tracking_backend_execution_mode = TRACKING_BACKEND_EXECUTION_MODE_BATCH_VIEWS
        if "--cotracker-update-mode" not in explicit:
            args.cotracker_update_mode = "batch"
        if "--tracker-batch-query-count-policy" not in explicit:
            args.tracker_batch_query_count_policy = TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON
        if "--cotracker-prewarm-backends" not in explicit and "--no-cotracker-prewarm-backends" not in explicit:
            args.cotracker_prewarm_backends = False
        if "--cotracker-input-fps" not in explicit:
            args.cotracker_input_fps = DEFAULT_COTRACKER_INPUT_FPS
        if "--render-target-fps" not in explicit:
            args.render_target_fps = DEFAULT_RENDER_TARGET_FPS
        if "--litetracker-repo-dir" not in explicit:
            args.litetracker_repo_dir = DEFAULT_DEMO32_LITETRACKER_REPO_DIR
        if "--litetracker-weights" not in explicit:
            args.litetracker_weights = DEFAULT_DEMO32_LITETRACKER_WEIGHTS
        if "--tracker-visualization-mode" not in explicit:
            args.tracker_visualization_mode = DEFAULT_DEMO32_TRACKER_VISUALIZATION_MODE
        if "--overlay-reject-outside-semantic-bbox" not in explicit and "--no-overlay-reject-outside-semantic-bbox" not in explicit:
            args.overlay_reject_outside_semantic_bbox = False
        if "--overlay-control-point-markers" not in explicit and "--no-overlay-control-point-markers" not in explicit:
            args.overlay_control_point_markers = True
    if "--controller-mask-erode-px" not in explicit or getattr(args, "controller_mask_erode_px", None) is None:
        args.controller_mask_erode_px = default_controller_mask_erode_px_for_mode(str(args.mode))
    return args


def effective_tracking_backend_execution_mode(args: argparse.Namespace) -> str:
    mode = normalize_tracker_execution_mode(
        getattr(args, "tracking_backend_execution_mode", TRACKING_BACKEND_EXECUTION_MODE_AUTO)
    )
    legacy_update_mode = str(
        getattr(args, "cotracker_update_mode", demo3_runtime.DEFAULT_COTRACKER_UPDATE_MODE)
    ).strip().lower().replace("_", "-")
    if mode == TRACKING_BACKEND_EXECUTION_MODE_AUTO and legacy_update_mode in {"batch", "serial"}:
        return TRACKING_BACKEND_EXECUTION_MODE_BATCH_VIEWS if legacy_update_mode == "batch" else TRACKING_BACKEND_EXECUTION_MODE_SERIAL
    return mode


def validate_args(
    args: argparse.Namespace,
    *,
    require_calibration: bool = False,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
) -> None:
    demo_label = demo_label_for_args(args)
    demo32 = is_demo32_preset(args)
    camera_ids = demo3_runtime.parse_camera_ids(args.camera_ids)
    if len(camera_ids) != 3:
        raise ValueError(f"{demo_label} requires exactly three RealSense cameras.")
    if len(set(camera_ids)) != 3:
        raise ValueError(f"{demo_label} requires exactly three distinct RealSense cameras.")
    depth_source = str(args.depth_source).strip().lower()
    if not demo32 and (depth_source == demo3_runtime.DEPTH_SOURCE_FFS or depth_source.startswith("ffs")):
        raise ValueError("Demo 3.1 does not support FFS. Use --depth-source realsense.")
    if demo32 and depth_source != demo3_runtime.DEPTH_SOURCE_FFS:
        raise ValueError("Demo 3.2 requires FFS depth. Use --depth-source ffs.")
    if not demo32 and depth_source != demo3_runtime.DEPTH_SOURCE_REALSENSE:
        raise ValueError("Demo 3.1 depth source must be realsense.")
    _normalize_mask_source(str(args.mask_source))
    tracker_backend = normalize_tracker_backend(args.cotracker_backend)
    litetracker_runtime = normalize_litetracker_runtime(args.litetracker_runtime)
    normalize_tracker_execution_mode(args.tracking_backend_execution_mode)
    effective_execution_mode = effective_tracking_backend_execution_mode(args)
    normalize_tracker_batch_query_count_policy(args.tracker_batch_query_count_policy)
    if demo32 and tracker_backend != TRACKER_BACKEND_LITETRACKER:
        raise ValueError("Demo 3.2 requires --cotracker-backend litetracker.")
    if litetracker_runtime == LITETRACKER_RUNTIME_ONNX_CUDA and tracker_backend != TRACKER_BACKEND_LITETRACKER:
        raise ValueError("--litetracker-runtime onnx-cuda requires --cotracker-backend litetracker.")
    if litetracker_runtime == LITETRACKER_RUNTIME_ONNX_CUDA and effective_execution_mode != TRACKING_BACKEND_EXECUTION_MODE_SERIAL:
        raise ValueError("--litetracker-runtime onnx-cuda is serial-only; pass --tracking-backend-execution-mode serial.")
    if int(args.litetracker_onnx_opset) < 1:
        raise ValueError("--litetracker-onnx-opset must be positive.")
    if int(args.litetracker_onnx_optimization_level) < 0:
        raise ValueError("--litetracker-onnx-optimization-level must be >= 0.")
    if int(args.locotrack_window_frames) < 1:
        raise ValueError("--locotrack-window-frames must be >= 1.")
    if int(args.locotrack_query_chunk_size) < 1:
        raise ValueError("--locotrack-query-chunk-size must be >= 1.")
    parse_locotrack_resolution(args.locotrack_resolution)
    if str(args.cotracker_query_mode) != demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE:
        raise ValueError(f"{demo_label} currently supports only --cotracker-query-mode phystwin_dense.")
    demo3_runtime.normalize_cotracker_query_count_request(args.cotracker_query_count)
    demo3_runtime.normalize_controller_pcd_max_points_per_camera(args.controller_pcd_max_points_per_camera)
    if str(args.trackable_mask_build_policy) not in TRACKABLE_MASK_BUILD_POLICIES:
        raise ValueError(f"--trackable-mask-build-policy must be one of {TRACKABLE_MASK_BUILD_POLICIES}.")
    if str(args.trackable_query_init_strategy) not in TRACKABLE_QUERY_INIT_STRATEGIES:
        raise ValueError(f"--trackable-query-init-strategy must be one of {TRACKABLE_QUERY_INIT_STRATEGIES}.")
    if int(args.controller_trackable_max_points_per_camera) < 0:
        raise ValueError("--controller-trackable-max-points-per-camera must be >= 0.")
    if int(args.sam31_init_min_mask_pixels) < 1:
        raise ValueError("--sam31-init-min-mask-pixels must be >= 1.")
    if resolved_controller_mask_erode_px(args) < 0:
        raise ValueError("--controller-mask-erode-px must be >= 0.")
    if str(args.object_point_control) not in demo3_runtime.OBJECT_POINT_CONTROLS:
        raise ValueError(f"{demo_label} unsupported --object-point-control {args.object_point_control}")
    if str(args.object_volume_origin) not in demo3_runtime.PHYSTWIN_VOLUME_ORIGINS:
        raise ValueError(f"{demo_label} unsupported --object-volume-origin {args.object_volume_origin}")
    if float(args.object_volume_voxel_m) <= 0.0:
        raise ValueError("--object-volume-voxel-m must be positive.")
    if float(args.object_volume_min_voxel_m) <= 0.0 or float(args.object_volume_max_voxel_m) <= 0.0:
        raise ValueError("--object-volume-min-voxel-m and --object-volume-max-voxel-m must be positive.")
    if float(args.object_volume_min_voxel_m) > float(args.object_volume_max_voxel_m):
        raise ValueError("--object-volume-min-voxel-m must be <= --object-volume-max-voxel-m.")
    if float(args.object_volume_target_ms) <= 0.0:
        raise ValueError("--object-volume-target-ms must be > 0.")
    if int(args.object_volume_emergency_max_points) < 0:
        raise ValueError("--object-volume-emergency-max-points must be >= 0.")
    if int(args.object_volume_points_per_voxel) < 1:
        raise ValueError("--object-volume-points-per-voxel must be >= 1.")
    if float(args.controller_render_voxel_m) < 0.0:
        raise ValueError("--controller-render-voxel-m must be >= 0.")
    if int(args.controller_render_max_points) < 0:
        raise ValueError("--controller-render-max-points must be >= 0.")
    if int(args.edgetam_live_session_keep_frames) < 1:
        raise ValueError("--edgetam-live-session-keep-frames must be >= 1.")
    if bool(args.debug_identity_c2w) and bool(args.debug_invert_c2w):
        raise ValueError(f"{demo_label} accepts only one of --debug-identity-c2w or --debug-invert-c2w.")
    if args.debug_only_camera_idx is not None and int(args.debug_only_camera_idx) not in set(camera_ids):
        raise ValueError(f"--debug-only-camera-idx {args.debug_only_camera_idx} is not in --camera-ids {camera_ids}.")
    if int(args.gpu_sampling_device_index) < 0:
        raise ValueError("--gpu-sampling-device-index must be >= 0.")
    if args.gpu_sampling_device_indexes is not None and any(int(index) < 0 for index in args.gpu_sampling_device_indexes):
        raise ValueError("--gpu-sampling-device-indexes must be >= 0.")
    if float(args.gpu_sampling_interval_s) <= 0.0:
        raise ValueError("--gpu-sampling-interval-s must be > 0.")
    if int(args.overlay_max_points_per_camera) < 0:
        raise ValueError("--overlay-max-points-per-camera must be >= 0; use 0 for all selected visible tracks.")
    if str(args.overlay_display_scope) not in demo3_runtime.OVERLAY_DISPLAY_SCOPES:
        raise ValueError(f"--overlay-display-scope must be one of {demo3_runtime.OVERLAY_DISPLAY_SCOPES}.")
    if float(args.overlay_max_distance_from_controller_m) < 0.0:
        raise ValueError("--overlay-max-distance-from-controller-m must be non-negative.")
    if int(args.overlay_control_point_count) < 0:
        raise ValueError("--overlay-control-point-count must be >= 0.")
    if float(args.overlay_control_point_radius_m) < 0.0:
        raise ValueError("--overlay-control-point-radius-m must be non-negative.")
    if str(args.tracker_visualization_mode) not in TRACKER_VISUALIZATION_MODES:
        raise ValueError(f"--tracker-visualization-mode must be one of {TRACKER_VISUALIZATION_MODES}.")
    if float(args.tracker_3d_snap_radius_px) < 0.0:
        raise ValueError("--tracker-3d-snap-radius-px must be non-negative.")
    if float(args.tracker_3d_marker_radius_m) < 0.0:
        raise ValueError("--tracker-3d-marker-radius-m must be non-negative.")
    if int(args.tracker_control_points_per_camera) < 0:
        raise ValueError("--tracker-control-points-per-camera must be >= 0.")
    if str(args.tracker_control_point_selection) not in TRACKER_CONTROL_POINT_SELECTIONS:
        raise ValueError(f"--tracker-control-point-selection must be one of {TRACKER_CONTROL_POINT_SELECTIONS}.")
    if float(args.cotracker_input_fps) < 0.0:
        raise ValueError("--cotracker-input-fps must be non-negative.")
    if str(args.cotracker_update_mode) not in demo3_runtime.COTRACKER_UPDATE_MODES:
        raise ValueError(f"--cotracker-update-mode must be one of {demo3_runtime.COTRACKER_UPDATE_MODES}.")
    if str(args.mask_gpu) == str(args.cotracker_gpu) and not bool(args.allow_single_gpu_debug):
        raise ValueError(f"{demo_label} requires distinct --mask-gpu and --cotracker-gpu unless --allow-single-gpu-debug is passed.")
    if bool(args.require_two_cuda) and not bool(args.allow_single_gpu_debug):
        count = _cuda_count(cuda_device_count_provider)
        if count < 2:
            raise RuntimeError(f"{demo_label} requires at least two CUDA devices before process isolation; found {count}.")
    if require_calibration and not Path(args.calibrate_path).is_file():
        raise FileNotFoundError(f"{demo_label} requires calibrate.pkl for three-camera world fusion: {args.calibrate_path}")


def build_cotracker_process_config(args: argparse.Namespace) -> CoTrackerProcessConfig:
    execution_mode = effective_tracking_backend_execution_mode(args)
    return CoTrackerProcessConfig(
        camera_ids=demo3_runtime.parse_camera_ids(args.camera_ids),
        cotracker_gpu=str(args.cotracker_gpu),
        cotracker_backend=normalize_tracker_backend(args.cotracker_backend),
        backend_execution_mode=execution_mode,
        query_mode=str(args.cotracker_query_mode),
        query_count_request=demo3_runtime.normalize_cotracker_query_count_request(args.cotracker_query_count),
        seed=int(args.cotracker_seed),
        sampling_device="cuda",
        init_requires_object_and_controller=True,
        overlay_max_points_per_camera=int(args.overlay_max_points_per_camera),
        overlay_display_scope=str(args.overlay_display_scope),
        input_max_age_ms=float(args.cotracker_input_max_age_ms),
        process_mode=str(args.cotracker_process_mode),
        device="cuda",
        prewarm_backends=bool(args.cotracker_prewarm_backends),
        update_mode=effective_legacy_update_mode(execution_mode),
        trackon2_checkpoint=args.trackon2_checkpoint,
        trackon2_config=args.trackon2_config,
        trackon2_repo_dir=args.trackon2_repo_dir,
        litetracker_weights=args.litetracker_weights,
        litetracker_repo_dir=args.litetracker_repo_dir,
        litetracker_runtime=normalize_litetracker_runtime(args.litetracker_runtime),
        litetracker_onnx_dir=args.litetracker_onnx_dir,
        litetracker_export_onnx=bool(args.litetracker_export_onnx),
        litetracker_onnx_opset=int(args.litetracker_onnx_opset),
        litetracker_onnx_optimization_level=int(args.litetracker_onnx_optimization_level),
        locotrack_repo_dir=args.locotrack_repo_dir,
        locotrack_checkpoint=args.locotrack_checkpoint,
        locotrack_model_size=str(args.locotrack_model_size),
        locotrack_window_frames=int(args.locotrack_window_frames),
        locotrack_resolution=parse_locotrack_resolution(args.locotrack_resolution),
        locotrack_query_chunk_size=int(args.locotrack_query_chunk_size),
        locotrack_autocast_dtype=str(args.locotrack_autocast_dtype),
        tracker_batch_query_count_policy=normalize_tracker_batch_query_count_policy(
            args.tracker_batch_query_count_policy
        ),
    )


def build_contract(
    args: argparse.Namespace,
    *,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    camera_ids = demo3_runtime.parse_camera_ids(args.camera_ids)
    render_waited_for_mask = str(args.fusion_mask_policy) == FUSION_MASK_POLICY_STRICT
    mode = demo3_runtime.resolve_demo3_mode(str(args.mode))
    query_count_request = demo3_runtime.normalize_cotracker_query_count_request(args.cotracker_query_count)
    tracker_backend = normalize_tracker_backend(args.cotracker_backend)
    tracker_spec = tracker_backend_spec(tracker_backend)
    litetracker_runtime = normalize_litetracker_runtime(args.litetracker_runtime)
    execution_mode = effective_tracking_backend_execution_mode(args)
    legacy_update_mode = effective_legacy_update_mode(execution_mode)
    tracker_visualization_mode = str(args.tracker_visualization_mode)
    tracker_surface_mode = tracker_visualization_mode == TRACKER_VISUALIZATION_MODE_SURFACE_MARKERS
    tracker_legacy_mode = tracker_visualization_mode == TRACKER_VISUALIZATION_MODE_LEGACY_3D_LIFT
    tracker_all_tracks_mode = tracker_visualization_mode == TRACKER_VISUALIZATION_MODE_ALL_TRACKS_3D_LIFT
    tracker_direct_depth_lift_mode = tracker_legacy_mode or tracker_all_tracks_mode
    controller_mask_erode_px = resolved_controller_mask_erode_px(args)
    overlay_scope_label = _overlay_scope_to_surface_label(str(args.overlay_display_scope))
    tracker_marker_color = (
        TRACKER_MARKER_LABEL_COLORS_RGB[SURFACE_ANCHOR_LABEL_UNION]
        if tracker_direct_depth_lift_mode
        else TRACKER_MARKER_LABEL_COLORS_RGB.get(
            overlay_scope_label,
            TRACKER_MARKER_LABEL_COLORS_RGB[SURFACE_ANCHOR_LABEL_UNION],
        )
    )
    if tracker_backend == TRACKER_BACKEND_LITETRACKER:
        tracker_prewarm_mode = "model_load_only" if bool(args.cotracker_prewarm_backends) else "lazy_query_init"
    elif tracker_backend == TRACKER_BACKEND_LOCOTRACK:
        tracker_prewarm_mode = "model_load_only" if bool(args.cotracker_prewarm_backends) else "disabled"
    else:
        tracker_prewarm_mode = "backend_model_prewarm" if bool(args.cotracker_prewarm_backends) else "disabled"
    tracker_query_dependent_init = tracker_backend == TRACKER_BACKEND_LITETRACKER
    tracker_online_semantics = "windowed" if tracker_backend == TRACKER_BACKEND_LOCOTRACK else "online"
    locotrack_resolution = parse_locotrack_resolution(args.locotrack_resolution)
    batch_enabled_by_contract = bool(
        tracker_spec.supports_batch_views
        and execution_mode in {TRACKING_BACKEND_EXECUTION_MODE_AUTO, TRACKING_BACKEND_EXECUTION_MODE_BATCH_VIEWS}
    )
    demo32 = is_demo32_preset(args)
    depth_source = demo3_runtime.DEPTH_SOURCE_FFS if demo32 else demo3_runtime.DEPTH_SOURCE_REALSENSE
    demo32_tracker_stage = (
        "litetracker_batch3_auto_fallback"
        if demo32 and execution_mode == TRACKING_BACKEND_EXECUTION_MODE_AUTO
        else "litetracker_batch3"
        if demo32 and legacy_update_mode == "batch"
        else "litetracker_serial"
    )
    pipeline_order = (
        "capture",
        "ffs_batch3_opt5_depth",
        "edgetam",
        demo32_tracker_stage,
        "render_and_diagnostics",
    ) if demo32 else (
        "capture",
        "realsense_depth",
        "edgetam",
        "point_tracker",
        "render_and_diagnostics",
    )
    hot_path_forbids = [
        "ffs_remote",
        "ffs_ir_alignment",
        "track_process_data.pkl",
        "inverse_physics",
        "cross_gpu_cuda_tensor_transfer",
    ]
    if not demo32:
        hot_path_forbids = ["ffs", "ffs_tensorrt", *hot_path_forbids]
    contract: dict[str, Any] = {
        "demo": demo_name_for_args(args),
        "preset": str(args.preset),
        "input_source": "live_realsense",
        "offline_mode_available": False,
        "offline_tracking_available": False,
        "init_mode": "sam31_first_frame",
        "sam31_init_quick_fail_empty_masks": bool(args.sam31_init_quick_fail_empty_masks),
        "sam31_init_min_mask_pixels": int(args.sam31_init_min_mask_pixels),
        "sam31_init_required_masks": [
            str(args.object_prompt),
            str(mode["controller_prompt"]),
        ],
        "mask_propagation": "hf_edgetam_online",
        "dual_gpu_enabled": True,
        "required_cuda_devices": 2,
        "physical_cuda_device_count": int(_cuda_count(cuda_device_count_provider)),
        "requires_three_realsense": True,
        "num_cameras": int(len(camera_ids)),
        "num_realsense_cameras": int(len(camera_ids)),
        "camera_ids": list(camera_ids),
        "serials": list(args.serials or []),
        "calibrate_path": str(args.calibrate_path),
        "calibrate_pkl_loaded": bool(Path(args.calibrate_path).is_file()),
        "mask_gpu_physical": int(args.mask_gpu),
        "cotracker_gpu_physical": int(args.cotracker_gpu),
        "ffs_gpu_physical": 0 if demo32 else None,
        "edgetam_gpu_physical": 0 if demo32 else int(args.mask_gpu),
        "sam31_gpu_physical": 0 if demo32 else int(args.mask_gpu),
        "litetracker_gpu_physical": int(args.cotracker_gpu) if demo32 else None,
        "locotrack_gpu_physical": int(args.cotracker_gpu) if tracker_backend == TRACKER_BACKEND_LOCOTRACK else None,
        "ffs_edgetam_same_gpu": bool(demo32),
        "shared_runtime_gpu_placement": (
            "ffs_edgetam_gpu0_litetracker_gpu1" if demo32 else "mask_gpu0_track_gpu1"
        ),
        "main_cuda_visible_devices": str(args.mask_gpu),
        "cotracker_cuda_visible_devices": str(args.cotracker_gpu),
        "gpu_plan": str(args.gpu_plan),
        "depth_source": depth_source,
        "uses_ffs": bool(demo32),
        "async_depth_pipeline": bool(demo32),
        "pipeline_order": list(pipeline_order),
        "shared_runtime_preset": (
            "demo2.3-dual4090-maxfps" if demo32 else "demo2.1.5-live-fast-native"
        ),
        "shared_runtime_gpu_pipeline_mode": "dual-gpu-split" if demo32 else "single-owner",
        "mask_source": demo3_runtime.MASK_SOURCE_HF_EDGETAM,
        "edgetam_batch_vision_encoder": True,
        "edgetam_live_session_keep_frames": int(args.edgetam_live_session_keep_frames),
        "edgetam_live_session_pruning": True,
        "semantic_mode": str(mode["semantic_mode"]),
        "shared_experiment_mode": str(mode["experiment_mode"]),
        "shared_runtime_track_mode": demo3_runtime.SHARED_TRACK_MODE_CONTROLLER_OBJECT,
        "tracking_mask_scope": demo3_runtime.TRACK_SCOPE_OBJECT_CONTROLLER_UNION,
        "object_prompt": str(args.object_prompt),
        "controller_prompt": str(mode["controller_prompt"]),
        "tracking_controller_label": str(mode["controller_label"]),
        "cotracker_enabled": not bool(args.disable_cotracker),
        "cotracker_backend": tracker_backend,
        "tracker_backend": tracker_backend,
        "tracker_backend_family": tracker_spec.family,
        "tracking_backend_spec": tracker_spec.to_dict(),
        "tracking_backend_execution_mode": execution_mode,
        "tracking_backend_batch_dimension": "camera" if batch_enabled_by_contract else "none",
        "tracking_backend_batch_size": int(len(camera_ids) if batch_enabled_by_contract else 1),
        "tracking_backend_batch_supported": bool(tracker_spec.supports_batch_views),
        "tracking_backend_supports_batch_views": bool(tracker_spec.supports_batch_views),
        "tracking_backend_supports_online": bool(tracker_spec.supports_online),
        "tracking_backend_online_semantics": tracker_online_semantics,
        "tracking_backend_batch_support_status": str(tracker_spec.batch_support_status),
        "tracking_backend_batch_auto_selected": bool(
            execution_mode == TRACKING_BACKEND_EXECUTION_MODE_AUTO and tracker_spec.supports_batch_views
        ),
        "tracker_batch_query_count_policy": normalize_tracker_batch_query_count_policy(
            args.tracker_batch_query_count_policy
        ),
        "trackon2_checkpoint": args.trackon2_checkpoint,
        "trackon2_config": args.trackon2_config,
        "trackon2_repo_dir": args.trackon2_repo_dir,
        "litetracker_weights": args.litetracker_weights,
        "litetracker_repo_dir": args.litetracker_repo_dir,
        "litetracker_runtime": litetracker_runtime,
        "litetracker_onnx_dir": args.litetracker_onnx_dir,
        "litetracker_export_onnx": bool(args.litetracker_export_onnx),
        "litetracker_onnx_opset": int(args.litetracker_onnx_opset),
        "litetracker_onnx_opset_actual": max(int(args.litetracker_onnx_opset), 18),
        "litetracker_onnx_optimization_level": int(args.litetracker_onnx_optimization_level),
        "locotrack_model_size": str(args.locotrack_model_size),
        "locotrack_window_frames": int(args.locotrack_window_frames),
        "locotrack_resolution": [int(locotrack_resolution[0]), int(locotrack_resolution[1])],
        "locotrack_query_chunk_size": int(args.locotrack_query_chunk_size),
        "locotrack_autocast_dtype": str(args.locotrack_autocast_dtype),
        "locotrack_checkpoint": args.locotrack_checkpoint,
        "locotrack_repo_dir": args.locotrack_repo_dir,
        "tracker_env_name": "demo_3_1_max",
        "ffs_contract": (
            {
                "checkpoint": DEFAULT_FFS_MODEL_NAME,
                "valid_iters": DEFAULT_FFS_VALID_ITERS,
                "max_disp": DEFAULT_FFS_MAX_DISP,
                "builderOptimizationLevel": DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
                "trt_batch_size": 3,
                "trt_model_dir": str(DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR),
                "worker_mode": "dual-gpu-split-ffs-worker",
                "schedule": "strict3-latest",
                "depth_stage": "before_edgetam",
                "batch3_isolated_artifact": True,
            }
            if demo32
            else None
        ),
        "cotracker_owner": "process",
        "cotracker_process_mode": str(args.cotracker_process_mode),
        "cotracker_prewarm_backends": bool(args.cotracker_prewarm_backends),
        "tracker_prewarm_backends": bool(args.cotracker_prewarm_backends),
        "tracker_prewarm_mode": tracker_prewarm_mode,
        "tracker_ready_state": "ready_to_receive_inputs",
        "tracker_query_dependent_init": bool(tracker_query_dependent_init),
        "tracker_query_dependent_init_pending_until_first_input": bool(tracker_query_dependent_init),
        "cotracker_update_mode": legacy_update_mode,
        "cotracker_batch_size_target": int(len(camera_ids)),
        "cotracker_batch_fallback_enabled": execution_mode == TRACKING_BACKEND_EXECUTION_MODE_AUTO,
        "cotracker_input_fps": float(args.cotracker_input_fps),
        "cotracker_input_max_age_ms": float(args.cotracker_input_max_age_ms),
        "cotracker_result_stale_timeout_ms": float(args.cotracker_result_stale_timeout_ms),
        "wait_for_tracking_overlay": bool(args.wait_for_tracking_overlay),
        "tracking_overlay_required_before_first_render": not bool(args.disable_cotracker),
        "tracking_overlay_required_for_render": not bool(args.disable_cotracker),
        "render_requires_new_cotracker_result": not bool(args.disable_cotracker),
        "render_reuses_cached_cotracker_result": False,
        "tracking_overlay_color_rgb": [int(v) for v in tracker_marker_color],
        "tracking_overlay_color_mode": "by_camera" if bool(args.overlay_debug_color_by_camera) else "solid",
        "tracking_overlay_debug_color_by_camera": bool(args.overlay_debug_color_by_camera),
        "tracker_visualization_mode": tracker_visualization_mode,
        "tracker_3d_marker_mode": "surface_snap" if tracker_surface_mode else tracker_visualization_mode,
        "tracker_3d_marker_shape": "sphere",
        "tracker_legacy_lift_used": bool(tracker_legacy_mode),
        "tracker_direct_depth_lift_used": bool(tracker_direct_depth_lift_mode),
        "tracker_all_tracks_anchor_mode": bool(tracker_all_tracks_mode),
        "tracker_surface_gate_enabled": bool(tracker_surface_mode),
        "tracker_3d_snap_radius_px": float(args.tracker_3d_snap_radius_px),
        "tracker_3d_marker_radius_m": float(args.tracker_3d_marker_radius_m),
        "tracker_control_points_per_camera": int(args.tracker_control_points_per_camera),
        "tracker_control_point_selection": str(args.tracker_control_point_selection),
        "tracking_overlay_lift_method": (
            "surface_snap"
            if tracker_surface_mode
            else "all_tracks_depth_lift"
            if tracker_all_tracks_mode
            else "semantic_projection_grid"
        ),
        "tracking_query_mode": demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE,
        "tracking_query_count_requested": str(query_count_request),
        "tracking_query_count_rule": demo3_runtime.TRACKING_QUERY_COUNT_RULE_PHYSTWIN_DENSE,
        "tracking_sampling": demo3_runtime.TRACKING_SAMPLING_TORCH_RANDPERM,
        "tracking_max_query_points_per_camera": demo3_runtime.PHYSTWIN_DENSE_MAX_POINTS,
        "trackable_mask_build_policy": str(args.trackable_mask_build_policy),
        "trackable_mask_build_stage": "first_valid_tracking_input",
        "trackable_query_init_strategy": str(args.trackable_query_init_strategy),
        "trackable_mask_source": (
            "standard_filter_survivors"
            if demo32 and str(args.trackable_mask_build_policy) != TRACKABLE_MASK_BUILD_POLICY_DISABLED
            else "raw_semantic_union"
        ),
        "tracking_input_mask_semantics": (
            "standard_filter_trackable_masks"
            if demo32 and str(args.trackable_mask_build_policy) != TRACKABLE_MASK_BUILD_POLICY_DISABLED
            else "raw_semantic_masks"
        ),
        "tracker_query_source": (
            "union_trackable_mask"
            if demo32 and str(args.trackable_mask_build_policy) != TRACKABLE_MASK_BUILD_POLICY_DISABLED
            else "object_controller_union_mask"
        ),
        "object_mask_semantics": (
            "object_trackable_mask"
            if demo32 and str(args.trackable_mask_build_policy) != TRACKABLE_MASK_BUILD_POLICY_DISABLED
            else "raw_object_mask"
        ),
        "controller_mask_semantics": (
            "controller_trackable_mask"
            if demo32 and str(args.trackable_mask_build_policy) != TRACKABLE_MASK_BUILD_POLICY_DISABLED
            else "raw_controller_mask"
        ),
        "controller_trackable_max_points_per_camera": int(args.controller_trackable_max_points_per_camera),
        "controller_trackable_cap_stage": "after_standard_filter",
        "controller_mask_erode_px": int(controller_mask_erode_px),
        "controller_mask_erode_unit": "px",
        "controller_mask_erode_stage": "before_tracking_union_and_trackable_filter",
        "controller_mask_erode_applies_to": "tracking_input_and_anchor_masks",
        "controller_mask_erode_default_source": (
            "mode_demo"
            if str(mode["semantic_mode"]) == demo3_runtime.MODE_DEMO
            and int(controller_mask_erode_px) == DEFAULT_DEMO_MODE_CONTROLLER_MASK_ERODE_PX
            else "explicit_or_non_demo_default"
        ),
        "controller_pcd_max_points_per_camera": demo3_runtime.normalize_controller_pcd_max_points_per_camera(
            args.controller_pcd_max_points_per_camera
        ),
        "controller_pcd_cap_stage": demo3_runtime.CONTROLLER_PCD_CAP_STAGE,
        "controller_pcd_cap_sampling": demo3_runtime.CONTROLLER_PCD_CAP_SAMPLING,
        "cotracker_seed": int(args.cotracker_seed),
        "phystwin_dense_compatible": bool(demo3_runtime.phystwin_dense_compatible_for_args(args)),
        "cotracker_window_len": demo3_runtime.DEFAULT_COTRACKER_WINDOW_LEN,
        "cotracker_publish_step": demo3_runtime.DEFAULT_COTRACKER_PUBLISH_STEP,
        "ipc_payload": "cpu_numpy_latest_wins",
        "tracking_input_contains_depth": False,
        "tracking_input_contains_intrinsics": False,
        "tracking_input_contains_c2w": False,
        "world_lift_owner": "main_process",
        "cross_gpu_cuda_tensor_transfer": False,
        "shared_runtime_tracking_backend": "none",
        "overlay_max_points_per_camera": int(args.overlay_max_points_per_camera),
        "overlay_display_scope": str(args.overlay_display_scope),
        "overlay_display_classification": "first_frame_mask_membership",
        "overlay_bbox_filter_enabled": bool(args.overlay_reject_outside_semantic_bbox and not tracker_all_tracks_mode),
        "overlay_bbox_filter_scope": str(args.overlay_display_scope),
        "overlay_bbox_filter_margin_m": float(args.overlay_max_distance_from_controller_m),
        "tracking_control_point_markers": bool(
            tracker_surface_mode
            or tracker_all_tracks_mode
            or (tracker_legacy_mode and bool(args.overlay_control_point_markers))
        ),
        "tracking_control_point_count_requested": (
            int(args.tracker_control_points_per_camera) * len(camera_ids)
            if tracker_surface_mode
            else 0
            if tracker_all_tracks_mode
            else int(args.overlay_control_point_count)
        ),
        "tracking_control_points_per_camera": int(args.tracker_control_points_per_camera),
        "tracking_control_point_radius_m": (
            float(args.tracker_3d_marker_radius_m)
            if tracker_surface_mode
            or tracker_all_tracks_mode
            else float(args.overlay_control_point_radius_m)
        ),
        "tracking_control_point_color_rgb": [int(v) for v in tracker_marker_color],
        "tracking_control_point_sampling": (
            f"{args.tracker_control_point_selection}_surface_snap"
            if tracker_surface_mode
            else "all_visible_depth_valid_tracks_no_surface_or_bbox_gate"
            if tracker_all_tracks_mode
            else "farthest_point_sample_after_lift_scope_and_bbox"
        ),
        "overlay_render_raw_track_points": bool(args.overlay_render_raw_track_points and tracker_legacy_mode),
        "overlay_trail_len": int(args.overlay_trail_len),
        "overlay_stale_timeout_ms": float(args.overlay_stale_timeout_ms),
        "fusion_mask_policy": str(args.fusion_mask_policy),
        "mask_stale_timeout_ms": float(args.mask_stale_timeout_ms),
        "render_mode": str(args.render_mode),
        "render_target_fps": float(args.render_target_fps),
        "render_resample_latest": bool(args.render_resample_latest),
        "render_backend": None if args.render_backend is None else str(args.render_backend),
        "render_layer_mode": None if args.render_layer_mode is None else str(args.render_layer_mode),
        "render_copy_mode": None if args.render_copy_mode is None else str(args.render_copy_mode),
        "pcd_color_mode": str(args.pcd_color_mode),
        "render_micro_profile": True,
        "render_latest_wins": True,
        "render_waited_for_cotracker": not bool(args.disable_cotracker),
        "render_waited_for_fresh_cotracker_result": not bool(args.disable_cotracker),
        "render_driver": "cotracker_child_output",
        "render_trigger": "new_cotracker_result",
        "tracking_pending_render_packet_max_groups": int(DEFAULT_PENDING_RENDER_PACKET_GROUPS),
        "tracking_render_packet_match_policy": TRACKING_RENDER_PACKET_MATCH_POLICY,
        "render_waited_for_mask": bool(render_waited_for_mask),
        "render_object_filter": {
            "point_control": str(args.object_point_control),
            "voxel_m": float(args.object_volume_voxel_m),
            "origin_policy": str(args.object_volume_origin),
            "adaptive": bool(args.object_volume_adaptive),
            "min_voxel_m": float(args.object_volume_min_voxel_m),
            "max_voxel_m": float(args.object_volume_max_voxel_m),
            "target_ms": float(args.object_volume_target_ms),
            "emergency_max_points": int(args.object_volume_emergency_max_points),
            "points_per_voxel": int(args.object_volume_points_per_voxel),
        },
        "render_controller_filter": {
            "render_voxel_m": float(args.controller_render_voxel_m),
            "render_voxel_downsample": float(args.controller_render_voxel_m) > 0.0,
            "render_max_points": int(args.controller_render_max_points),
            "render_cap_enabled": int(args.controller_render_max_points) > 0,
            "render_only": True,
            "affects_tracking_markers": False,
        },
        "debug_fusion": {
            "color_by_camera": bool(args.debug_color_by_camera),
            "save_per_camera_pcd": bool(args.debug_save_per_camera_pcd),
            "save_mask_overlays": bool(args.debug_save_mask_overlays),
            "identity_c2w": bool(args.debug_identity_c2w),
            "invert_c2w": bool(args.debug_invert_c2w),
            "only_camera_idx": None if args.debug_only_camera_idx is None else int(args.debug_only_camera_idx),
            "max_saved_groups": (
                None if args.debug_fusion_max_saved_groups is None else int(args.debug_fusion_max_saved_groups)
            ),
        },
        "gpu_sampling": {
            "enabled": bool(args.gpu_sampling),
            "interval_s": float(args.gpu_sampling_interval_s),
            "backend": str(args.gpu_sampling_backend),
            "device_index": int(args.gpu_sampling_device_index),
            "device_indexes": (
                list(demo3_runtime._gpu_sampling_device_indexes_for_args(args))
                if demo3_runtime._gpu_sampling_device_indexes_for_args(args) is not None
                else None
            ),
        },
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "output_root": str(args.output_root),
        "hot_path_forbids": hot_path_forbids,
    }
    contract["profile_summary_fields"] = build_empty_dual_gpu_profile_summary(contract)
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "input_source",
        "offline_mode_available",
        "dual_gpu_enabled",
        "required_cuda_devices",
        "mask_gpu_physical",
        "cotracker_gpu_physical",
        "ffs_gpu_physical",
        "edgetam_gpu_physical",
        "sam31_gpu_physical",
        "litetracker_gpu_physical",
        "locotrack_gpu_physical",
        "ffs_edgetam_same_gpu",
        "shared_runtime_gpu_placement",
        "main_cuda_visible_devices",
        "cotracker_cuda_visible_devices",
        "depth_source",
        "uses_ffs",
        "async_depth_pipeline",
        "shared_runtime_preset",
        "shared_runtime_gpu_pipeline_mode",
        "pipeline_order",
        "ffs_contract",
        "mask_source",
        "edgetam_batch_vision_encoder",
        "edgetam_live_session_keep_frames",
        "edgetam_live_session_pruning",
        "init_mode",
        "sam31_init_quick_fail_empty_masks",
        "sam31_init_min_mask_pixels",
        "sam31_init_required_masks",
        "mask_propagation",
        "semantic_mode",
        "tracking_mask_scope",
        "tracking_query_mode",
        "tracking_query_count_requested",
        "tracking_query_count_rule",
        "tracking_sampling",
        "trackable_mask_build_policy",
        "trackable_query_init_strategy",
        "trackable_mask_source",
        "tracking_input_mask_semantics",
        "tracker_query_source",
        "controller_trackable_max_points_per_camera",
        "controller_trackable_cap_stage",
        "controller_mask_erode_px",
        "controller_mask_erode_stage",
        "controller_mask_erode_applies_to",
        "controller_pcd_max_points_per_camera",
        "controller_pcd_cap_stage",
        "cotracker_seed",
        "wait_for_tracking_overlay",
        "render_requires_new_cotracker_result",
        "render_reuses_cached_cotracker_result",
        "tracker_visualization_mode",
        "tracker_3d_marker_mode",
        "tracker_3d_marker_shape",
        "tracker_legacy_lift_used",
        "tracker_direct_depth_lift_used",
        "tracker_all_tracks_anchor_mode",
        "tracker_surface_gate_enabled",
        "tracker_3d_snap_radius_px",
        "tracker_3d_marker_radius_m",
        "tracker_control_points_per_camera",
        "tracker_control_point_selection",
        "tracking_overlay_lift_method",
        "tracking_overlay_color_mode",
        "overlay_max_points_per_camera",
        "overlay_display_scope",
        "overlay_bbox_filter_enabled",
        "overlay_bbox_filter_margin_m",
        "tracking_control_point_markers",
        "tracking_control_point_count_requested",
        "tracking_control_point_radius_m",
        "overlay_render_raw_track_points",
        "phystwin_dense_compatible",
        "cotracker_backend",
        "tracker_backend",
        "tracker_backend_family",
        "tracker_env_name",
        "litetracker_runtime",
        "litetracker_onnx_dir",
        "litetracker_export_onnx",
        "litetracker_onnx_opset",
        "litetracker_onnx_opset_actual",
        "litetracker_onnx_optimization_level",
        "locotrack_model_size",
        "locotrack_window_frames",
        "locotrack_resolution",
        "locotrack_query_chunk_size",
        "locotrack_autocast_dtype",
        "locotrack_checkpoint",
        "locotrack_repo_dir",
        "tracking_backend_execution_mode",
        "tracking_backend_batch_dimension",
        "tracking_backend_batch_size",
        "tracking_backend_batch_supported",
        "tracking_backend_supports_batch_views",
        "tracking_backend_supports_online",
        "tracking_backend_online_semantics",
        "tracker_batch_query_count_policy",
        "tracker_prewarm_mode",
        "tracker_ready_state",
        "tracker_query_dependent_init",
        "cotracker_owner",
        "cotracker_process_mode",
        "cotracker_prewarm_backends",
        "cotracker_update_mode",
        "cross_gpu_cuda_tensor_transfer",
        "ipc_payload",
        "fusion_mask_policy",
        "pcd_color_mode",
        "render_waited_for_cotracker",
        "render_waited_for_fresh_cotracker_result",
        "render_driver",
        "render_trigger",
        "tracking_pending_render_packet_max_groups",
        "tracking_render_packet_match_policy",
        "render_waited_for_mask",
        "render_controller_filter",
        "output_root",
    )
    lines = []
    for key in keys:
        value = contract[key]
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"{key} = {rendered}")
    lines.append(json.dumps(contract, indent=2, sort_keys=True))
    return "\n".join(lines)


def _write_profile(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fresh_tracking_result_or_none(
    result: TrackingResultLitePacket | None,
    *,
    now_s: float,
    stale_timeout_ms: float,
) -> TrackingResultLitePacket | None:
    if result is None:
        return None
    age_ms = max(0.0, (float(now_s) - float(result.publish_timestamp_s)) * 1000.0)
    if age_ms > float(stale_timeout_ms):
        return None
    return result


def validate_live_realsense_contract(
    args: argparse.Namespace,
    *,
    connected_serials_provider: ConnectedSerialsProvider | None = None,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    validate_args(args, require_calibration=True, cuda_device_count_provider=cuda_device_count_provider)
    demo_label = demo_label_for_args(args)
    provider = connected_serials_provider or demo3_runtime._get_connected_realsense_serials
    connected_serials = list(provider())
    requested_serials = list(args.serials or [])
    if requested_serials:
        if len(requested_serials) != 3:
            raise RuntimeError(f"{demo_label} requires exactly three requested RealSense serials when --serials is used.")
        missing = [serial for serial in requested_serials if serial not in connected_serials]
        if missing:
            raise RuntimeError(f"{demo_label} requested RealSense serials are not connected: {missing}")
        active_serials = requested_serials
    else:
        if len(connected_serials) != 3:
            raise RuntimeError(
                f"{demo_label} requires exactly three connected RealSense cameras when --serials is not provided. "
                f"connected={len(connected_serials)}"
            )
        active_serials = connected_serials

    from qqtt.env.camera.calibration_metadata import load_calibration_reference_serials

    calibration_reference_serials = load_calibration_reference_serials(args.calibrate_path)
    if calibration_reference_serials is not None:
        if len(calibration_reference_serials) != 3:
            raise RuntimeError(
                f"{demo_label} requires calibrate.pkl metadata for exactly three cameras. "
                f"calibration_reference_serials={len(calibration_reference_serials)}"
            )
        missing_from_calibration = [serial for serial in active_serials if serial not in calibration_reference_serials]
        if missing_from_calibration:
            raise RuntimeError(
                f"{demo_label} active RealSense serials are not covered by calibrate.pkl metadata. "
                f"missing={missing_from_calibration}"
            )
    try:
        calibration_transform_count = demo3_runtime._calibration_transform_count(args.calibrate_path)
    except Exception as exc:
        raise RuntimeError(f"{demo_label} calibration validation failed: {exc}") from exc
    if calibration_transform_count != 3:
        raise RuntimeError(
            f"{demo_label} requires calibrate.pkl to contain exactly three camera-to-world transforms. "
            f"transform_count={calibration_transform_count}"
        )
    return {
        "connected_serials": connected_serials,
        "active_serials": active_serials,
        "calibration_reference_serials": calibration_reference_serials,
        "calibration_transform_count": int(calibration_transform_count),
    }


def build_shared_runtime_args(
    args: argparse.Namespace,
    *,
    shared_runtime_module: Any | None,
    live_validation: dict[str, Any],
    shared_profile_path: Path | None,
) -> argparse.Namespace:
    shared = shared_runtime_module or demo3_runtime._load_shared_runtime_module()
    shared_args = demo3_runtime.build_shared_runtime_args(
        args,
        shared_runtime_module=shared,
        live_validation=live_validation,
        shared_profile_path=shared_profile_path,
    )
    shared_args.tracking_backend = "none"
    shared_args.tracking_source = "cached"
    shared_args.show_tracking_overlay = False
    tracker_backend = normalize_tracker_backend(args.cotracker_backend)
    if tracker_backend == TRACKER_BACKEND_LITETRACKER:
        tracker_prewarm_mode = "model_load_only" if bool(args.cotracker_prewarm_backends) else "lazy_query_init"
    elif tracker_backend == TRACKER_BACKEND_LOCOTRACK:
        tracker_prewarm_mode = "model_load_only" if bool(args.cotracker_prewarm_backends) else "disabled"
    else:
        tracker_prewarm_mode = "backend_model_prewarm" if bool(args.cotracker_prewarm_backends) else "disabled"
    shared_args.external_tracker_backend = tracker_backend
    shared_args.external_tracker_update_mode = effective_tracking_backend_execution_mode(args)
    shared_args.external_tracker_prewarm_mode = tracker_prewarm_mode
    shared_args.external_tracker_required_for_render = bool(args.wait_for_tracking_overlay)
    shared_args.external_tracker_marker_required = bool(
        is_demo32_preset(args) and not bool(args.disable_cotracker) and bool(args.overlay_control_point_markers)
    )
    if is_demo32_preset(args):
        shared_args.preset = shared.PRESET_DEMO23_DUAL4090_MAXFPS
        shared_args.preset_canonical = shared.PRESET_DEMO23_DUAL4090_MAXFPS
        shared_args.demo_version_override = "demo3.2"
        shared_args.demo_display_name_override = "Demo 3.2"
        shared_args.depth_source = shared.DEPTH_SOURCE_FFS
        shared_args.ffs_trt_batch_size = 3
        shared_args.ffs_trt_model_dir = str(DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR)
        shared_args.gpu_pipeline_mode = shared.GPU_PIPELINE_MODE_DUAL_GPU_SPLIT
        shared_args.ffs_worker_mode = "shared"
        shared_args.ffs_schedule = "strict3-latest"
        shared_args.ffs_device = "cuda:0"
        shared_args.edgetam_device = "cuda:0"
        shared_args.sam31_device = "cuda:0"
        shared_args.demo32_ffs_edgetam_same_gpu = True
        shared_args.demo32_gpu_placement = "ffs_edgetam_gpu0_litetracker_gpu1"
        shared_args.dual_gpu_queue_size = 2
        shared_args.dual_gpu_transport = "pickle"
        shared_args.dual_gpu_start_method = "spawn"
        shared_args.dual_gpu_processes = True
        shared_args.enable_pcd_filter = True
        shared_args.pcd_filter_mode = "async"
        shared_args.depth_min_m = shared.DEFAULT_DEMO22_DEPTH_MIN_M
    else:
        shared_args.depth_source = demo3_runtime.DEPTH_SOURCE_REALSENSE
    shared_args.edgetam_batch_vision_encoder = True
    if hasattr(shared_args, "render_target_fps"):
        shared_args.render_target_fps = float(args.render_target_fps)
    shared_args.demo31_top_level_profile_json_output = args.profile_json_output
    shared_args.overlay_debug_color_by_camera = bool(args.overlay_debug_color_by_camera)
    shared_args.tracker_visualization_mode = str(args.tracker_visualization_mode)
    shared_args.tracker_3d_snap_radius_px = float(args.tracker_3d_snap_radius_px)
    shared_args.tracker_3d_marker_radius_m = float(args.tracker_3d_marker_radius_m)
    shared_args.tracker_control_points_per_camera = int(args.tracker_control_points_per_camera)
    shared_args.tracker_control_point_selection = str(args.tracker_control_point_selection)
    shared_args.overlay_control_point_markers = bool(args.overlay_control_point_markers)
    shared_args.overlay_control_point_count = int(args.overlay_control_point_count)
    shared_args.overlay_control_point_radius_m = float(args.overlay_control_point_radius_m)
    shared_args.overlay_render_raw_track_points = bool(args.overlay_render_raw_track_points)
    shared_args.overlay_display_scope = str(args.overlay_display_scope)
    shared_args.overlay_reject_outside_semantic_bbox = bool(args.overlay_reject_outside_semantic_bbox)
    shared_args.overlay_max_distance_from_controller_m = float(args.overlay_max_distance_from_controller_m)
    shared_args.trackable_mask_build_policy = str(args.trackable_mask_build_policy)
    shared_args.trackable_query_init_strategy = str(args.trackable_query_init_strategy)
    shared_args.controller_trackable_max_points_per_camera = int(args.controller_trackable_max_points_per_camera)
    shared_args.controller_mask_erode_px = resolved_controller_mask_erode_px(args)
    shared_args.controller_render_voxel_m = float(args.controller_render_voxel_m)
    shared_args.controller_render_max_points = int(args.controller_render_max_points)
    shared_args.sam31_init_quick_fail_empty_masks = bool(args.sam31_init_quick_fail_empty_masks)
    shared_args.sam31_init_min_mask_pixels = int(args.sam31_init_min_mask_pixels)
    return shared_args


def erode_binary_mask_px(mask: np.ndarray, erode_px: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    radius = int(erode_px)
    if radius <= 0 or mask_bool.size == 0:
        return np.ascontiguousarray(mask_bool, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError("controller mask erosion expects a 2D mask")
    height, width = mask_bool.shape
    padded = np.pad(mask_bool, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
    eroded = np.ones((height, width), dtype=bool)
    kernel_width = 2 * radius + 1
    for dy in range(kernel_width):
        for dx in range(kernel_width):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return np.ascontiguousarray(eroded, dtype=bool)


def _phystwin_union_tracking_masks(
    mask_packet: Any,
    *,
    controller_mask_erode_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_mask = np.asarray(mask_packet.object_mask, dtype=bool)
    controller_mask = erode_binary_mask_px(mask_packet.controller_mask, controller_mask_erode_px)
    union_mask = np.asarray(object_mask | controller_mask, dtype=bool)
    return union_mask, object_mask, controller_mask


class Demo31MaskPolicyJoinBuffer:
    """Join capture/depth with strict or latest-reuse mask semantics."""

    def __init__(
        self,
        *,
        max_groups: int = 8,
        policy: str = FUSION_MASK_POLICY_LATEST_REUSE,
        stale_timeout_ms: float = DEFAULT_MASK_STALE_TIMEOUT_MS,
    ) -> None:
        self.max_groups = int(max_groups)
        self.policy = str(policy)
        self.stale_timeout_ms = float(stale_timeout_ms)
        self._captures: dict[int, Any] = {}
        self._depths: dict[int, Any] = {}
        self._masks: dict[int, tuple[Any, float]] = {}
        self.capture_stale_drops = 0
        self.depth_stale_drops = 0
        self.mask_stale_drops = 0
        self.ready_join_count = 0
        self.mask_selection_count = 0
        self.mask_reuse_count = 0
        self.mask_age_ms_samples: list[float] = []
        self.mask_group_delta_samples: list[float] = []
        self._selection_by_group: dict[int, dict[str, Any]] = {}

    def put_capture(self, group: Any) -> None:
        self._captures[int(group.group_id)] = group
        self._prune()

    def put_depth(self, depth: Any) -> None:
        self._depths[int(depth.group_id)] = depth
        self._prune()

    def put_mask(self, mask: Any) -> None:
        self._masks[int(mask.group_id)] = (mask, time.perf_counter())
        self._prune()

    def pop_latest_ready(self) -> tuple[Any, Any, Any] | None:
        ready_depth = sorted(set(self._captures) & set(self._depths))
        if not ready_depth:
            return None
        now_s = time.perf_counter()
        for group_id in reversed(ready_depth):
            mask = self._select_mask_for_group(group_id=group_id, now_s=now_s)
            if mask is None:
                continue
            capture = self._captures.pop(group_id)
            depth = self._depths.pop(group_id)
            self.ready_join_count += 1
            self._drop_older_capture_depth(group_id)
            return capture, depth, mask
        return None

    def snapshot(self) -> dict[str, Any]:
        age = percentile_summary(self.mask_age_ms_samples)
        return {
            "max_groups": int(self.max_groups),
            "policy": str(self.policy),
            "capture_pending": int(len(self._captures)),
            "depth_pending": int(len(self._depths)),
            "mask_pending": int(len(self._masks)),
            "capture_stale_drops": int(self.capture_stale_drops),
            "depth_stale_drops": int(self.depth_stale_drops),
            "mask_stale_drops": int(self.mask_stale_drops),
            "ready_join_count": int(self.ready_join_count),
            "mask_selection_count": int(self.mask_selection_count),
            "mask_reuse_count": int(self.mask_reuse_count),
            "mask_reuse_ratio": float(self.mask_reuse_count / self.mask_selection_count)
            if self.mask_selection_count
            else 0.0,
            "mask_age_ms_median": float(age["median"]),
            "mask_age_ms_p95": float(age["p95"]),
            "mask_group_delta_median": float(percentile_summary(self.mask_group_delta_samples)["median"]),
            "mask_group_delta_p95": float(percentile_summary(self.mask_group_delta_samples)["p95"]),
        }

    def selection_for_group(self, group_id: int) -> dict[str, Any] | None:
        item = self._selection_by_group.get(int(group_id))
        return None if item is None else dict(item)

    def _select_mask_for_group(self, *, group_id: int, now_s: float) -> Any | None:
        if not self._masks:
            return None
        if self.policy == FUSION_MASK_POLICY_STRICT:
            entry = self._masks.get(int(group_id))
            if entry is None:
                return None
            mask_group, arrival_s = entry
        else:
            source_group_id = max(self._masks)
            mask_group, arrival_s = self._masks[source_group_id]
        age_ms = max(0.0, (float(now_s) - float(arrival_s)) * 1000.0)
        if age_ms > self.stale_timeout_ms:
            self.mask_stale_drops += 1
            return None
        self.mask_selection_count += 1
        self.mask_age_ms_samples.append(float(age_ms))
        source_group_id = int(mask_group.group_id)
        reused = source_group_id != int(group_id)
        self.mask_group_delta_samples.append(float(abs(int(group_id) - source_group_id)))
        self._selection_by_group[int(group_id)] = {
            "target_group_id": int(group_id),
            "source_group_id": int(source_group_id),
            "age_ms": float(age_ms),
            "reused": bool(reused),
        }
        if reused:
            self.mask_reuse_count += 1
        return self._retarget_mask_group(
            mask_group,
            target_group_id=int(group_id),
            source_group_id=source_group_id,
            age_ms=float(age_ms),
            reused=bool(reused),
        )

    def _retarget_mask_group(
        self,
        mask_group: Any,
        *,
        target_group_id: int,
        source_group_id: int,
        age_ms: float,
        reused: bool,
    ) -> Any:
        if int(mask_group.group_id) == int(target_group_id):
            packets = dict(mask_group.mask_packets)
        else:
            packets = {
                int(camera_idx): replace(packet, group_id=int(target_group_id))
                for camera_idx, packet in mask_group.mask_packets.items()
            }
        return Demo31RetargetedMaskGroup(
            group_id=int(target_group_id),
            mask_packets=packets,
            edgetam_stage_wall_ms=float(mask_group.edgetam_stage_wall_ms),
            edgetam_stage_sum_model_ms=float(mask_group.edgetam_stage_sum_model_ms),
            edgetam_stage_mode=str(mask_group.edgetam_stage_mode),
            source_group_id=int(source_group_id),
            mask_age_ms=float(age_ms),
            mask_reused=bool(reused),
        )

    def _drop_older_capture_depth(self, group_id: int) -> None:
        for table, counter_name in (
            (self._captures, "capture_stale_drops"),
            (self._depths, "depth_stale_drops"),
        ):
            stale = [old_group_id for old_group_id in table if old_group_id < group_id]
            for old_group_id in stale:
                table.pop(old_group_id, None)
            setattr(self, counter_name, getattr(self, counter_name) + len(stale))

    def _prune(self) -> None:
        for table, counter_name in (
            (self._captures, "capture_stale_drops"),
            (self._depths, "depth_stale_drops"),
            (self._masks, "mask_stale_drops"),
        ):
            while len(table) > self.max_groups:
                oldest = min(table)
                table.pop(oldest, None)
                setattr(self, counter_name, getattr(self, counter_name) + 1)
        keep_after = min([*self._captures, *self._depths, *self._masks], default=None)
        if keep_after is not None:
            stale = [group_id for group_id in self._selection_by_group if group_id < keep_after]
            for group_id in stale:
                self._selection_by_group.pop(group_id, None)


def make_demo31_live_runtime_class(shared_runtime_module: Any, *, process_client_factory: ProcessClientFactory | None = None):
    base_cls = shared_runtime_module.Demo21Runtime

    class Demo31LiveRuntime(base_cls):
        def __init__(
            self,
            args: argparse.Namespace,
            *,
            demo31_contract: dict[str, Any],
            cotracker_process_config: CoTrackerProcessConfig,
            cotracker_enabled: bool = True,
        ) -> None:
            super().__init__(args)
            if not hasattr(self, "_summary"):
                self._summary = {}
            if not hasattr(self, "_init_profile_update"):
                self._init_profile_update = lambda *_args, **_kwargs: None
            if not hasattr(self, "_profile_rel_s"):
                self._profile_rel_s = lambda *_args, **_kwargs: 0.0
            self.demo31_contract = dict(demo31_contract)
            self.demo31_cotracker_enabled = bool(cotracker_enabled)
            self.demo31_cotracker_config = cotracker_process_config
            self.demo31_process_client = (
                (process_client_factory or start_cotracker_process)(cotracker_process_config)
            if self.demo31_cotracker_enabled
            else None
            )
            self.demo31_process_status_events: list[dict[str, Any]] = []
            if self.demo31_process_client is not None:
                self._summary["demo31_cotracker_process_eager_started_before_camera"] = True
                self._summary["demo31_cotracker_pid"] = int(getattr(self.demo31_process_client, "pid", 0) or 0)
                self._init_profile_update(
                    ("demo31", "cotracker_process", "eager_start"),
                    {
                        "enabled": True,
                        "before_camera_startup": True,
                        "pid": int(getattr(self.demo31_process_client, "pid", 0) or 0),
                        "prewarm_backends": bool(getattr(cotracker_process_config, "prewarm_backends", True)),
                        "started_s": self._profile_rel_s(),
                    },
                )
            self.stage_join_buffer = Demo31MaskPolicyJoinBuffer(
                max_groups=8,
                policy=str(self.demo31_contract["fusion_mask_policy"]),
                stale_timeout_ms=float(self.demo31_contract["mask_stale_timeout_ms"]),
            )
            self.demo31_lift_input_cache = Demo31LiftInputCache()
            self.demo31_surface_anchor_cache = Demo31SurfaceAnchorCache(
                max_groups=int(
                    self.demo31_contract.get(
                        "tracking_pending_render_packet_max_groups",
                        DEFAULT_PENDING_RENDER_PACKET_GROUPS,
                    )
                )
            )
            self.demo31_mask_cache = LatestMaskCache()
            self.demo31_last_tracking_input_s: float | None = None
            self.demo31_tracking_input_publish_times_s: list[float] = []
            self.demo31_tracking_input_skip_count = 0
            self.demo31_tracking_input_queue_replace_count = 0
            self.demo31_tracking_input_drop_count = 0
            self.demo31_pending_render_packets: dict[int, Any] = {}
            self.demo31_pending_render_lock = threading.Lock()
            self.demo31_pending_render_packet_max_groups = max(
                1,
                int(
                    self.demo31_contract.get(
                        "tracking_pending_render_packet_max_groups",
                        DEFAULT_PENDING_RENDER_PACKET_GROUPS,
                    )
                ),
            )
            self.demo31_pending_render_packet_drop_count = 0
            self.demo31_tracking_result_without_render_packet_count = 0
            self.demo31_tracking_result_exact_render_packet_count = 0
            self.demo31_tracking_result_nearest_render_packet_count = 0
            self.demo31_tracking_result_without_lift_input_count = 0
            self.demo31_overlay_age_ms_samples: list[float] = []
            self.demo31_overlay_model_ms_samples: list[float] = []
            self.demo31_overlay_e2e_ms_samples: list[float] = []
            self.demo31_cotracker_publish_times_s: list[float] = []
            self.demo31_overlay_render_group_delta_samples: list[float] = []
            self.demo31_tracking_mask_age_ms_samples: list[float] = []
            self.demo31_tracking_mask_reuse_count = 0
            self.demo31_tracking_mask_selection_count = 0
            self.demo31_controller_mask_pixels_before_erode_by_camera: dict[int, int] = {}
            self.demo31_controller_mask_pixels_after_erode_by_camera: dict[int, int] = {}
            self.demo31_overlay_render_group_mismatch_count = 0
            self.demo32_trackable_masks_initialized_by_camera: set[int] = set()
            self.demo32_trackable_mask_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            self.demo32_trackable_mask_stats_by_camera: dict[int, dict[str, Any]] = {}
            self.demo32_first_trackable_mask_group_id: int | None = None
            self.demo32_first_trackable_mask_s: float | None = None
            self.demo32_first_tracking_input_publish_s: float | None = None
            self.demo31_wait_for_tracking_overlay = bool(
                self.demo31_contract.get("wait_for_tracking_overlay", DEFAULT_WAIT_FOR_TRACKING_OVERLAY)
            ) and self.demo31_cotracker_enabled
            self.demo31_tracking_overlay_warmup_skipped_render_count = 0
            self.demo31_tracking_overlay_render_blocked_count = 0
            self.demo31_tracking_overlay_first_render_group_id: int | None = None
            self.demo31_tracking_stats: dict[str, dict[int, int]] = {}

        def stop(self) -> None:
            self.stop_event.set()
            self._drain_demo31_process_status()
            if self.demo31_process_client is not None:
                self.demo31_process_client.stop(timeout_s=2.0)
                self._drain_demo31_process_status()
            self._write_demo31_pre_teardown_profile()
            super().stop()

        def _thread_specs(self) -> list[tuple[str, Callable[[], None]]]:
            specs = list(super()._thread_specs())
            if self.demo31_cotracker_enabled:
                specs.append(("demo31-tracker-render", self._tracker_driven_render_worker))
            return specs

        def _write_demo31_pre_teardown_profile(self) -> None:
            path = getattr(self.args, "demo31_top_level_profile_json_output", None)
            if path is None:
                return
            snapshot = self.demo31_snapshot()
            summary = build_empty_dual_gpu_profile_summary(self.demo31_contract)
            _merge_cotracker_process_snapshot_metrics(summary, snapshot)
            payload = {
                "contract": dict(self.demo31_contract),
                "summary": summary,
                "cotracker_process_snapshot": snapshot,
                "shared_runtime_profile": (
                    None
                    if getattr(self.args, "profile_json_output", None) is None
                    else str(getattr(self.args, "profile_json_output"))
                ),
                "runtime_note": (
                    "Pre-teardown Demo 3.1 profile written before legacy Open3D "
                    "cleanup so live profiling survives workstation teardown crashes."
                ),
                "pre_teardown_profile": True,
            }
            _write_profile(Path(path), payload)

        def _drain_demo31_process_status(self) -> list[dict[str, Any]]:
            if self.demo31_process_client is None or not hasattr(self.demo31_process_client, "drain_status_events"):
                return []
            events = self.demo31_process_client.drain_status_events()
            for event in events:
                if not isinstance(event, dict):
                    continue
                event = dict(event)
                self.demo31_process_status_events.append(event)
                if str(event.get("type")) == "error":
                    self._summary["demo31_cotracker_process_error"] = str(event.get("error", "unknown"))
                    self._summary["demo31_tracker_process_error"] = str(event.get("error", "unknown"))
                    self._summary["demo31_cotracker_process_error_stage"] = str(event.get("stage", "cotracker"))
                    self._summary["demo31_tracker_process_error_stage"] = str(event.get("stage", "tracker"))
                    self._init_profile_update(("demo31", "cotracker_process", "error"), event)
                    self._init_profile_update(("demo31", "tracker_process", "error"), event)
                    continue
                if str(event.get("type")) != "ready":
                    continue
                ready_to_receive = bool(event.get("ready_to_receive_inputs", True))
                self._summary["demo31_cotracker_process_ready"] = True
                self._summary["demo31_tracker_process_ready"] = True
                self._summary["demo31_tracker_ready_to_receive_inputs"] = ready_to_receive
                self._summary["demo31_tracker_ready_state"] = str(event.get("ready_state", "ready_to_receive_inputs"))
                self._summary["demo31_cotracker_process_init_ms"] = float(event.get("total_init_ms", 0.0) or 0.0)
                self._summary["demo31_tracker_process_init_ms"] = float(event.get("total_init_ms", 0.0) or 0.0)
                self._summary["demo31_cotracker_prewarm_backends"] = bool(event.get("prewarm_backends", False))
                self._summary["demo31_tracker_prewarm_backends"] = bool(event.get("prewarm_backends", False))
                self._summary["demo31_tracker_prewarm_mode"] = str(event.get("tracker_prewarm_mode", "unknown"))
                self._summary["demo31_tracker_query_dependent_init_pending"] = bool(
                    event.get("tracker_query_dependent_init_pending", False)
                )
                warmup_profile = event.get("warmup_profile") if isinstance(event.get("warmup_profile"), dict) else {}
                self._summary["demo31_cotracker_backend_warmup_ms"] = float(
                    warmup_profile.get("total_ms", 0.0) if isinstance(warmup_profile, dict) else 0.0
                )
                self._summary["demo31_tracker_backend_warmup_ms"] = float(
                    warmup_profile.get("total_ms", 0.0) if isinstance(warmup_profile, dict) else 0.0
                )
                self._init_profile_update(
                    ("demo31", "cotracker_process", "ready"),
                    {
                        "cuda_visible_devices": event.get("cuda_visible_devices"),
                        "prewarm_backends": bool(event.get("prewarm_backends", False)),
                        "tracker_prewarm_mode": event.get("tracker_prewarm_mode"),
                        "ready_state": event.get("ready_state", "ready_to_receive_inputs"),
                        "ready_to_receive_inputs": ready_to_receive,
                        "tracker_query_dependent_init_pending": bool(
                            event.get("tracker_query_dependent_init_pending", False)
                        ),
                        "total_init_ms": float(event.get("total_init_ms", 0.0) or 0.0),
                        "warmup_profile": warmup_profile,
                        "ready_receive_s": self._profile_rel_s(),
                    },
                )
                self._init_profile_update(("demo31", "tracker_process", "ready"), event)
            return events

        def _build_surface_anchor_snapshot(
            self,
            *,
            group_id: int,
            timestamp_s: float,
            depth_by_camera: dict[int, np.ndarray],
            intrinsics_by_camera: dict[int, np.ndarray],
            c2w_by_camera: dict[int, np.ndarray],
            mask_by_camera: dict[int, np.ndarray],
            object_mask_by_camera: dict[int, np.ndarray],
            controller_mask_by_camera: dict[int, np.ndarray],
        ) -> SurfaceAnchorIndexSnapshot:
            layers: dict[tuple[int, str], SurfaceAnchorLayer] = {}
            depth_min_m = float(getattr(self.args, "depth_min_m", 0.0))
            depth_max_m = float(getattr(self.args, "depth_max_m", 0.0))
            for camera_idx in sorted(mask_by_camera):
                idx = int(camera_idx)
                if idx not in depth_by_camera or idx not in intrinsics_by_camera or idx not in c2w_by_camera:
                    continue
                label_masks = {
                    SURFACE_ANCHOR_LABEL_OBJECT: object_mask_by_camera.get(idx),
                    SURFACE_ANCHOR_LABEL_CONTROLLER: controller_mask_by_camera.get(idx),
                    SURFACE_ANCHOR_LABEL_UNION: mask_by_camera.get(idx),
                }
                for label, mask in label_masks.items():
                    if mask is None:
                        continue
                    layers[(idx, str(label))] = _surface_anchor_layer_from_mask(
                        camera_idx=idx,
                        label=str(label),
                        mask=np.asarray(mask, dtype=bool),
                        depth_m=depth_by_camera[idx],
                        intrinsics=intrinsics_by_camera[idx],
                        c2w=c2w_by_camera[idx],
                        depth_min_m=depth_min_m,
                        depth_max_m=depth_max_m,
                    )
            return SurfaceAnchorIndexSnapshot(
                group_id=int(group_id),
                timestamp_s=float(timestamp_s),
                layers=layers,
            )

        def _cap_demo31_controller_masks(
            self,
            *,
            depth_group: Any,
            masks: dict[int, Any],
        ) -> dict[int, Any]:
            capped_masks, controller_cap_profile = demo3_runtime.cap_controller_pcd_masks(
                masks,
                camera_ids=tuple(int(item) for item in self.args.camera_ids),
                max_points_per_camera=int(
                    self.demo31_contract.get(
                        "controller_pcd_max_points_per_camera",
                        demo3_runtime.DEFAULT_CONTROLLER_PCD_MAX_POINTS_PER_CAMERA,
                    )
                ),
                seed=int(self.demo31_contract.get("cotracker_seed", demo3_runtime.DEFAULT_COTRACKER_SEED)),
            )
            if hasattr(self, "_profile_update"):
                self._profile_update(
                    int(depth_group.group_id),
                    controller_pcd_mask_cap=controller_cap_profile,
                )
            return capped_masks

        def _demo32_trackable_filter_config(self) -> TrackableMaskFilterConfig:
            return TrackableMaskFilterConfig(
                depth_min_m=float(getattr(self.args, "depth_min_m", 0.0)),
                depth_max_m=float(getattr(self.args, "depth_max_m", 0.0)),
                object_point_control=str(
                    getattr(self.args, "object_point_control", demo3_runtime.OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME)
                ),
                object_volume_voxel_m=float(
                    getattr(
                        self.args,
                        "object_volume_voxel_m",
                        demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
                    )
                ),
                object_volume_origin=str(
                    getattr(
                        self.args,
                        "object_volume_origin",
                        demo3_runtime.PHYSTWIN_VOLUME_ORIGIN_WORLD,
                    )
                ),
                object_volume_points_per_voxel=int(
                    getattr(
                        self.args,
                        "object_volume_points_per_voxel",
                        demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
                    )
                ),
                object_postprocess=str(getattr(self.args, "object_postprocess", "enhanced-pt")),
                controller_postprocess=str(getattr(self.args, "controller_postprocess", "pt-filter")),
                phystwin_radius_m=float(getattr(self.args, "phystwin_radius_m", 0.01)),
                phystwin_nb_points=int(getattr(self.args, "phystwin_nb_points", 12)),
                enhanced_component_voxel_size_m=float(getattr(self.args, "enhanced_component_voxel_size_m", 0.006)),
                enhanced_keep_near_main_gap_m=float(getattr(self.args, "enhanced_keep_near_main_gap_m", 0.035)),
                controller_trackable_max_points_per_camera=int(
                    self.demo31_contract.get(
                        "controller_trackable_max_points_per_camera",
                        DEFAULT_DEMO32_CONTROLLER_TRACKABLE_MAX_POINTS_PER_CAMERA,
                    )
                ),
                seed=int(self.demo31_contract.get("cotracker_seed", demo3_runtime.DEFAULT_COTRACKER_SEED)),
            )

        def _apply_demo32_trackable_masks(
            self,
            *,
            group_id: int,
            timestamp_s: float,
            mask_by_camera: dict[int, np.ndarray],
            object_mask_by_camera: dict[int, np.ndarray],
            controller_mask_by_camera: dict[int, np.ndarray],
            depth_by_camera: dict[int, np.ndarray],
            intrinsics_by_camera: dict[int, np.ndarray],
            c2w_by_camera: dict[int, np.ndarray],
        ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
            if str(self.demo31_contract.get("demo", "")) != "demo3.2":
                return mask_by_camera, object_mask_by_camera, controller_mask_by_camera, {}
            policy = str(
                self.demo31_contract.get(
                    "trackable_mask_build_policy",
                    DEFAULT_DEMO32_TRACKABLE_MASK_BUILD_POLICY,
                )
            )
            if policy == TRACKABLE_MASK_BUILD_POLICY_DISABLED:
                return mask_by_camera, object_mask_by_camera, controller_mask_by_camera, {
                    "trackable_mask_build_policy": policy,
                    "trackable_mask_source": "raw_semantic_masks",
                    "trackable_mask_applied": False,
                }
            config = self._demo32_trackable_filter_config()
            out_union: dict[int, np.ndarray] = {}
            out_object: dict[int, np.ndarray] = {}
            out_controller: dict[int, np.ndarray] = {}
            built_this_group: list[int] = []
            reused_cameras: list[int] = []
            skipped_cameras: list[int] = []
            for camera_idx in self.args.camera_ids:
                idx = int(camera_idx)
                if policy == TRACKABLE_MASK_BUILD_POLICY_INIT_ONLY and idx in self.demo32_trackable_mask_cache:
                    cached_object, cached_controller, cached_union = self.demo32_trackable_mask_cache[idx]
                    out_object[idx] = np.asarray(cached_object, dtype=bool)
                    out_controller[idx] = np.asarray(cached_controller, dtype=bool)
                    out_union[idx] = np.asarray(cached_union, dtype=bool)
                    reused_cameras.append(idx)
                    continue
                if (
                    idx not in object_mask_by_camera
                    or idx not in controller_mask_by_camera
                    or idx not in depth_by_camera
                    or idx not in intrinsics_by_camera
                    or idx not in c2w_by_camera
                ):
                    if idx in mask_by_camera:
                        out_object[idx] = np.zeros_like(np.asarray(object_mask_by_camera.get(idx, mask_by_camera[idx]), dtype=bool))
                        out_controller[idx] = np.zeros_like(np.asarray(controller_mask_by_camera.get(idx, mask_by_camera[idx]), dtype=bool))
                        out_union[idx] = np.zeros_like(np.asarray(mask_by_camera[idx], dtype=bool))
                    skipped_cameras.append(idx)
                    continue
                result = build_standard_filter_trackable_masks_for_camera(
                    camera_idx=idx,
                    depth_m=depth_by_camera[idx],
                    object_mask=object_mask_by_camera[idx],
                    controller_mask=controller_mask_by_camera[idx],
                    intrinsics=intrinsics_by_camera[idx],
                    c2w=c2w_by_camera[idx],
                    config=config,
                )
                out_object[idx] = result.object_mask
                out_controller[idx] = result.controller_mask
                out_union[idx] = result.union_mask
                self.demo32_trackable_mask_stats_by_camera[idx] = dict(result.stats)
                object_count = int(np.count_nonzero(result.object_mask))
                controller_count = int(np.count_nonzero(result.controller_mask))
                if object_count > 0 and controller_count > 0:
                    self.demo32_trackable_masks_initialized_by_camera.add(idx)
                    self.demo32_trackable_mask_cache[idx] = (
                        np.ascontiguousarray(result.object_mask, dtype=bool),
                        np.ascontiguousarray(result.controller_mask, dtype=bool),
                        np.ascontiguousarray(result.union_mask, dtype=bool),
                    )
                    if self.demo32_first_trackable_mask_group_id is None:
                        self.demo32_first_trackable_mask_group_id = int(group_id)
                        self.demo32_first_trackable_mask_s = float(timestamp_s)
                built_this_group.append(idx)
            summary = summarize_trackable_stats(self.demo32_trackable_mask_stats_by_camera)
            summary.update(
                {
                    "trackable_mask_build_policy": policy,
                    "trackable_query_init_strategy": str(
                        self.demo31_contract.get(
                            "trackable_query_init_strategy",
                            DEFAULT_DEMO32_TRACKABLE_QUERY_INIT_STRATEGY,
                        )
                    ),
                    "trackable_mask_applied": True,
                    "trackable_mask_build_stage": "first_valid_tracking_input",
                    "first_trackable_mask_group_id": self.demo32_first_trackable_mask_group_id,
                    "first_trackable_mask_s": self.demo32_first_trackable_mask_s,
                    "trackable_mask_built_cameras": built_this_group,
                    "trackable_mask_reused_cameras": reused_cameras,
                    "trackable_mask_skipped_cameras": skipped_cameras,
                    "trackable_mask_initialized_cameras": sorted(self.demo32_trackable_masks_initialized_by_camera),
                }
            )
            return out_union, out_object, out_controller, summary

        def _publish_demo31_tracking_input(
            self,
            *,
            depth_group: Any,
            masks: dict[int, Any],
            publish_hook: str,
        ) -> None:
            now_s = time.perf_counter()
            rgb_by_camera: dict[int, np.ndarray] = {}
            mask_by_camera: dict[int, np.ndarray] = {}
            object_mask_by_camera: dict[int, np.ndarray] = {}
            controller_mask_by_camera: dict[int, np.ndarray] = {}
            depth_by_camera: dict[int, np.ndarray] = {}
            intrinsics_by_camera: dict[int, np.ndarray] = {}
            c2w_by_camera: dict[int, np.ndarray] = {}
            controller_mask_erode_px = int(self.demo31_contract.get("controller_mask_erode_px", 0) or 0)
            controller_mask_pixels_before_erode_by_camera: dict[int, int] = {}
            controller_mask_pixels_after_erode_by_camera: dict[int, int] = {}
            mask_selection = (
                self.stage_join_buffer.selection_for_group(int(depth_group.group_id))
                if hasattr(self.stage_join_buffer, "selection_for_group")
                else None
            )
            mask_source_group_id = int(mask_selection.get("source_group_id", depth_group.group_id)) if mask_selection else int(depth_group.group_id)
            mask_age_ms = float(mask_selection.get("age_ms", 0.0)) if mask_selection else 0.0
            mask_reused = bool(mask_selection.get("reused", False)) if mask_selection else False
            for camera_idx in self.args.camera_ids:
                idx = int(camera_idx)
                if idx not in masks or idx not in depth_group.depths:
                    continue
                mask_packet = masks[idx]
                rgb_by_camera[idx] = np.ascontiguousarray(np.asarray(mask_packet.color_bgr)[..., ::-1])
                controller_mask_pixels_before_erode_by_camera[idx] = int(
                    np.count_nonzero(np.asarray(mask_packet.controller_mask, dtype=bool))
                )
                union_mask, object_mask, controller_mask = _phystwin_union_tracking_masks(
                    mask_packet,
                    controller_mask_erode_px=controller_mask_erode_px,
                )
                mask_by_camera[idx] = union_mask
                object_mask_by_camera[idx] = object_mask
                controller_mask_by_camera[idx] = controller_mask
                controller_mask_pixels_after_erode_by_camera[idx] = int(np.count_nonzero(controller_mask))
                depth_by_camera[idx] = np.asarray(depth_group.depths[idx].depth_m, dtype=np.float32)
                if getattr(self, "_stream_metadata", None) and idx < len(self._stream_metadata):
                    intrinsics_by_camera[idx] = np.asarray(
                        self._stream_metadata[idx]["K_color"],
                        dtype=np.float32,
                    ).reshape(3, 3)
                if idx in getattr(self, "_c2w_by_camera", {}):
                    c2w_by_camera[idx] = np.asarray(self._c2w_by_camera[idx], dtype=np.float32).reshape(4, 4)
            trackable_mask_profile: dict[str, Any] = {}
            mask_by_camera, object_mask_by_camera, controller_mask_by_camera, trackable_mask_profile = (
                self._apply_demo32_trackable_masks(
                    group_id=int(depth_group.group_id),
                    timestamp_s=now_s,
                    mask_by_camera=mask_by_camera,
                    object_mask_by_camera=object_mask_by_camera,
                    controller_mask_by_camera=controller_mask_by_camera,
                    depth_by_camera=depth_by_camera,
                    intrinsics_by_camera=intrinsics_by_camera,
                    c2w_by_camera=c2w_by_camera,
                )
            )
            self.demo31_controller_mask_pixels_before_erode_by_camera = dict(
                controller_mask_pixels_before_erode_by_camera
            )
            self.demo31_controller_mask_pixels_after_erode_by_camera = dict(
                controller_mask_pixels_after_erode_by_camera
            )
            if mask_by_camera:
                self.demo31_mask_cache.publish(
                    group_id=int(depth_group.group_id),
                    timestamp_s=now_s,
                    mask_by_camera=mask_by_camera,
                )
            profile_payload: dict[str, Any] = {
                "publish_hook": str(publish_hook),
                "published": False,
                "skipped": False,
                "process_enabled": self.demo31_process_client is not None,
                "camera_count": int(len(mask_by_camera)),
                "mask_source_group_id": int(mask_source_group_id),
                "mask_age_ms": float(mask_age_ms),
                "mask_reused": bool(mask_reused),
                "surface_anchor_cache_published": False,
                "trackable_mask": trackable_mask_profile,
                "controller_mask_erode_px": int(controller_mask_erode_px),
                "controller_mask_pixels_before_erode_by_camera": dict(controller_mask_pixels_before_erode_by_camera),
                "controller_mask_pixels_after_erode_by_camera": dict(controller_mask_pixels_after_erode_by_camera),
            }
            if self.demo31_process_client is None or not rgb_by_camera or not mask_by_camera:
                if hasattr(self, "_profile_update"):
                    self._profile_update(int(depth_group.group_id), demo31_tracking_input=profile_payload)
                return
            if not should_publish_tracking_input(
                now_s=now_s,
                last_publish_s=self.demo31_last_tracking_input_s,
                target_fps=float(self.demo31_contract["cotracker_input_fps"]),
            ):
                self.demo31_tracking_input_skip_count += 1
                profile_payload["skipped"] = True
                if hasattr(self, "_profile_update"):
                    self._profile_update(int(depth_group.group_id), demo31_tracking_input=profile_payload)
                return

            frame_idx = int(max(depth_group.per_camera_frame_seq.values()) if depth_group.per_camera_frame_seq else depth_group.group_id)
            self.demo31_lift_input_cache.publish(
                group_id=int(depth_group.group_id),
                timestamp_s=now_s,
                depth_by_camera=depth_by_camera,
                intrinsics_by_camera=intrinsics_by_camera,
                c2w_by_camera=c2w_by_camera,
                mask_by_camera=mask_by_camera,
                object_mask_by_camera=object_mask_by_camera,
                controller_mask_by_camera=controller_mask_by_camera,
            )
            self.demo31_surface_anchor_cache.publish(
                self._build_surface_anchor_snapshot(
                    group_id=int(depth_group.group_id),
                    timestamp_s=now_s,
                    depth_by_camera=depth_by_camera,
                    intrinsics_by_camera=intrinsics_by_camera,
                    c2w_by_camera=c2w_by_camera,
                    mask_by_camera=mask_by_camera,
                    object_mask_by_camera=object_mask_by_camera,
                    controller_mask_by_camera=controller_mask_by_camera,
                )
            )
            replaced_count = self.demo31_process_client.publish_input(
                TrackingInputLitePacket(
                    group_id=int(depth_group.group_id),
                    frame_idx=frame_idx,
                    timestamp_s=now_s,
                    rgb_by_camera=rgb_by_camera,
                    mask_by_camera=mask_by_camera,
                    object_mask_by_camera=object_mask_by_camera,
                    controller_mask_by_camera=controller_mask_by_camera,
                    mask_source_group_id=mask_source_group_id,
                    mask_age_ms=mask_age_ms,
                    mask_reused=mask_reused,
                )
            )
            self.demo31_tracking_input_queue_replace_count += int(replaced_count or 0)
            self.demo31_last_tracking_input_s = now_s
            if self.demo32_first_tracking_input_publish_s is None and str(self.demo31_contract.get("demo", "")) == "demo3.2":
                self.demo32_first_tracking_input_publish_s = float(now_s)
            self.demo31_tracking_input_publish_times_s.append(float(now_s))
            profile_payload.update(
                {
                    "published": True,
                    "queue_replaced_count": int(replaced_count or 0),
                    "frame_idx": int(frame_idx),
                    "surface_anchor_cache_published": True,
                    "lift_input_cache_published": True,
                    "first_tracking_input_publish_s": self.demo32_first_tracking_input_publish_s,
                }
            )
            if hasattr(self, "_profile_update"):
                self._profile_update(int(depth_group.group_id), demo31_tracking_input=profile_payload)

        def _build_raw_fused_packet(self, *, depth_group: Any, masks: dict[int, Any], ray_cache: dict[int, Any], rng: np.random.Generator):
            capped_masks = self._cap_demo31_controller_masks(depth_group=depth_group, masks=masks)
            self._publish_demo31_tracking_input(
                depth_group=depth_group,
                masks=capped_masks,
                publish_hook="raw_fused_async",
            )
            return super()._build_raw_fused_packet(depth_group=depth_group, masks=capped_masks, ray_cache=ray_cache, rng=rng)

        def _build_fused_packet(self, *, depth_group: Any, masks: dict[int, Any], ray_cache: dict[int, Any], rng: np.random.Generator):
            capped_masks = self._cap_demo31_controller_masks(depth_group=depth_group, masks=masks)
            self._publish_demo31_tracking_input(
                depth_group=depth_group,
                masks=capped_masks,
                publish_hook="fused_packet",
            )
            return super()._build_fused_packet(depth_group=depth_group, masks=capped_masks, ray_cache=ray_cache, rng=rng)

        def _publish_render_packet(self, packet: Any) -> None:
            self._remember_pending_render_packet(packet)
            if not self.demo31_cotracker_enabled:
                super()._publish_render_packet(packet)
                return
            tracking_overlay_warmup_blocked = self.demo31_tracking_overlay_first_render_group_id is None
            self._profile_update(
                int(packet.group_id),
                demo31_tracking_overlay={
                    "overlay_available": False,
                    "overlay_points": 0,
                    "overlay_points_by_camera": {},
                    "overlay_ms": 0.0,
                    "overlay_group_id": None,
                    "incoming_render_group_id": int(packet.group_id),
                    "render_group_id": int(packet.group_id),
                    "overlay_render_group_delta": None,
                    "render_driver": "cotracker_child_output",
                    "render_trigger": "pcd_packet_cached_until_tracker",
                    "rendered_on_new_cotracker_result": False,
                    "tracking_overlay_render_blocked": True,
                    "tracking_overlay_warmup_blocked": bool(tracking_overlay_warmup_blocked),
                    "tracking_overlay_required_before_first_render": True,
                    "tracking_overlay_required_for_render": True,
                    "render_requires_new_cotracker_result": True,
                    "render_reuses_cached_cotracker_result": False,
                    "overlay_lift_cache_hit": False,
                    "tracking_result_has_matching_render_packet": False,
                    "tracking_result_used_render_packet": False,
                    "tracking_result_used_nearest_render_packet": False,
                    "tracking_render_packet_match_mode": "pending",
                    "tracking_render_packet_match_policy": TRACKING_RENDER_PACKET_MATCH_POLICY,
                    "tracking_render_packet_group_id": None,
                    "tracking_render_packet_group_delta": None,
                    "tracking_nearest_render_packet_abs_delta": None,
                    "tracking_pending_render_packet_count_before_match": None,
                    "tracking_pending_render_packet_max_groups": int(self.demo31_pending_render_packet_max_groups),
                    "tracking_pending_render_packet_had_lift_candidate": False,
                    "cotracker_model_ms": None,
                    "cotracker_e2e_ms": None,
                    "render_waited_for_cotracker": True,
                    "render_waited_for_fresh_cotracker_result": True,
                    "cross_gpu_cuda_tensor_transfer": False,
                },
            )

        def _tracker_driven_render_worker(self) -> None:
            while not self.stop_event.is_set():
                handled = self._publish_next_tracker_driven_render_once(now_s=time.perf_counter())
                if not handled:
                    time.sleep(0.001)

        def _publish_next_tracker_driven_render_once(self, *, now_s: float) -> bool:
            overlay = self._take_fresh_tracking_result(now_s=now_s)
            if overlay is None:
                return False
            self._publish_tracker_driven_render(overlay, overlay_start_s=now_s)
            return True

        def _take_render_packet_for_tracking_result(
            self,
            group_id: int,
            *,
            cached_group_ids: set[int] | None = None,
            require_exact: bool = False,
        ) -> tuple[Any | None, dict[str, Any]]:
            requested_group_id = int(group_id)
            lift_group_ids = (
                set(int(group_id) for group_id in cached_group_ids)
                if cached_group_ids is not None
                else self.demo31_lift_input_cache.cached_group_ids()
            )
            with self.demo31_pending_render_lock:
                pending_ids_before = sorted(int(item) for item in self.demo31_pending_render_packets)
                render_packet = self.demo31_pending_render_packets.pop(requested_group_id, None)
                if render_packet is not None:
                    self.demo31_tracking_result_exact_render_packet_count += 1
                    return render_packet, {
                        "tracking_render_packet_match_mode": "exact",
                        "tracking_result_has_matching_render_packet": True,
                        "tracking_result_used_render_packet": True,
                        "tracking_result_used_nearest_render_packet": False,
                        "tracking_render_packet_group_id": int(render_packet.group_id),
                        "tracking_render_packet_group_delta": 0,
                        "tracking_nearest_render_packet_abs_delta": 0,
                        "tracking_pending_render_packet_count_before_match": int(len(pending_ids_before)),
                        "tracking_pending_render_packet_had_lift_candidate": requested_group_id in lift_group_ids,
                    }
                if bool(require_exact):
                    self.demo31_tracking_result_without_render_packet_count += 1
                    return None, {
                        "tracking_render_packet_match_mode": "missing-exact",
                        "tracking_result_has_matching_render_packet": False,
                        "tracking_result_used_render_packet": False,
                        "tracking_result_used_nearest_render_packet": False,
                        "tracking_render_packet_group_id": None,
                        "tracking_render_packet_group_delta": None,
                        "tracking_nearest_render_packet_abs_delta": None,
                        "tracking_pending_render_packet_count_before_match": int(len(pending_ids_before)),
                        "tracking_pending_render_packet_had_lift_candidate": requested_group_id in lift_group_ids,
                    }
                if not pending_ids_before:
                    self.demo31_tracking_result_without_render_packet_count += 1
                    return None, {
                        "tracking_render_packet_match_mode": "missing",
                        "tracking_result_has_matching_render_packet": False,
                        "tracking_result_used_render_packet": False,
                        "tracking_result_used_nearest_render_packet": False,
                        "tracking_render_packet_group_id": None,
                        "tracking_render_packet_group_delta": None,
                        "tracking_nearest_render_packet_abs_delta": None,
                        "tracking_pending_render_packet_count_before_match": 0,
                        "tracking_pending_render_packet_had_lift_candidate": False,
                    }
                lift_candidate_ids = [group for group in pending_ids_before if group in lift_group_ids]
                candidate_ids = lift_candidate_ids or pending_ids_before
                nearest_group_id = min(
                    candidate_ids,
                    key=lambda candidate: (
                        abs(int(candidate) - requested_group_id),
                        0 if int(candidate) >= requested_group_id else 1,
                        int(candidate),
                    ),
                )
                render_packet = self.demo31_pending_render_packets.pop(nearest_group_id, None)
                if render_packet is None:
                    self.demo31_tracking_result_without_render_packet_count += 1
                    return None, {
                        "tracking_render_packet_match_mode": "missing",
                        "tracking_result_has_matching_render_packet": False,
                        "tracking_result_used_render_packet": False,
                        "tracking_result_used_nearest_render_packet": False,
                        "tracking_render_packet_group_id": None,
                        "tracking_render_packet_group_delta": None,
                        "tracking_nearest_render_packet_abs_delta": None,
                        "tracking_pending_render_packet_count_before_match": int(len(pending_ids_before)),
                        "tracking_pending_render_packet_had_lift_candidate": bool(lift_candidate_ids),
                    }
                delta = int(render_packet.group_id) - requested_group_id
                self.demo31_tracking_result_nearest_render_packet_count += 1
                return render_packet, {
                    "tracking_render_packet_match_mode": "nearest",
                    "tracking_result_has_matching_render_packet": False,
                    "tracking_result_used_render_packet": True,
                    "tracking_result_used_nearest_render_packet": True,
                    "tracking_render_packet_group_id": int(render_packet.group_id),
                    "tracking_render_packet_group_delta": int(delta),
                    "tracking_nearest_render_packet_abs_delta": int(abs(delta)),
                    "tracking_pending_render_packet_count_before_match": int(len(pending_ids_before)),
                    "tracking_pending_render_packet_had_lift_candidate": int(render_packet.group_id) in lift_group_ids,
                }

        def _publish_tracker_driven_render(self, overlay: TrackingResultLitePacket, *, overlay_start_s: float) -> None:
            render_requires_new_tracker = bool(self.demo31_cotracker_enabled)
            overlay_points = np.empty((0, 3), dtype=np.float32)
            overlay_colors = np.empty((0, 3), dtype=np.uint8)
            overlay_input_points_by_camera: dict[int, int] = {}
            overlay_lifted_points_by_camera: dict[int, int] = {}
            overlay_points_by_camera: dict[int, int] = {}
            overlay_centroid_by_camera: dict[int, list[float] | None] = {}
            overlay_centroid_before_bbox_by_camera: dict[int, list[float] | None] = {}
            overlay_bbox_input_points_by_camera: dict[int, int] = {}
            overlay_bbox_kept_points_by_camera: dict[int, int] = {}
            overlay_bbox_rejected_by_camera: dict[int, int] = {}
            tracking_control_points_by_camera: dict[int, int] = {}
            tracking_control_point_count = 0
            tracking_control_marker_points = 0
            overlay_track_points = np.empty((0, 3), dtype=np.float32)
            overlay_control_point_centroid: list[float] | None = None
            tracker_visualization_mode = str(
                getattr(
                    self.args,
                    "tracker_visualization_mode",
                    self.demo31_contract.get("tracker_visualization_mode", TRACKER_VISUALIZATION_MODE_LEGACY_3D_LIFT),
                )
            )
            surface_marker_mode = tracker_visualization_mode == TRACKER_VISUALIZATION_MODE_SURFACE_MARKERS
            legacy_lift_mode = tracker_visualization_mode == TRACKER_VISUALIZATION_MODE_LEGACY_3D_LIFT
            all_tracks_lift_mode = tracker_visualization_mode == TRACKER_VISUALIZATION_MODE_ALL_TRACKS_3D_LIFT
            direct_depth_lift_mode = legacy_lift_mode or all_tracks_lift_mode
            inert_visualization_mode = tracker_visualization_mode in {
                TRACKER_VISUALIZATION_MODE_NONE,
                TRACKER_VISUALIZATION_MODE_2D_DEBUG,
            }
            control_markers_enabled = bool(
                getattr(
                    self.args,
                    "overlay_control_point_markers",
                    bool(self.demo31_contract.get("tracking_control_point_markers", surface_marker_mode)),
                )
                )
            if surface_marker_mode:
                control_markers_enabled = True
            if all_tracks_lift_mode:
                control_markers_enabled = True
            if inert_visualization_mode:
                control_markers_enabled = False
            render_raw_tracks = bool(
                getattr(
                    self.args,
                    "overlay_render_raw_track_points",
                    bool(
                        self.demo31_contract.get(
                            "overlay_render_raw_track_points",
                            legacy_lift_mode and not control_markers_enabled,
                        )
                    ),
                )
                and legacy_lift_mode
            )
            control_point_count_requested = int(
                getattr(
                    self.args,
                    "overlay_control_point_count",
                    int(
                        self.demo31_contract.get(
                            "tracking_control_point_count_requested",
                            DEFAULT_OVERLAY_CONTROL_POINT_COUNT,
                        )
                    ),
                )
            )
            tracker_control_points_per_camera = int(
                getattr(
                    self.args,
                    "tracker_control_points_per_camera",
                    int(self.demo31_contract.get("tracker_control_points_per_camera", DEFAULT_TRACKER_CONTROL_POINTS_PER_CAMERA)),
                )
            )
            tracker_control_point_selection = str(
                getattr(
                    self.args,
                    "tracker_control_point_selection",
                    self.demo31_contract.get("tracker_control_point_selection", DEFAULT_TRACKER_CONTROL_POINT_SELECTION),
                )
            )
            control_point_radius_m = float(
                getattr(
                    self.args,
                    "overlay_control_point_radius_m",
                    float(
                        self.demo31_contract.get(
                            "tracking_control_point_radius_m",
                            DEFAULT_OVERLAY_CONTROL_POINT_RADIUS_M,
                        )
                    ),
                )
            )
            if surface_marker_mode:
                control_point_radius_m = float(
                    getattr(
                        self.args,
                        "tracker_3d_marker_radius_m",
                        float(self.demo31_contract.get("tracker_3d_marker_radius_m", DEFAULT_TRACKER_3D_MARKER_RADIUS_M)),
                    )
                )
                control_point_count_requested = max(0, tracker_control_points_per_camera) * len(
                    tuple(getattr(self.args, "camera_ids", ()))
                )
            if all_tracks_lift_mode:
                control_point_radius_m = float(
                    getattr(
                        self.args,
                        "tracker_3d_marker_radius_m",
                        float(self.demo31_contract.get("tracker_3d_marker_radius_m", DEFAULT_TRACKER_3D_MARKER_RADIUS_M)),
                    )
                )
                control_point_count_requested = 0
            tracker_snap_radius_px = float(
                getattr(
                    self.args,
                    "tracker_3d_snap_radius_px",
                    float(self.demo31_contract.get("tracker_3d_snap_radius_px", DEFAULT_TRACKER_3D_SNAP_RADIUS_PX)),
                )
            )
            tracker_surface_anchor_cache_hit = False
            tracker_surface_anchor_group_id: int | None = None
            tracker_marker_accepted_by_camera: dict[int, int] = {}
            tracker_marker_rejected_by_camera: dict[int, int] = {}
            tracker_marker_pixel_error_median_by_camera: dict[int, float] = {}
            tracker_marker_pixel_error_p95_by_camera: dict[int, float] = {}
            tracker_marker_layer_by_camera: dict[int, str] = {}
            bbox_filter_enabled = False
            overlay_lift_cache_hit = False
            overlay_group_id: int | None = None
            overlay_render_group_delta: int | None = None
            render_packet = None
            overlay_group_id = int(overlay.group_id)
            render_packet, render_match_profile = self._take_render_packet_for_tracking_result(
                overlay_group_id,
                cached_group_ids=(
                    self.demo31_surface_anchor_cache.cached_group_ids()
                    if surface_marker_mode
                    else None
                ),
                require_exact=bool(surface_marker_mode),
            )
            overlay_render_group_delta = render_match_profile["tracking_render_packet_group_delta"]
            if overlay_render_group_delta is not None:
                self.demo31_overlay_render_group_delta_samples.append(float(abs(overlay_render_group_delta)))
                if overlay_render_group_delta != 0:
                    self.demo31_overlay_render_group_mismatch_count += 1
            lift_inputs = None
            surface_snapshot = None
            if surface_marker_mode and render_packet is not None and int(render_packet.group_id) == int(overlay_group_id):
                surface_snapshot = self.demo31_surface_anchor_cache.get(int(overlay_group_id))
                tracker_surface_anchor_cache_hit = surface_snapshot is not None
                tracker_surface_anchor_group_id = None if surface_snapshot is None else int(surface_snapshot.group_id)
            elif direct_depth_lift_mode and render_packet is not None:
                lift_inputs = self.demo31_lift_input_cache.get(int(render_packet.group_id))
            if direct_depth_lift_mode and render_packet is not None and lift_inputs is None:
                self.demo31_tracking_result_without_lift_input_count += 1
            if render_packet is not None:
                color_by_camera = bool(getattr(self.args, "overlay_debug_color_by_camera", False))
                lift_mask_scope = str(getattr(self.args, "overlay_display_scope", demo3_runtime.DEFAULT_OVERLAY_DISPLAY_SCOPE))
                bbox_filter_enabled = bool(
                    getattr(
                        self.args,
                        "overlay_reject_outside_semantic_bbox",
                        DEFAULT_OVERLAY_REJECT_OUTSIDE_SEMANTIC_BBOX,
                    )
                ) and not all_tracks_lift_mode
                bbox_margin_m = float(
                    getattr(
                        self.args,
                        "overlay_max_distance_from_controller_m",
                        DEFAULT_OVERLAY_MAX_DISTANCE_FROM_CONTROLLER_M,
                    )
                )
                bbox_reference_points = _semantic_bbox_reference_points(
                    scope=lift_mask_scope,
                    render_packet=render_packet,
                )
                marker_parts: list[np.ndarray] = []
                marker_color_parts: list[np.ndarray] = []
                if surface_marker_mode and surface_snapshot is not None:
                    surface_label = _overlay_scope_to_surface_label(lift_mask_scope)
                    snapped_point_chunks: list[np.ndarray] = []
                    snapped_color_chunks: list[np.ndarray] = []
                    snapped_camera_id_chunks: list[np.ndarray] = []
                    for camera_idx, tracks_yx in overlay.camera_tracks_yx.items():
                        idx = int(camera_idx)
                        tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
                        visibility = np.asarray(
                            overlay.camera_visibility.get(idx, np.zeros((len(tracks),), dtype=np.float32)),
                            dtype=np.float32,
                        ).reshape(-1)
                        overlay_input_points_by_camera[idx] = int(len(tracks))
                        layer = surface_snapshot.layers.get((idx, surface_label))
                        tracker_marker_layer_by_camera[idx] = surface_label
                        if layer is None:
                            visible_count = int(np.count_nonzero(visibility > 0.0))
                            tracker_marker_accepted_by_camera[idx] = 0
                            tracker_marker_rejected_by_camera[idx] = visible_count
                            overlay_points_by_camera[idx] = 0
                            continue
                        snapped = _snap_tracks_to_surface_result(
                            tracks_yx=tracks,
                            visibility=visibility,
                            surface_layer=layer,
                            radius_px=tracker_snap_radius_px,
                            max_points=tracker_control_points_per_camera,
                            selection=tracker_control_point_selection,
                        )
                        points = snapped.points_world.astype(np.float32, copy=False)
                        tracker_marker_accepted_by_camera[idx] = int(snapped.accepted)
                        tracker_marker_rejected_by_camera[idx] = int(snapped.rejected)
                        if len(snapped.pixel_errors):
                            tracker_marker_pixel_error_median_by_camera[idx] = float(np.median(snapped.pixel_errors))
                            tracker_marker_pixel_error_p95_by_camera[idx] = float(np.percentile(snapped.pixel_errors, 95))
                        else:
                            tracker_marker_pixel_error_median_by_camera[idx] = 0.0
                            tracker_marker_pixel_error_p95_by_camera[idx] = 0.0
                        overlay_lifted_points_by_camera[idx] = int(len(points))
                        overlay_centroid_before_bbox_by_camera[idx] = _point_centroid(points)
                        overlay_bbox_input_points_by_camera[idx] = int(len(points))
                        if bbox_filter_enabled and len(points):
                            bbox_keep = _semantic_bbox_keep_mask(points, bbox_reference_points, margin_m=bbox_margin_m)
                            kept = int(np.count_nonzero(bbox_keep))
                            overlay_bbox_kept_points_by_camera[idx] = kept
                            overlay_bbox_rejected_by_camera[idx] = int(len(points)) - kept
                            points = points[bbox_keep]
                        else:
                            overlay_bbox_kept_points_by_camera[idx] = int(len(points))
                            overlay_bbox_rejected_by_camera[idx] = 0
                        overlay_points_by_camera[idx] = int(len(points))
                        overlay_centroid_by_camera[idx] = _point_centroid(points)
                        if len(points) == 0:
                            continue
                        snapped_point_chunks.append(points)
                        snapped_camera_id_chunks.append(np.full((len(points),), idx, dtype=np.int32))
                        color = _surface_marker_color(surface_label, idx, color_by_camera=color_by_camera)
                        snapped_color_chunks.append(_overlay_color_array(len(points), color))
                    if snapped_point_chunks:
                        overlay_track_points = np.concatenate(snapped_point_chunks, axis=0).astype(np.float32)
                        overlay_track_colors = np.concatenate(snapped_color_chunks, axis=0).astype(np.uint8)
                        overlay_track_camera_ids = np.concatenate(snapped_camera_id_chunks, axis=0)
                        control_points = overlay_track_points
                        control_camera_ids = overlay_track_camera_ids
                        tracking_control_point_count = int(len(control_points))
                        overlay_control_point_centroid = _point_centroid(control_points)
                        for camera_idx in sorted(set(int(item) for item in control_camera_ids.tolist())):
                            tracking_control_points_by_camera[int(camera_idx)] = int(
                                np.count_nonzero(control_camera_ids == int(camera_idx))
                            )
                        control_marker_points, control_marker_colors = _control_point_marker_cloud(
                            control_points,
                            overlay_track_colors,
                            radius_m=control_point_radius_m,
                        )
                        tracking_control_marker_points = int(len(control_marker_points))
                        if len(control_marker_points):
                            marker_parts.append(control_marker_points)
                            marker_color_parts.append(control_marker_colors)
                elif direct_depth_lift_mode and lift_inputs is not None:
                    overlay_lift_cache_hit = True
                    lifted_points = []
                    lifted_colors = []
                    lifted_camera_id_chunks = []
                    for camera_idx, tracks_yx in overlay.camera_tracks_yx.items():
                        idx = int(camera_idx)
                        if (
                            idx not in lift_inputs.depth_by_camera
                            or idx not in lift_inputs.intrinsics_by_camera
                            or idx not in lift_inputs.c2w_by_camera
                        ):
                            continue
                        overlay_input_points_by_camera[idx] = int(len(np.asarray(tracks_yx).reshape(-1, 2)))
                        lift_mask = (
                            None
                            if all_tracks_lift_mode
                            else _lift_mask_for_overlay_scope(
                                scope=lift_mask_scope,
                                camera_idx=idx,
                                lift_inputs=lift_inputs,
                            )
                        )
                        lifted = lift_tracks_yx_to_world(
                            tracks_yx=tracks_yx,
                            visibility=overlay.camera_visibility[idx],
                            depth=lift_inputs.depth_by_camera[idx],
                            intrinsics=lift_inputs.intrinsics_by_camera[idx],
                            c2w=lift_inputs.c2w_by_camera[idx],
                            depth_scale_m_per_unit=1.0,
                            mask=lift_mask,
                        )
                        if all_tracks_lift_mode:
                            tracker_marker_layer_by_camera[idx] = "all-tracks"
                            tracker_marker_accepted_by_camera[idx] = int(len(lifted.points_world))
                            tracker_marker_rejected_by_camera[idx] = int(
                                overlay_input_points_by_camera[idx] - len(lifted.points_world)
                            )
                        if lifted.points_world.size:
                            points = lifted.points_world.astype(np.float32, copy=False)
                            overlay_lifted_points_by_camera[idx] = int(len(points))
                            overlay_centroid_before_bbox_by_camera[idx] = _point_centroid(points)
                            overlay_bbox_input_points_by_camera[idx] = int(len(points))
                            if bbox_filter_enabled:
                                bbox_keep = _semantic_bbox_keep_mask(points, bbox_reference_points, margin_m=bbox_margin_m)
                                kept = int(np.count_nonzero(bbox_keep))
                                overlay_bbox_kept_points_by_camera[idx] = kept
                                overlay_bbox_rejected_by_camera[idx] = int(len(points)) - kept
                                points = points[bbox_keep]
                            else:
                                overlay_bbox_kept_points_by_camera[idx] = int(len(points))
                                overlay_bbox_rejected_by_camera[idx] = 0
                            if len(points) == 0:
                                overlay_points_by_camera[idx] = 0
                                overlay_centroid_by_camera[idx] = None
                                continue
                            lifted_points.append(points)
                            lifted_camera_id_chunks.append(np.full((len(points),), idx, dtype=np.int32))
                            overlay_points_by_camera[idx] = int(len(points))
                            overlay_centroid_by_camera[idx] = _point_centroid(points)
                            color = (
                                _overlay_debug_color_rgb(idx)
                                if color_by_camera
                                else tuple(int(v) for v in demo3_runtime.OVERLAY_COLOR_RGB.tolist())
                            )
                            lifted_colors.append(_overlay_color_array(len(points), color))
                    if lifted_points:
                        overlay_track_points = np.concatenate(lifted_points, axis=0).astype(np.float32)
                        overlay_track_colors = np.concatenate(lifted_colors, axis=0).astype(np.uint8)
                        overlay_track_camera_ids = np.concatenate(lifted_camera_id_chunks, axis=0)
                        if all_tracks_lift_mode:
                            control_points = overlay_track_points
                            control_camera_ids = overlay_track_camera_ids
                            tracking_control_point_count = int(len(control_points))
                            overlay_control_point_centroid = _point_centroid(control_points)
                            for camera_idx in sorted(set(int(item) for item in control_camera_ids.tolist())):
                                tracking_control_points_by_camera[int(camera_idx)] = int(
                                    np.count_nonzero(control_camera_ids == int(camera_idx))
                                )
                            if bool(getattr(self.args, "overlay_debug_color_by_camera", False)):
                                control_colors = np.asarray(
                                    [_overlay_debug_color_rgb(int(camera_idx)) for camera_idx in control_camera_ids],
                                    dtype=np.uint8,
                                )
                            else:
                                control_colors = _overlay_color_array(
                                    len(control_points),
                                    DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB,
                                )
                            control_marker_points, control_marker_colors = _control_point_marker_cloud(
                                control_points,
                                control_colors,
                                radius_m=control_point_radius_m,
                            )
                            tracking_control_marker_points = int(len(control_marker_points))
                            if len(control_marker_points):
                                marker_parts.append(control_marker_points)
                                marker_color_parts.append(control_marker_colors)
                        elif render_raw_tracks:
                            marker_parts.append(overlay_track_points)
                            marker_color_parts.append(overlay_track_colors)
                        if (not all_tracks_lift_mode) and control_markers_enabled:
                            control_indices = _farthest_point_sample_indices(
                                overlay_track_points,
                                control_point_count_requested,
                            )
                            control_points = overlay_track_points[control_indices]
                            control_camera_ids = overlay_track_camera_ids[control_indices]
                            tracking_control_point_count = int(len(control_points))
                            overlay_control_point_centroid = _point_centroid(control_points)
                            for camera_idx in sorted(set(int(item) for item in control_camera_ids.tolist())):
                                tracking_control_points_by_camera[int(camera_idx)] = int(
                                    np.count_nonzero(control_camera_ids == int(camera_idx))
                                )
                            if bool(getattr(self.args, "overlay_debug_color_by_camera", False)):
                                control_colors = np.asarray(
                                    [_overlay_debug_color_rgb(int(camera_idx)) for camera_idx in control_camera_ids],
                                    dtype=np.uint8,
                                )
                            else:
                                control_colors = _overlay_color_array(
                                    len(control_points),
                                    DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB,
                                )
                            control_marker_points, control_marker_colors = _control_point_marker_cloud(
                                control_points,
                                control_colors,
                                radius_m=control_point_radius_m,
                            )
                            tracking_control_marker_points = int(len(control_marker_points))
                            if len(control_marker_points):
                                marker_parts.append(control_marker_points)
                                marker_color_parts.append(control_marker_colors)
                if marker_parts:
                    overlay_points = np.concatenate(marker_parts, axis=0).astype(np.float32)
                    overlay_colors = np.concatenate(marker_color_parts, axis=0).astype(np.uint8)
                if len(overlay_points) or inert_visualization_mode:
                    render_replace_kwargs: dict[str, Any] = {
                        "controller_points_m": np.concatenate(
                            [render_packet.controller_points_m, overlay_points],
                            axis=0,
                        ),
                        "controller_colors_rgb": np.concatenate(
                            [render_packet.controller_colors_rgb, overlay_colors],
                            axis=0,
                        ),
                    }
                    if hasattr(render_packet, "tracker_model_ms"):
                        render_replace_kwargs.update(
                            {
                                "tracker_backend": str(
                                    getattr(
                                        overlay,
                                        "tracker_backend",
                                        self.demo31_contract.get("tracker_backend", TRACKER_BACKEND_COTRACKER3),
                                    )
                                ),
                                "tracker_update_mode": str(overlay.cotracker_update_mode),
                                "tracker_batch_size": int(overlay.cotracker_batch_size),
                                "tracker_model_ms": float(overlay.model_ms),
                                "tracker_e2e_ms": float(overlay.e2e_ms),
                                "tracker_publish_to_render_ms": float(
                                    (overlay_start_s - float(overlay.publish_timestamp_s)) * 1000.0
                                ),
                                "tracker_source_to_render_ms": float(
                                    (overlay_start_s - float(overlay.source_timestamp_s)) * 1000.0
                                ),
                                "tracker_overlay_group_id": int(overlay.group_id),
                            }
                        )
                    render_packet = replace(
                        render_packet,
                        **render_replace_kwargs,
                    )
            tracking_overlay_render_blocked = bool(
                render_requires_new_tracker and len(overlay_points) == 0 and not inert_visualization_mode
            )
            tracking_overlay_warmup_blocked = bool(
                tracking_overlay_render_blocked and self.demo31_tracking_overlay_first_render_group_id is None
            )
            overlay_ms = float((time.perf_counter() - overlay_start_s) * 1000.0)
            profile_group_id = int(overlay_group_id if render_packet is None else render_packet.group_id)
            self._profile_update(
                profile_group_id,
                demo31_tracking_overlay={
                    "overlay_available": True,
                    "overlay_points": int(len(overlay_points)),
                    "overlay_track_points": int(len(overlay_track_points)),
                    "overlay_color_rgb": [
                        int(v)
                        for v in self.demo31_contract.get(
                            "tracking_control_point_color_rgb",
                            list(DEFAULT_OVERLAY_CONTROL_POINT_COLOR_RGB),
                        )
                    ],
                    "overlay_color_mode": (
                        "by_camera" if bool(getattr(self.args, "overlay_debug_color_by_camera", False)) else "solid"
                    ),
                    "tracker_visualization_mode": str(tracker_visualization_mode),
                    "tracker_3d_marker_mode": "surface_snap" if surface_marker_mode else str(tracker_visualization_mode),
                    "tracker_3d_marker_shape": "sphere",
                    "tracker_legacy_lift_used": bool(legacy_lift_mode),
                    "tracker_direct_depth_lift_used": bool(direct_depth_lift_mode),
                    "tracker_all_tracks_anchor_mode": bool(all_tracks_lift_mode),
                    "tracker_surface_gate_enabled": bool(surface_marker_mode),
                    "tracker_3d_snap_radius_px": float(tracker_snap_radius_px),
                    "tracker_3d_marker_radius_m": float(control_point_radius_m),
                    "tracker_control_points_per_camera": int(tracker_control_points_per_camera),
                    "tracker_control_point_selection": str(tracker_control_point_selection),
                    "tracker_surface_anchor_cache_hit": bool(tracker_surface_anchor_cache_hit),
                    "tracker_surface_anchor_group_id": tracker_surface_anchor_group_id,
                    "tracker_marker_accepted_by_camera": dict(tracker_marker_accepted_by_camera),
                    "tracker_marker_rejected_by_camera": dict(tracker_marker_rejected_by_camera),
                    "tracker_marker_pixel_error_median_by_camera": dict(tracker_marker_pixel_error_median_by_camera),
                    "tracker_marker_pixel_error_p95_by_camera": dict(tracker_marker_pixel_error_p95_by_camera),
                    "tracker_marker_layer_by_camera": dict(tracker_marker_layer_by_camera),
                    "tracker_marker_points_rendered": int(len(overlay_points)),
                    "tracker_marker_points_appended": bool(len(overlay_points) > 0),
                    "overlay_lift_method": (
                        "surface_snap"
                        if surface_marker_mode
                        else "all_tracks_depth_lift"
                        if all_tracks_lift_mode
                        else "semantic_projection_grid"
                    ),
                    "overlay_lift_mask_scope": str(
                        "none"
                        if all_tracks_lift_mode
                        else getattr(self.args, "overlay_display_scope", demo3_runtime.DEFAULT_OVERLAY_DISPLAY_SCOPE)
                    ),
                    "overlay_input_points_by_camera": dict(overlay_input_points_by_camera),
                    "overlay_points_by_camera": dict(overlay_points_by_camera),
                    "overlay_rejected_by_scope_mask_by_camera": (
                        {int(camera_idx): 0 for camera_idx in overlay_input_points_by_camera}
                        if all_tracks_lift_mode
                        else {
                            int(camera_idx): int(input_count)
                            - int(overlay_lifted_points_by_camera.get(int(camera_idx), 0))
                            for camera_idx, input_count in overlay_input_points_by_camera.items()
                        }
                    ),
                    "overlay_rejected_by_depth_or_bounds_by_camera": {
                        int(camera_idx): int(input_count) - int(overlay_lifted_points_by_camera.get(int(camera_idx), 0))
                        for camera_idx, input_count in overlay_input_points_by_camera.items()
                    },
                    "overlay_bbox_filter_enabled": bool(bbox_filter_enabled),
                    "overlay_bbox_filter_scope": str(
                        getattr(self.args, "overlay_display_scope", demo3_runtime.DEFAULT_OVERLAY_DISPLAY_SCOPE)
                    ),
                    "overlay_bbox_filter_margin_m": float(
                        getattr(
                            self.args,
                            "overlay_max_distance_from_controller_m",
                            DEFAULT_OVERLAY_MAX_DISTANCE_FROM_CONTROLLER_M,
                        )
                    ),
                    "overlay_bbox_input_points_by_camera": dict(overlay_bbox_input_points_by_camera),
                    "overlay_bbox_kept_points_by_camera": dict(overlay_bbox_kept_points_by_camera),
                    "overlay_bbox_rejected_by_camera": dict(overlay_bbox_rejected_by_camera),
                    "overlay_world_centroid_by_camera_before_bbox": dict(overlay_centroid_before_bbox_by_camera),
                    "overlay_world_centroid_by_camera": dict(overlay_centroid_by_camera),
                    "tracking_control_point_markers": bool(control_markers_enabled),
                    "tracking_control_point_count_requested": int(control_point_count_requested),
                    "tracking_control_point_count": int(tracking_control_point_count),
                    "tracking_control_points_by_camera": dict(tracking_control_points_by_camera),
                    "tracking_control_point_radius_m": float(control_point_radius_m),
                    "tracking_control_point_sampling": (
                        f"{tracker_control_point_selection}_surface_snap"
                        if surface_marker_mode
                        else "all_visible_depth_valid_tracks_no_surface_or_bbox_gate"
                        if all_tracks_lift_mode
                        else "farthest_point_sample_after_lift_scope_and_bbox"
                    ),
                    "tracking_control_marker_points": int(tracking_control_marker_points),
                    "tracking_control_point_centroid": overlay_control_point_centroid,
                    "overlay_render_raw_track_points": bool(render_raw_tracks),
                    "overlay_ms": overlay_ms,
                    "overlay_group_id": overlay_group_id,
                    "incoming_render_group_id": int(overlay_group_id),
                    "render_group_id": profile_group_id,
                    "overlay_render_group_delta": overlay_render_group_delta,
                    "render_driver": "cotracker_child_output",
                    "render_trigger": "new_cotracker_result",
                    "rendered_on_new_cotracker_result": bool(render_packet is not None and len(overlay_points) > 0),
                    "tracking_overlay_render_blocked": tracking_overlay_render_blocked,
                    "tracking_overlay_warmup_blocked": tracking_overlay_warmup_blocked,
                    "tracking_overlay_required_before_first_render": bool(render_requires_new_tracker),
                    "tracking_overlay_required_for_render": bool(render_requires_new_tracker),
                    "render_requires_new_cotracker_result": bool(render_requires_new_tracker),
                    "render_reuses_cached_cotracker_result": False,
                    "tracking_mask_source_group_id": (
                        None if overlay is None or overlay.mask_source_group_id is None else int(overlay.mask_source_group_id)
                    ),
                    "tracking_mask_age_ms": 0.0 if overlay is None else float(overlay.mask_age_ms),
                    "tracking_mask_reused": False if overlay is None else bool(overlay.mask_reused),
                    "overlay_lift_cache_hit": bool(overlay_lift_cache_hit),
                    "tracking_result_has_matching_render_packet": bool(
                        render_match_profile["tracking_result_has_matching_render_packet"]
                    ),
                    "tracking_result_used_render_packet": bool(
                        render_match_profile["tracking_result_used_render_packet"]
                    ),
                    "tracking_result_used_nearest_render_packet": bool(
                        render_match_profile["tracking_result_used_nearest_render_packet"]
                    ),
                    "tracking_render_packet_match_mode": str(
                        render_match_profile["tracking_render_packet_match_mode"]
                    ),
                    "tracking_render_packet_match_policy": TRACKING_RENDER_PACKET_MATCH_POLICY,
                    "tracking_render_packet_group_id": render_match_profile["tracking_render_packet_group_id"],
                    "tracking_render_packet_group_delta": render_match_profile["tracking_render_packet_group_delta"],
                    "tracking_nearest_render_packet_abs_delta": render_match_profile[
                        "tracking_nearest_render_packet_abs_delta"
                    ],
                    "tracking_pending_render_packet_count_before_match": render_match_profile[
                        "tracking_pending_render_packet_count_before_match"
                    ],
                    "tracking_pending_render_packet_max_groups": int(self.demo31_pending_render_packet_max_groups),
                    "tracking_pending_render_packet_had_lift_candidate": bool(
                        render_match_profile["tracking_pending_render_packet_had_lift_candidate"]
                    ),
                    "cotracker_model_ms": None if overlay is None else float(overlay.model_ms),
                    "cotracker_e2e_ms": None if overlay is None else float(overlay.e2e_ms),
                    "cotracker_publish_to_render_ms": (
                        None if overlay is None else float((overlay_start_s - float(overlay.publish_timestamp_s)) * 1000.0)
                    ),
                    "cotracker_source_to_render_ms": (
                        None if overlay is None else float((overlay_start_s - float(overlay.source_timestamp_s)) * 1000.0)
                    ),
                    "cotracker_publish_range": None if overlay is None else [int(item) for item in overlay.publish_range],
                    "cotracker_update_mode": None if overlay is None else str(overlay.cotracker_update_mode),
                    "cotracker_batch_size": None if overlay is None else int(overlay.cotracker_batch_size),
                    "cotracker_batch_update_count": (
                        None if overlay is None else int(overlay.cotracker_batch_update_count)
                    ),
                    "cotracker_serial_group_update_count": (
                        None if overlay is None else int(overlay.cotracker_serial_group_update_count)
                    ),
                    "cotracker_serial_camera_update_count": (
                        None if overlay is None else int(overlay.cotracker_serial_camera_update_count)
                    ),
                    "cotracker_serial_fallback_count": (
                        None if overlay is None else int(overlay.cotracker_serial_fallback_count)
                    ),
                    "cotracker_batch_error_count": (
                        None if overlay is None else int(overlay.cotracker_batch_error_count)
                    ),
                    "cotracker_batch_disabled_reason": (
                        None if overlay is None else overlay.cotracker_batch_disabled_reason
                    ),
                    "tracking_query_count_actual_by_camera": (
                        {} if overlay is None else dict(overlay.tracking_query_count_actual_by_camera)
                    ),
                    "overlay_display_count_by_camera": (
                        {} if overlay is None else dict(overlay.overlay_display_count_by_camera)
                    ),
                    "overlay_display_controller_count_by_camera": (
                        {} if overlay is None else dict(overlay.overlay_display_controller_count_by_camera)
                    ),
                    "overlay_display_object_count_by_camera": (
                        {} if overlay is None else dict(overlay.overlay_display_object_count_by_camera)
                    ),
                    "render_waited_for_cotracker": bool(render_requires_new_tracker),
                    "render_waited_for_fresh_cotracker_result": bool(render_requires_new_tracker),
                    "cross_gpu_cuda_tensor_transfer": False,
                },
            )
            if tracking_overlay_render_blocked:
                self.demo31_tracking_overlay_render_blocked_count += 1
                if tracking_overlay_warmup_blocked:
                    self.demo31_tracking_overlay_warmup_skipped_render_count += 1
                return
            if render_packet is None:
                if not render_requires_new_tracker:
                    return
                return
            self._drop_pending_render_packets_through(int(render_packet.group_id))
            if self.demo31_tracking_overlay_first_render_group_id is None:
                self.demo31_tracking_overlay_first_render_group_id = int(render_packet.group_id)
            super()._publish_render_packet(render_packet)

        def _remember_pending_render_packet(self, packet: Any) -> None:
            with self.demo31_pending_render_lock:
                self.demo31_pending_render_packets[int(packet.group_id)] = packet
                while len(self.demo31_pending_render_packets) > int(self.demo31_pending_render_packet_max_groups):
                    oldest = min(self.demo31_pending_render_packets)
                    self.demo31_pending_render_packets.pop(oldest, None)
                    self.demo31_pending_render_packet_drop_count += 1

        def _drop_pending_render_packets_through(self, group_id: int) -> None:
            with self.demo31_pending_render_lock:
                stale_ids = [key for key in self.demo31_pending_render_packets if int(key) <= int(group_id)]
                for key in stale_ids:
                    self.demo31_pending_render_packets.pop(key, None)

        def _take_fresh_tracking_result(self, *, now_s: float) -> TrackingResultLitePacket | None:
            if self.demo31_process_client is None:
                return None
            result = self.demo31_process_client.get_result()
            if result is not None:
                fresh = fresh_tracking_result_or_none(
                    result,
                    now_s=now_s,
                    stale_timeout_ms=float(self.demo31_contract["cotracker_result_stale_timeout_ms"]),
                )
                if fresh is None:
                    self.demo31_tracking_input_drop_count += 1
                else:
                    self._record_new_tracking_result(fresh, now_s=now_s)
                    return fresh
            return None

        def _record_new_tracking_result(self, result: TrackingResultLitePacket, *, now_s: float) -> None:
            age_ms = max(0.0, (now_s - float(result.publish_timestamp_s)) * 1000.0)
            self.demo31_overlay_age_ms_samples.append(float(age_ms))
            self.demo31_overlay_model_ms_samples.append(float(result.model_ms))
            self.demo31_overlay_e2e_ms_samples.append(float(result.e2e_ms))
            self.demo31_cotracker_publish_times_s.append(float(result.publish_timestamp_s))
            self.demo31_tracking_mask_selection_count += 1
            self.demo31_tracking_mask_reuse_count += int(bool(result.mask_reused))
            self.demo31_tracking_mask_age_ms_samples.append(float(result.mask_age_ms))
            self.demo31_tracking_stats = {
                "tracking_query_count_actual_by_camera": dict(result.tracking_query_count_actual_by_camera),
                "tracking_union_pixels_by_camera": dict(result.tracking_union_pixels_by_camera),
                "tracking_object_pixels_by_camera": dict(result.tracking_object_pixels_by_camera),
                "tracking_controller_pixels_by_camera": dict(result.tracking_controller_pixels_by_camera),
                "tracking_sample_object_hits_by_camera": dict(result.tracking_sample_object_hits_by_camera),
                "tracking_sample_controller_hits_by_camera": dict(result.tracking_sample_controller_hits_by_camera),
                "tracking_sample_overlap_hits_by_camera": dict(result.tracking_sample_overlap_hits_by_camera),
                "tracking_sample_background_hits_by_camera": dict(result.tracking_sample_background_hits_by_camera),
                "overlay_display_scope": str(result.overlay_display_scope),
                "overlay_display_count_by_camera": dict(result.overlay_display_count_by_camera),
                "overlay_display_object_count_by_camera": dict(result.overlay_display_object_count_by_camera),
                "overlay_display_controller_count_by_camera": dict(result.overlay_display_controller_count_by_camera),
                "cotracker_update_mode": str(result.cotracker_update_mode),
                "tracker_backend": str(getattr(result, "tracker_backend", self.demo31_contract.get("tracker_backend", TRACKER_BACKEND_COTRACKER3))),
                "tracking_backend_execution_mode": str(
                    getattr(result, "tracking_backend_execution_mode", self.demo31_contract.get("tracking_backend_execution_mode", TRACKING_BACKEND_EXECUTION_MODE_AUTO))
                ),
                "tracker_batch_query_count_policy": str(
                    getattr(result, "tracker_batch_query_count_policy", self.demo31_contract.get("tracker_batch_query_count_policy", TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED))
                ),
                "tracking_backend_effective_query_count": int(
                    getattr(result, "tracking_backend_effective_query_count", 0)
                ),
                "tracking_backend_query_count_truncated_by_camera": dict(
                    getattr(result, "tracking_backend_query_count_truncated_by_camera", {})
                ),
                "tracking_backend_batch_fallback_reason": getattr(
                    result,
                    "tracking_backend_batch_fallback_reason",
                    result.cotracker_batch_disabled_reason,
                ),
                "cotracker_batch_size": int(result.cotracker_batch_size),
                "cotracker_batch_update_count": int(result.cotracker_batch_update_count),
                "cotracker_serial_group_update_count": int(result.cotracker_serial_group_update_count),
                "cotracker_serial_camera_update_count": int(result.cotracker_serial_camera_update_count),
                "cotracker_serial_fallback_count": int(result.cotracker_serial_fallback_count),
                "cotracker_batch_error_count": int(result.cotracker_batch_error_count),
                "cotracker_batch_disabled_reason": result.cotracker_batch_disabled_reason,
                "tracking_mask_source_group_id": (
                    None if result.mask_source_group_id is None else int(result.mask_source_group_id)
                ),
                "tracking_mask_age_ms": float(result.mask_age_ms),
                "tracking_mask_reused": bool(result.mask_reused),
            }

        def demo31_snapshot(self) -> dict[str, Any]:
            self._drain_demo31_process_status()
            process_snapshot = (
                self.demo31_process_client.snapshot()
                if self.demo31_process_client is not None and hasattr(self.demo31_process_client, "snapshot")
                else None
            )
            age = percentile_summary(self.demo31_overlay_age_ms_samples)
            model = percentile_summary(self.demo31_overlay_model_ms_samples)
            e2e = percentile_summary(self.demo31_overlay_e2e_ms_samples)
            overlay_delta = percentile_summary(self.demo31_overlay_render_group_delta_samples)
            tracking_mask_age = percentile_summary(self.demo31_tracking_mask_age_ms_samples)
            with self.demo31_pending_render_lock:
                pending_render_count = int(len(self.demo31_pending_render_packets))
            return {
                "process": process_snapshot,
                "process_status_events": list(self.demo31_process_status_events),
                "stage_join_buffer": self.stage_join_buffer.snapshot()
                if hasattr(self.stage_join_buffer, "snapshot")
                else {},
                "tracking_input_skip_count": int(self.demo31_tracking_input_skip_count),
                "tracking_input_queue_replace_count": int(self.demo31_tracking_input_queue_replace_count),
                "tracking_input_drop_count": int(self.demo31_tracking_input_drop_count),
                "cotracker_input_count": int(len(self.demo31_tracking_input_publish_times_s)),
                "cotracker_input_fps": float(event_fps(self.demo31_tracking_input_publish_times_s)),
                "cotracker_result_count": int(len(self.demo31_cotracker_publish_times_s)),
                "cotracker_publish_fps": float(event_fps(self.demo31_cotracker_publish_times_s)),
                "tracking_pending_render_packets": pending_render_count,
                "tracking_pending_render_packet_max_groups": int(self.demo31_pending_render_packet_max_groups),
                "tracking_pending_render_packet_drop_count": int(self.demo31_pending_render_packet_drop_count),
                "tracking_result_without_render_packet_count": int(
                    self.demo31_tracking_result_without_render_packet_count
                ),
                "tracking_result_exact_render_packet_count": int(
                    self.demo31_tracking_result_exact_render_packet_count
                ),
                "tracking_result_nearest_render_packet_count": int(
                    self.demo31_tracking_result_nearest_render_packet_count
                ),
                "tracking_result_without_lift_input_count": int(self.demo31_tracking_result_without_lift_input_count),
                "tracking_render_packet_match_policy": TRACKING_RENDER_PACKET_MATCH_POLICY,
                "overlay_age_ms_median": float(age["median"]),
                "overlay_age_ms_p95": float(age["p95"]),
                "cotracker_model_ms_median": float(model["median"]),
                "cotracker_model_ms_p95": float(model["p95"]),
                "cotracker_e2e_ms_median": float(e2e["median"]),
                "cotracker_e2e_ms_p95": float(e2e["p95"]),
                "overlay_render_group_delta_median": float(overlay_delta["median"]),
                "overlay_render_group_delta_p95": float(overlay_delta["p95"]),
                "overlay_render_group_mismatch_count": int(self.demo31_overlay_render_group_mismatch_count),
                "tracking_overlay_warmup_skipped_render_count": int(
                    self.demo31_tracking_overlay_warmup_skipped_render_count
                ),
                "tracking_overlay_render_blocked_count": int(self.demo31_tracking_overlay_render_blocked_count),
                "tracking_overlay_first_render_group_id": self.demo31_tracking_overlay_first_render_group_id,
                "tracking_input_mask_reuse_ratio": (
                    float(self.demo31_tracking_mask_reuse_count / self.demo31_tracking_mask_selection_count)
                    if self.demo31_tracking_mask_selection_count
                    else 0.0
                ),
                "tracking_input_mask_age_ms_median": float(tracking_mask_age["median"]),
                "tracking_input_mask_age_ms_p95": float(tracking_mask_age["p95"]),
                "mask_cache": self.demo31_mask_cache.snapshot(),
                "lift_input_cache": self.demo31_lift_input_cache.snapshot(),
                "surface_anchor_cache": self.demo31_surface_anchor_cache.snapshot(),
                "controller_mask_erode_px": int(self.demo31_contract.get("controller_mask_erode_px", 0) or 0),
                "controller_mask_pixels_before_erode_by_camera": dict(
                    self.demo31_controller_mask_pixels_before_erode_by_camera
                ),
                "controller_mask_pixels_after_erode_by_camera": dict(
                    self.demo31_controller_mask_pixels_after_erode_by_camera
                ),
                "first_trackable_mask_group_id": self.demo32_first_trackable_mask_group_id,
                "first_trackable_mask_s": self.demo32_first_trackable_mask_s,
                "first_tracking_input_publish_s": self.demo32_first_tracking_input_publish_s,
                "trackable_mask_initialized_cameras": sorted(self.demo32_trackable_masks_initialized_by_camera),
                "trackable_mask_stats": summarize_trackable_stats(self.demo32_trackable_mask_stats_by_camera),
                "tracking_stats": dict(self.demo31_tracking_stats),
            }

    return Demo31LiveRuntime


class Demo31Runtime:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        shared_runtime_module: Any | None = None,
        shared_runtime_cls: type | None = None,
        connected_serials_provider: ConnectedSerialsProvider | None = None,
        cuda_device_count_provider: CudaDeviceCountProvider | None = None,
        process_client_factory: ProcessClientFactory | None = None,
    ) -> None:
        self.args = args
        self.cuda_device_count_provider = cuda_device_count_provider
        self.contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        self.shared_runtime_module = shared_runtime_module
        self.shared_runtime_cls = shared_runtime_cls
        self.connected_serials_provider = connected_serials_provider
        self.process_client_factory = process_client_factory

    def run(self) -> dict[str, Any]:
        live_validation = validate_live_realsense_contract(
            self.args,
            connected_serials_provider=self.connected_serials_provider,
            cuda_device_count_provider=self.cuda_device_count_provider,
        )
        shared = self.shared_runtime_module or demo3_runtime._load_shared_runtime_module()
        shared_profile = demo3_runtime._shared_profile_path(self.args)
        shared_args = build_shared_runtime_args(
            self.args,
            shared_runtime_module=shared,
            live_validation=live_validation,
            shared_profile_path=shared_profile,
        )
        runtime_cls = self.shared_runtime_cls or make_demo31_live_runtime_class(
            shared,
            process_client_factory=self.process_client_factory,
        )
        if self.shared_runtime_cls is None:
            runtime = runtime_cls(
                shared_args,
                demo31_contract=self.contract,
                cotracker_process_config=build_cotracker_process_config(self.args),
                cotracker_enabled=not bool(self.args.disable_cotracker),
            )
        else:
            runtime = runtime_cls(shared_args)
        exit_code = int(runtime.run())
        shared_payload = demo3_runtime._load_json_if_exists(shared_profile)
        snapshot = runtime.demo31_snapshot() if hasattr(runtime, "demo31_snapshot") else None
        summary = self._build_summary(runtime=runtime, exit_code=exit_code, snapshot=snapshot, shared_payload=shared_payload)
        profile = {
            "contract": self.contract,
            "summary": summary,
            "live_validation": live_validation,
            "shared_runtime_profile": None if shared_profile is None else str(shared_profile),
            "shared_runtime_profile_payload": shared_payload,
            "cotracker_process_snapshot": snapshot,
            "runtime_note": "Demo 3.1 delegates capture/mask/fusion/render to the shared runtime and runs CoTracker3 in an isolated latest-wins process.",
            "exit_code": exit_code,
        }
        _write_profile(self.args.profile_json_output, profile)
        return profile

    def _build_summary(
        self,
        *,
        runtime: Any,
        exit_code: int,
        snapshot: dict[str, Any] | None,
        shared_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        summary = build_empty_dual_gpu_profile_summary(self.contract)
        final = getattr(runtime, "_summary", {}).get("final", {}) if hasattr(runtime, "_summary") else {}
        warm = (shared_payload or {}).get("summary_after_warmup", {})
        gpu_by_device = (shared_payload or {}).get("gpu_sampling", {}).get("summary_by_device_after_warmup", {})

        def _gpu_metric(device_index: int, metric: str, stat: str) -> float:
            if not isinstance(gpu_by_device, dict):
                return 0.0
            device_summary = gpu_by_device.get(str(int(device_index)), {})
            if not isinstance(device_summary, dict):
                return 0.0
            value = demo3_runtime._nested_get(device_summary, ("metrics", metric, stat), 0.0)
            return float(value or 0.0)

        summary.update(
            {
                "exit_code": int(exit_code),
                "rendered_fps": float(final.get("render_fps", warm.get("render_fps", 0.0)) or 0.0),
                "render_loop_fps": float(final.get("render_fps", warm.get("render_fps", 0.0)) or 0.0),
                "new_fused_pcd_fps": float(final.get("fusion_fps", warm.get("fusion_fps", 0.0)) or 0.0),
                "capture_group_fps": float(final.get("capture_group_fps", warm.get("capture_group_fps", 0.0)) or 0.0),
                "gpu0_util_median": _gpu_metric(0, "gpu_util_pct", "median"),
                "gpu0_util_p95": _gpu_metric(0, "gpu_util_pct", "p95"),
                "gpu0_mem_used_gb": _gpu_metric(0, "memory_used_mb", "median") / 1024.0,
                "gpu1_util_median": _gpu_metric(1, "gpu_util_pct", "median"),
                "gpu1_util_p95": _gpu_metric(1, "gpu_util_pct", "p95"),
                "gpu1_mem_used_gb": _gpu_metric(1, "memory_used_mb", "median") / 1024.0,
                "main_process_pid": int(os.getpid()),
            }
        )
        if snapshot:
            process = snapshot.get("process") or {}
            process_ready = process.get("ready") if isinstance(process.get("ready"), dict) else {}
            warmup_profile = (
                process_ready.get("warmup_profile")
                if isinstance(process_ready, dict) and isinstance(process_ready.get("warmup_profile"), dict)
                else {}
            )
            mask_cache = snapshot.get("stage_join_buffer") or snapshot.get("mask_cache") or {}
            input_endpoint = process.get("input_endpoint") or {}
            tracking_stats = snapshot.get("tracking_stats") or {}
            summary.update(
                {
                    "cotracker_process_pid": int(process.get("pid", 0) or 0),
                    "cotracker_process_ready": bool(process_ready),
                    "tracker_process_pid": int(process.get("pid", 0) or 0),
                    "tracker_process_ready": bool(process_ready),
                    "tracker_ready_to_receive_inputs": bool(
                        process_ready.get("ready_to_receive_inputs", bool(process_ready))
                        if isinstance(process_ready, dict)
                        else False
                    ),
                    "tracker_process_ready_s": float(
                        (process_ready.get("total_init_ms", 0.0) if isinstance(process_ready, dict) else 0.0) / 1000.0
                    ),
                    "tracker_ready_to_receive_inputs_s": float(
                        (process_ready.get("total_init_ms", 0.0) if isinstance(process_ready, dict) else 0.0) / 1000.0
                        if isinstance(process_ready, dict) and bool(process_ready.get("ready_to_receive_inputs", False))
                        else 0.0
                    ),
                    "tracker_ready_state": str(
                        process_ready.get("ready_state", self.contract.get("tracker_ready_state", "ready_to_receive_inputs"))
                        if isinstance(process_ready, dict)
                        else self.contract.get("tracker_ready_state", "ready_to_receive_inputs")
                    ),
                    "cotracker_process_total_init_ms": float(
                        process_ready.get("total_init_ms", 0.0) if isinstance(process_ready, dict) else 0.0
                    ),
                    "tracker_process_total_init_ms": float(
                        process_ready.get("total_init_ms", 0.0) if isinstance(process_ready, dict) else 0.0
                    ),
                    "cotracker_prewarm_backends": bool(
                        process_ready.get("prewarm_backends", self.contract.get("cotracker_prewarm_backends", True))
                        if isinstance(process_ready, dict)
                        else self.contract.get("cotracker_prewarm_backends", True)
                    ),
                    "tracker_prewarm_backends": bool(
                        process_ready.get("prewarm_backends", self.contract.get("tracker_prewarm_backends", True))
                        if isinstance(process_ready, dict)
                        else self.contract.get("tracker_prewarm_backends", True)
                    ),
                    "tracker_prewarm_mode": str(
                        process_ready.get("tracker_prewarm_mode", self.contract.get("tracker_prewarm_mode", "unknown"))
                        if isinstance(process_ready, dict)
                        else self.contract.get("tracker_prewarm_mode", "unknown")
                    ),
                    "tracker_query_dependent_init": bool(
                        process_ready.get(
                            "tracker_query_dependent_init",
                            self.contract.get("tracker_query_dependent_init", False),
                        )
                        if isinstance(process_ready, dict)
                        else self.contract.get("tracker_query_dependent_init", False)
                    ),
                    "tracker_query_dependent_init_pending": bool(
                        process_ready.get(
                            "tracker_query_dependent_init_pending",
                            self.contract.get("tracker_query_dependent_init_pending_until_first_input", False),
                        )
                        if isinstance(process_ready, dict)
                        else self.contract.get("tracker_query_dependent_init_pending_until_first_input", False)
                    ),
                    "cotracker_backend_warmup_ms": float(
                        warmup_profile.get("total_ms", 0.0) if isinstance(warmup_profile, dict) else 0.0
                    ),
                    "tracker_backend_warmup_ms": float(
                        warmup_profile.get("total_ms", 0.0) if isinstance(warmup_profile, dict) else 0.0
                    ),
                    "cotracker_backend_warmup_by_camera": (
                        warmup_profile.get("per_camera", {}) if isinstance(warmup_profile, dict) else {}
                    ),
                    "tracker_backend_warmup_by_camera": (
                        warmup_profile.get("per_camera", {}) if isinstance(warmup_profile, dict) else {}
                    ),
                    "cotracker_update_mode": str(
                        tracking_stats.get("cotracker_update_mode", self.contract.get("cotracker_update_mode", "batch"))
                    ),
                    "tracker_backend": str(
                        tracking_stats.get("tracker_backend", self.contract.get("tracker_backend", TRACKER_BACKEND_COTRACKER3))
                    ),
                    "tracker_backend_family": str(self.contract.get("tracker_backend_family", "cotracker")),
                    "tracking_backend_execution_mode": str(
                        tracking_stats.get(
                            "tracking_backend_execution_mode",
                            self.contract.get("tracking_backend_execution_mode", DEFAULT_TRACKING_BACKEND_EXECUTION_MODE),
                        )
                    ),
                    "tracker_batch_query_count_policy": str(
                        tracking_stats.get(
                            "tracker_batch_query_count_policy",
                            self.contract.get("tracker_batch_query_count_policy", TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED),
                        )
                    ),
                    "tracking_backend_batch_enabled": bool(
                        str(tracking_stats.get("cotracker_update_mode", self.contract.get("cotracker_update_mode", "batch")))
                        == "batch"
                    ),
                    "tracking_backend_batch_size": int(tracking_stats.get("cotracker_batch_size", 0) or 0),
                    "tracking_backend_effective_query_count": int(
                        tracking_stats.get("tracking_backend_effective_query_count", 0) or 0
                    ),
                    "tracking_backend_query_count_truncated_by_camera": dict(
                        tracking_stats.get("tracking_backend_query_count_truncated_by_camera", {})
                    ),
                    "tracking_backend_batch_fallback_reason": tracking_stats.get(
                        "tracking_backend_batch_fallback_reason",
                        tracking_stats.get("cotracker_batch_disabled_reason"),
                    ),
                    "cotracker_update_mode_effective": str(
                        tracking_stats.get("cotracker_update_mode", self.contract.get("cotracker_update_mode", "batch"))
                    ),
                    "cotracker_batch_size": int(tracking_stats.get("cotracker_batch_size", 0) or 0),
                    "cotracker_batch_update_count": int(tracking_stats.get("cotracker_batch_update_count", 0) or 0),
                    "cotracker_serial_group_update_count": int(
                        tracking_stats.get("cotracker_serial_group_update_count", 0) or 0
                    ),
                    "cotracker_serial_camera_update_count": int(
                        tracking_stats.get("cotracker_serial_camera_update_count", 0) or 0
                    ),
                    "cotracker_serial_fallback_count": int(
                        tracking_stats.get("cotracker_serial_fallback_count", 0) or 0
                    ),
                    "cotracker_batch_error_count": int(tracking_stats.get("cotracker_batch_error_count", 0) or 0),
                    "cotracker_batch_disabled_reason": tracking_stats.get("cotracker_batch_disabled_reason"),
                    "cotracker_input_drop_count": int(snapshot.get("tracking_input_drop_count", 0) or 0),
                    "cotracker_input_queue_replace_count": int(
                        snapshot.get("tracking_input_queue_replace_count", 0)
                        or input_endpoint.get("replaced", 0)
                        or 0
                    ),
                    "cotracker_model_ms_median": float(snapshot.get("cotracker_model_ms_median", 0.0) or 0.0),
                    "cotracker_model_ms_p95": float(snapshot.get("cotracker_model_ms_p95", 0.0) or 0.0),
                    "cotracker_e2e_ms_median": float(snapshot.get("cotracker_e2e_ms_median", 0.0) or 0.0),
                    "cotracker_e2e_ms_p95": float(snapshot.get("cotracker_e2e_ms_p95", 0.0) or 0.0),
                    "overlay_age_ms_median": float(snapshot.get("overlay_age_ms_median", 0.0) or 0.0),
                    "overlay_age_ms_p95": float(snapshot.get("overlay_age_ms_p95", 0.0) or 0.0),
                    "overlay_render_group_delta_median": float(
                        snapshot.get("overlay_render_group_delta_median", 0.0) or 0.0
                    ),
                    "overlay_render_group_delta_p95": float(
                        snapshot.get("overlay_render_group_delta_p95", 0.0) or 0.0
                    ),
                    "overlay_render_group_mismatch_count": int(
                        snapshot.get("overlay_render_group_mismatch_count", 0) or 0
                    ),
                    "tracking_overlay_warmup_skipped_render_count": int(
                        snapshot.get("tracking_overlay_warmup_skipped_render_count", 0) or 0
                    ),
                    "tracking_overlay_render_blocked_count": int(
                        snapshot.get("tracking_overlay_render_blocked_count", 0) or 0
                    ),
                    "tracking_overlay_first_render_group_id": snapshot.get("tracking_overlay_first_render_group_id"),
                    "tracking_pending_render_packets": int(snapshot.get("tracking_pending_render_packets", 0) or 0),
                    "tracking_pending_render_packet_max_groups": int(
                        snapshot.get("tracking_pending_render_packet_max_groups", 0) or 0
                    ),
                    "tracking_pending_render_packet_drop_count": int(
                        snapshot.get("tracking_pending_render_packet_drop_count", 0) or 0
                    ),
                    "tracking_result_without_render_packet_count": int(
                        snapshot.get("tracking_result_without_render_packet_count", 0) or 0
                    ),
                    "tracking_result_exact_render_packet_count": int(
                        snapshot.get("tracking_result_exact_render_packet_count", 0) or 0
                    ),
                    "tracking_result_nearest_render_packet_count": int(
                        snapshot.get("tracking_result_nearest_render_packet_count", 0) or 0
                    ),
                    "tracking_result_without_lift_input_count": int(
                        snapshot.get("tracking_result_without_lift_input_count", 0) or 0
                    ),
                    "tracking_render_packet_match_policy": str(
                        snapshot.get(
                            "tracking_render_packet_match_policy",
                            self.contract.get("tracking_render_packet_match_policy", TRACKING_RENDER_PACKET_MATCH_POLICY),
                        )
                    ),
                    "mask_reuse_ratio": float(mask_cache.get("mask_reuse_ratio", 0.0) or 0.0),
                    "mask_age_ms_median": float(mask_cache.get("mask_age_ms_median", 0.0) or 0.0),
                    "mask_age_ms_p95": float(mask_cache.get("mask_age_ms_p95", 0.0) or 0.0),
                    "mask_group_delta_median": float(mask_cache.get("mask_group_delta_median", 0.0) or 0.0),
                    "mask_group_delta_p95": float(mask_cache.get("mask_group_delta_p95", 0.0) or 0.0),
                    "tracking_input_mask_reuse_ratio": float(
                        snapshot.get("tracking_input_mask_reuse_ratio", 0.0) or 0.0
                    ),
                    "tracking_input_mask_age_ms_median": float(
                        snapshot.get("tracking_input_mask_age_ms_median", 0.0) or 0.0
                    ),
                    "tracking_input_mask_age_ms_p95": float(
                        snapshot.get("tracking_input_mask_age_ms_p95", 0.0) or 0.0
                    ),
                    "tracking_query_count_actual_by_camera": tracking_stats.get("tracking_query_count_actual_by_camera", {}),
                    "tracking_union_pixels_by_camera": tracking_stats.get("tracking_union_pixels_by_camera", {}),
                    "tracking_object_pixels_by_camera": tracking_stats.get("tracking_object_pixels_by_camera", {}),
                    "tracking_controller_pixels_by_camera": tracking_stats.get("tracking_controller_pixels_by_camera", {}),
                    "tracking_sample_object_hits_by_camera": tracking_stats.get("tracking_sample_object_hits_by_camera", {}),
                    "tracking_sample_controller_hits_by_camera": tracking_stats.get("tracking_sample_controller_hits_by_camera", {}),
                    "tracking_sample_overlap_hits_by_camera": tracking_stats.get("tracking_sample_overlap_hits_by_camera", {}),
                    "tracking_sample_background_hits_by_camera": tracking_stats.get("tracking_sample_background_hits_by_camera", {}),
                    "overlay_display_scope": tracking_stats.get(
                        "overlay_display_scope",
                        self.contract.get("overlay_display_scope", demo3_runtime.DEFAULT_OVERLAY_DISPLAY_SCOPE),
                    ),
                    "overlay_display_count_by_camera": tracking_stats.get("overlay_display_count_by_camera", {}),
                    "overlay_display_object_count_by_camera": tracking_stats.get(
                        "overlay_display_object_count_by_camera",
                        {},
                    ),
                    "overlay_display_controller_count_by_camera": tracking_stats.get(
                        "overlay_display_controller_count_by_camera",
                        {},
                    ),
                }
            )
            _merge_cotracker_process_snapshot_metrics(summary, snapshot)
        return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
    default_preset: str = PRESET_DEMO31_DUAL4090_HIGHFPS,
) -> int:
    parser = build_arg_parser(default_preset=default_preset)
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=demo3_runtime._explicit_cli_options(argv))
        validate_args(args, require_calibration=False, cuda_device_count_provider=cuda_device_count_provider)
        contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        if args.dry_run:
            print(format_contract(contract))
            _write_profile(args.profile_json_output, {"contract": contract, "summary": contract["profile_summary_fields"]})
            return 0
        profile = Demo31Runtime(args, cuda_device_count_provider=cuda_device_count_provider).run()
        print(json.dumps(profile["summary"], indent=2, sort_keys=True))
        return int(profile.get("exit_code", 0))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "Demo31Runtime",
    "FUSION_MASK_POLICY_LATEST_REUSE",
    "FUSION_MASK_POLICY_STRICT",
    "PRESET_DEMO31_DUAL4090_HIGHFPS",
    "PRESET_DEMO32_FFS_LITETRACKER",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_cotracker_process_config",
    "format_contract",
    "fresh_tracking_result_or_none",
    "main",
    "make_demo31_live_runtime_class",
    "Demo31MaskPolicyJoinBuffer",
    "validate_args",
    "validate_live_realsense_contract",
]
