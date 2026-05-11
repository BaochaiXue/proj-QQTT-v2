from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .lift import LiftedTrackFrame


def compute_2d_track_metrics(tracks_yx: np.ndarray, visibility: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> dict[str, Any]:
    tracks = np.asarray(tracks_yx, dtype=np.float32)
    visible = np.asarray(visibility)
    if visible.ndim == 3 and visible.shape[-1] == 1:
        visible = visible[..., 0]
    visible = visible.astype(bool)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks_yx must have shape (T,N,2); got {tracks.shape}")
    if visible.shape != tracks.shape[:2]:
        raise ValueError(f"visibility shape {visible.shape} does not match tracks {tracks.shape}")
    frame_count, num_tracks = tracks.shape[:2]
    inside_ratios: list[float] = []
    out_of_mask_ratios: list[float] = []
    out_of_bounds_ratios: list[float] = []
    if masks is not None:
        for frame_idx in range(min(frame_count, len(masks))):
            mask = np.asarray(masks[frame_idx], dtype=bool)
            height, width = mask.shape[:2]
            yy = np.rint(tracks[frame_idx, :, 0]).astype(np.int64)
            xx = np.rint(tracks[frame_idx, :, 1]).astype(np.int64)
            in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            frame_visible = visible[frame_idx]
            inside = np.zeros((num_tracks,), dtype=bool)
            inside[in_bounds] = mask[yy[in_bounds], xx[in_bounds]]
            visible_count = max(1, int(frame_visible.sum()))
            inside_ratios.append(float((frame_visible & in_bounds & inside).sum() / visible_count))
            out_of_bounds_ratios.append(float((frame_visible & ~in_bounds).sum() / visible_count))
            out_of_mask_ratios.append(float((frame_visible & in_bounds & ~inside).sum() / visible_count))
    survival_lengths = visible.sum(axis=0).astype(np.float32) if num_tracks else np.empty((0,), dtype=np.float32)
    if frame_count > 1 and num_tracks > 0:
        motion = np.linalg.norm(tracks[1:] - tracks[:-1], axis=2)
        motion_values = motion[visible[1:] & visible[:-1]]
    else:
        motion_values = np.empty((0,), dtype=np.float32)
    return {
        "visible_ratio_mean": float(visible.mean()) if visible.size else 0.0,
        "inside_mask_ratio_mean": float(np.mean(inside_ratios)) if inside_ratios else 0.0,
        "out_of_mask_ratio_mean": float(np.mean(out_of_mask_ratios)) if out_of_mask_ratios else 0.0,
        "out_of_bounds_ratio_mean": float(np.mean(out_of_bounds_ratios)) if out_of_bounds_ratios else 0.0,
        "track_survival_len_mean": float(np.mean(survival_lengths)) if survival_lengths.size else 0.0,
        "median_2d_motion_px": float(np.median(motion_values)) if motion_values.size else 0.0,
        "p95_2d_motion_px": float(np.percentile(motion_values, 95)) if motion_values.size else 0.0,
    }


def compute_3d_lift_metrics(lifted_frames: Sequence[LiftedTrackFrame]) -> dict[str, Any]:
    frames = list(lifted_frames)
    if not frames:
        return {"depth_valid_ratio_mean": 0.0, "lifted_3d_count_mean": 0.0, "per_camera_lifted_count_mean": {}}
    depth_ratios = [float(frame.stats.get("depth_valid_ratio", 0.0)) for frame in frames]
    lifted_counts = [float(frame.stats.get("num_lifted", len(frame.points_world))) for frame in frames]
    per_camera: dict[int, list[int]] = {}
    for frame in frames:
        for camera_idx in np.unique(frame.camera_ids):
            per_camera.setdefault(int(camera_idx), []).append(int((frame.camera_ids == camera_idx).sum()))
    return {
        "depth_valid_ratio_mean": float(np.mean(depth_ratios)) if depth_ratios else 0.0,
        "lifted_3d_count_mean": float(np.mean(lifted_counts)) if lifted_counts else 0.0,
        "per_camera_lifted_count_mean": {str(k): float(np.mean(v)) for k, v in sorted(per_camera.items())},
    }


def summarize_latencies_ms(samples_ms: list[float] | np.ndarray, *, prefix: str) -> dict[str, float]:
    samples = np.asarray(samples_ms, dtype=np.float64)
    if samples.size == 0:
        return {f"{prefix}_ms_median": 0.0, f"{prefix}_ms_p95": 0.0, f"{prefix}_fps": 0.0}
    median = float(np.median(samples))
    return {
        f"{prefix}_ms_median": median,
        f"{prefix}_ms_p95": float(np.percentile(samples, 95)),
        f"{prefix}_fps": 1000.0 / median if median > 0 else 0.0,
    }
