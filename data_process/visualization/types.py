from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CompareCaseSelection:
    aligned_root: Path
    native_case_dir: Path
    ffs_case_dir: Path
    same_case_mode: bool
    native_frame_idx: int
    ffs_frame_idx: int
    camera_ids: list[int]
    serial_numbers: list[str]
    native_c2w: list[np.ndarray]
    native_metadata: dict[str, Any] = field(default_factory=dict)
    ffs_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned_root": str(self.aligned_root),
            "native_case_dir": self.native_case_dir,
            "ffs_case_dir": self.ffs_case_dir,
            "same_case_mode": bool(self.same_case_mode),
            "native_frame_idx": int(self.native_frame_idx),
            "ffs_frame_idx": int(self.ffs_frame_idx),
            "camera_ids": [int(item) for item in self.camera_ids],
            "serial_numbers": list(self.serial_numbers),
            "native_c2w": self.native_c2w,
            "native_metadata": self.native_metadata,
            "ffs_metadata": self.ffs_metadata,
        }


@dataclass(slots=True)
class CameraCloud:
    camera_idx: int
    serial: str
    points: np.ndarray
    colors: np.ndarray
    K_color: np.ndarray
    c2w: np.ndarray
    color_path: str | None = None
    depth_dir_used: str | None = None
    used_float_depth: bool | None = None
    source_camera_idx: np.ndarray | None = None
    source_serial: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "camera_idx": int(self.camera_idx),
            "serial": self.serial,
            "points": self.points,
            "colors": self.colors,
            "K_color": self.K_color,
            "c2w": self.c2w,
        }
        if self.color_path is not None:
            payload["color_path"] = self.color_path
        if self.depth_dir_used is not None:
            payload["depth_dir_used"] = self.depth_dir_used
        if self.used_float_depth is not None:
            payload["used_float_depth"] = bool(self.used_float_depth)
        if self.source_camera_idx is not None:
            payload["source_camera_idx"] = self.source_camera_idx
        if self.source_serial is not None:
            payload["source_serial"] = self.source_serial
        return payload


@dataclass(slots=True)
class ObjectLayers:
    object_points: np.ndarray
    object_colors: np.ndarray
    context_points: np.ndarray
    context_colors: np.ndarray
    combined_points: np.ndarray
    combined_colors: np.ndarray
    object_camera_clouds: list[dict[str, Any]]
    context_camera_clouds: list[dict[str, Any]]
    combined_camera_clouds: list[dict[str, Any]]
    object_source_camera_idx: np.ndarray | None = None
    context_source_camera_idx: np.ndarray | None = None
    combined_source_camera_idx: np.ndarray | None = None
    per_camera_metrics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FrameCloudBundle:
    native_points: np.ndarray
    native_colors: np.ndarray
    native_stats: dict[str, Any]
    native_camera_clouds: list[dict[str, Any]]
    ffs_points: np.ndarray
    ffs_colors: np.ndarray
    ffs_stats: dict[str, Any]
    ffs_camera_clouds: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_points": self.native_points,
            "native_colors": self.native_colors,
            "native_stats": self.native_stats,
            "native_camera_clouds": self.native_camera_clouds,
            "ffs_points": self.ffs_points,
            "ffs_colors": self.ffs_colors,
            "ffs_stats": self.ffs_stats,
            "ffs_camera_clouds": self.ffs_camera_clouds,
        }


@dataclass(slots=True)
class SceneCrop:
    mode: str
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    focus_point: np.ndarray | None = None
    object_roi_min: np.ndarray | None = None
    object_roi_max: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_bounds_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "min": np.asarray(self.bounds_min, dtype=np.float32),
            "max": np.asarray(self.bounds_max, dtype=np.float32),
        }
        if self.object_roi_min is not None:
            payload["object_roi_min"] = np.asarray(self.object_roi_min, dtype=np.float32)
        if self.object_roi_max is not None:
            payload["object_roi_max"] = np.asarray(self.object_roi_max, dtype=np.float32)
        payload.update(self.metadata)
        return payload


@dataclass(slots=True)
class ViewConfig:
    view_name: str
    label: str
    center: np.ndarray
    camera_position: np.ndarray
    up: np.ndarray
    radius: float
    camera_idx: int | None = None
    serial: str | None = None
    angle_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "view_name": self.view_name,
            "label": self.label,
            "center": np.asarray(self.center, dtype=np.float32),
            "camera_position": np.asarray(self.camera_position, dtype=np.float32),
            "up": np.asarray(self.up, dtype=np.float32),
            "radius": float(self.radius),
        }
        if self.camera_idx is not None:
            payload["camera_idx"] = int(self.camera_idx)
        if self.serial is not None:
            payload["serial"] = self.serial
        if self.angle_deg is not None:
            payload["angle_deg"] = float(self.angle_deg)
        return payload


@dataclass(slots=True)
class OrbitPlan:
    orbit_steps: list[dict[str, Any]]
    orbit_path: np.ndarray
    orbit_supported_mask: list[bool]
    orbit_axis: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RenderRequest:
    points: np.ndarray
    colors: np.ndarray
    view_config: dict[str, Any]
    render_mode: str
    renderer: str
    width: int
    height: int
    projection_mode: str
    ortho_scale: float | None = None
    point_radius_px: int = 2
    supersample_scale: int = 1
    zoom_scale: float = 1.0


@dataclass(slots=True)
class RenderOutputSpec:
    name: str
    render_mode: str
    video_name: str
    gif_name: str
    sheet_name: str
    frames_dir_name: str
    write_video: bool = True
    write_gif: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "render_mode": self.render_mode,
            "video_name": self.video_name,
            "gif_name": self.gif_name,
            "sheet_name": self.sheet_name,
            "frames_dir_name": self.frames_dir_name,
            "write_video": bool(self.write_video),
            "write_gif": bool(self.write_gif),
        }


@dataclass(slots=True)
class RenderOutputs:
    output_dir: Path
    metadata: dict[str, Any]
    output_files: dict[str, dict[str, str | None]] = field(default_factory=dict)


@dataclass(slots=True)
class DisplayFrameContract:
    display_frame: str
    calibration_world_frame_kind: str
    uses_semantic_world: bool
    semantic_world_frame_kind: str | None
    overview_display_frame_kind: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_frame": self.display_frame,
            "calibration_world_frame_kind": self.calibration_world_frame_kind,
            "uses_semantic_world": bool(self.uses_semantic_world),
            "semantic_world_frame_kind": self.semantic_world_frame_kind,
            "overview_display_frame_kind": self.overview_display_frame_kind,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class AngleSelectionSummary:
    mode: str
    selected_step_idx: int
    selected_angle_deg: float
    selected_is_supported: bool
    object_projected_area_ratio: float
    object_bbox_fill_ratio: float
    object_multi_camera_support_ratio: float
    object_mismatch_residual_m: float
    context_dominance_penalty: float
    silhouette_penalty: float
    final_score: float
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_step_idx": int(self.selected_step_idx),
            "selected_angle_deg": float(self.selected_angle_deg),
            "selected_is_supported": bool(self.selected_is_supported),
            "object_projected_area_ratio": float(self.object_projected_area_ratio),
            "object_bbox_fill_ratio": float(self.object_bbox_fill_ratio),
            "object_multi_camera_support_ratio": float(self.object_multi_camera_support_ratio),
            "object_mismatch_residual_m": float(self.object_mismatch_residual_m),
            "context_dominance_penalty": float(self.context_dominance_penalty),
            "silhouette_penalty": float(self.silhouette_penalty),
            "final_score": float(self.final_score),
            "candidate_count": int(self.candidate_count),
        }


@dataclass(slots=True)
class TruthPairSelectionSummary:
    src_camera_idx: int
    dst_camera_idx: int
    mean_valid_ratio: float
    residual_gap: float
    object_warp_valid_ratio_native: float
    object_warp_valid_ratio_ffs: float
    object_residual_mean_native: float
    object_residual_mean_ffs: float
    object_edge_weighted_residual_mean_native: float
    object_edge_weighted_residual_mean_ffs: float
    object_overlap_area: float
    pair_object_visibility_score: float
    native: dict[str, Any] = field(default_factory=dict)
    ffs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_camera_idx": int(self.src_camera_idx),
            "dst_camera_idx": int(self.dst_camera_idx),
            "mean_valid_ratio": float(self.mean_valid_ratio),
            "residual_gap": float(self.residual_gap),
            "object_warp_valid_ratio_native": float(self.object_warp_valid_ratio_native),
            "object_warp_valid_ratio_ffs": float(self.object_warp_valid_ratio_ffs),
            "object_residual_mean_native": float(self.object_residual_mean_native),
            "object_residual_mean_ffs": float(self.object_residual_mean_ffs),
            "object_edge_weighted_residual_mean_native": float(self.object_edge_weighted_residual_mean_native),
            "object_edge_weighted_residual_mean_ffs": float(self.object_edge_weighted_residual_mean_ffs),
            "object_overlap_area": float(self.object_overlap_area),
            "pair_object_visibility_score": float(self.pair_object_visibility_score),
            "native": self.native,
            "ffs": self.ffs,
        }


@dataclass(slots=True)
class SourceSummary:
    object_source_histogram: dict[str, int] = field(default_factory=dict)
    context_source_histogram: dict[str, int] = field(default_factory=dict)
    combined_source_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_source_histogram": dict(self.object_source_histogram),
            "context_source_histogram": dict(self.context_source_histogram),
            "combined_source_histogram": dict(self.combined_source_histogram),
        }


@dataclass(slots=True)
class SupportSummary:
    valid_pixel_count: int = 0
    support_ratio_1: float = 0.0
    support_ratio_2: float = 0.0
    support_ratio_3: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_pixel_count": int(self.valid_pixel_count),
            "support_ratio_1": float(self.support_ratio_1),
            "support_ratio_2": float(self.support_ratio_2),
            "support_ratio_3": float(self.support_ratio_3),
        }


@dataclass(slots=True)
class MismatchSummary:
    overlap_pixel_count: int = 0
    residual_mean_m: float = 0.0
    residual_p90_m: float = 0.0
    residual_max_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlap_pixel_count": int(self.overlap_pixel_count),
            "residual_mean_m": float(self.residual_mean_m),
            "residual_p90_m": float(self.residual_p90_m),
            "residual_max_m": float(self.residual_max_m),
        }


@dataclass(slots=True)
class RoiPassSummary:
    mode: str | None
    bounds_min: list[float]
    bounds_max: list[float]
    object_roi_min: list[float] | None = None
    object_roi_max: list[float] | None = None
    object_point_count: int = 0
    object_volume: float = 0.0
    valid_camera_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "min": list(self.bounds_min),
            "max": list(self.bounds_max),
            "object_roi_min": None if self.object_roi_min is None else list(self.object_roi_min),
            "object_roi_max": None if self.object_roi_max is None else list(self.object_roi_max),
            "object_point_count": int(self.object_point_count),
            "object_volume": float(self.object_volume),
            "valid_camera_count": None if self.valid_camera_count is None else int(self.valid_camera_count),
        }


@dataclass(slots=True)
class SourceAttributionCloud:
    points: np.ndarray
    colors: np.ndarray
    source_camera_idx: np.ndarray
    source_serial: np.ndarray | None = None


@dataclass(slots=True)
class DebugArtifactPlan:
    output_dir: Path
    json_paths: list[Path] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    video_paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProductArtifactSet:
    output_dir: Path
    top_level_paths: dict[str, str] = field(default_factory=dict)
    summary_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "top_level_paths": dict(self.top_level_paths),
            "summary_paths": dict(self.summary_paths),
        }


@dataclass(slots=True)
class DebugArtifactSet:
    enabled: bool
    debug_dir: Path | None = None
    paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "debug_dir": None if self.debug_dir is None else str(self.debug_dir),
            "paths": dict(self.paths),
        }
