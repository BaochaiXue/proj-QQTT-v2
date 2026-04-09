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
class SupportSummary:
    histogram: dict[str, int]
    mean_support: float
    max_support: int
    point_count: int


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
