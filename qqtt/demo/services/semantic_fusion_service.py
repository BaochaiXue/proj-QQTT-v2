from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

import numpy as np

from qqtt.demo.services.service_types import CameraIntrinsics


@dataclass(frozen=True)
class CameraSemanticFrame:
    camera_idx: int
    group_id: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    object_mask: np.ndarray
    controller_mask: np.ndarray
    intrinsics: CameraIntrinsics
    c2w: np.ndarray


@dataclass(frozen=True)
class FusedSemanticPcd:
    group_id: int
    timestamp_s: float
    object_xyz: np.ndarray
    object_rgb: np.ndarray
    controller_xyz: np.ndarray
    controller_rgb: np.ndarray
    camera_debug_xyz: dict[int, np.ndarray] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticFusionConfig:
    depth_source: str
    max_depth_m: float = 3.0
    min_depth_m: float = 0.05
    debug_per_camera_colors: bool = False
    dump_ply_every_n: int | None = None
    mask_overlay_debug: bool = False


class IntrinsicsGridCache:
    """Caches normalized pixel grids for hot-path RGB-D backprojection."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]] = {}

    def get(
        self,
        camera_idx: int,
        height: int,
        width: int,
        intrinsics: CameraIntrinsics,
    ) -> tuple[np.ndarray, np.ndarray]:
        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.cx)
        cy = float(intrinsics.cy)
        if fx == 0.0 or fy == 0.0:
            raise ValueError(f"camera {camera_idx} intrinsics fx/fy must be non-zero")
        key = (int(camera_idx), int(height), int(width), fx, fy, cx, cy)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        u = np.arange(int(width), dtype=np.float32)
        v = np.arange(int(height), dtype=np.float32)
        x_row = (u - np.float32(cx)) / np.float32(fx)
        y_col = (v - np.float32(cy)) / np.float32(fy)
        x_norm_flat = np.tile(x_row, int(height)).astype(np.float32, copy=False)
        y_norm_flat = np.repeat(y_col, int(width)).astype(np.float32, copy=False)
        cached = (x_norm_flat, y_norm_flat)
        self._cache[key] = cached
        return cached

    def snapshot(self) -> dict[str, Any]:
        return {"intrinsics_grid_cache_entries": int(len(self._cache))}


class SemanticFusionService:
    """Shared fast path for three-camera semantic RGB-D point-cloud fusion."""

    def __init__(self, config: SemanticFusionConfig) -> None:
        self.config = config
        self.fuse_count = 0
        self.latest_group_id: int | None = None
        self.grid_cache = IntrinsicsGridCache()

    def fuse(self, frames: Mapping[int, CameraSemanticFrame]) -> FusedSemanticPcd:
        if not frames:
            raise ValueError("at least one semantic frame is required")
        group_ids = {int(frame.group_id) for frame in frames.values()}
        if len(group_ids) != 1:
            raise ValueError(f"semantic frames must share a group_id, got {sorted(group_ids)}")
        for camera_idx, frame in frames.items():
            self._validate_frame(int(camera_idx), frame)
        group_id = int(next(iter(group_ids)))

        mask_valid_ms = 0.0
        backproject_ms = 0.0
        transform_ms = 0.0
        debug_ms = 0.0
        object_xyz_chunks: list[np.ndarray] = []
        object_rgb_chunks: list[np.ndarray] = []
        controller_xyz_chunks: list[np.ndarray] = []
        controller_rgb_chunks: list[np.ndarray] = []
        camera_debug_xyz: dict[int, np.ndarray] = {}
        per_camera_counts: dict[int, dict[str, int]] = {}
        overlap_by_camera: dict[int, int] = {}

        for camera_idx, frame in sorted(frames.items()):
            camera_key = int(camera_idx)
            depth = np.asarray(frame.depth_m, dtype=np.float32)
            object_mask = np.asarray(frame.object_mask, dtype=bool)
            controller_mask = np.asarray(frame.controller_mask, dtype=bool)
            rgb = np.asarray(frame.rgb)
            height, width = depth.shape[:2]
            x_norm_flat, y_norm_flat = self.grid_cache.get(camera_key, height, width, frame.intrinsics)
            depth_flat = depth.reshape(-1)
            rgb_flat = rgb.reshape(-1, rgb.shape[-1])
            c2w = np.asarray(frame.c2w, dtype=np.float32)

            mask_start_s = time.perf_counter()
            valid_depth = (
                np.isfinite(depth)
                & (depth >= float(self.config.min_depth_m))
                & (depth <= float(self.config.max_depth_m))
            )
            object_valid = valid_depth & object_mask
            controller_valid = valid_depth & controller_mask
            overlap_by_camera[camera_key] = int(np.count_nonzero(valid_depth & object_mask & controller_mask))
            mask_valid_ms += float((time.perf_counter() - mask_start_s) * 1000.0)

            obj_xyz, obj_rgb, obj_back_ms, obj_transform_ms = self._backproject_mask(
                valid_mask=object_valid,
                depth_flat=depth_flat,
                rgb_flat=rgb_flat,
                x_norm_flat=x_norm_flat,
                y_norm_flat=y_norm_flat,
                c2w=c2w,
            )
            ctrl_xyz, ctrl_rgb, ctrl_back_ms, ctrl_transform_ms = self._backproject_mask(
                valid_mask=controller_valid,
                depth_flat=depth_flat,
                rgb_flat=rgb_flat,
                x_norm_flat=x_norm_flat,
                y_norm_flat=y_norm_flat,
                c2w=c2w,
            )
            backproject_ms += obj_back_ms + ctrl_back_ms
            transform_ms += obj_transform_ms + ctrl_transform_ms
            object_xyz_chunks.append(obj_xyz)
            object_rgb_chunks.append(obj_rgb)
            controller_xyz_chunks.append(ctrl_xyz)
            controller_rgb_chunks.append(ctrl_rgb)
            per_camera_counts[camera_key] = {
                "object": int(obj_xyz.shape[0]),
                "controller": int(ctrl_xyz.shape[0]),
            }
            if self.config.debug_per_camera_colors:
                debug_start_s = time.perf_counter()
                camera_debug_xyz[camera_key] = (
                    np.concatenate([obj_xyz, ctrl_xyz], axis=0)
                    if obj_xyz.shape[0] or ctrl_xyz.shape[0]
                    else np.empty((0, 3), dtype=np.float32)
                )
                debug_ms += float((time.perf_counter() - debug_start_s) * 1000.0)

        concat_start_s = time.perf_counter()
        object_xyz = self._concat_xyz(object_xyz_chunks)
        object_rgb = self._concat_rgb(object_rgb_chunks)
        controller_xyz = self._concat_xyz(controller_xyz_chunks)
        controller_rgb = self._concat_rgb(controller_rgb_chunks)
        concat_ms = float((time.perf_counter() - concat_start_s) * 1000.0)

        self.fuse_count += 1
        self.latest_group_id = group_id
        return FusedSemanticPcd(
            group_id=group_id,
            timestamp_s=float(max(frame.timestamp_s for frame in frames.values())),
            object_xyz=object_xyz,
            object_rgb=object_rgb,
            controller_xyz=controller_xyz,
            controller_rgb=controller_rgb,
            camera_debug_xyz=camera_debug_xyz,
            stats={
                "fusion_impl": "service-fast",
                "fusion_camera_count": int(len(frames)),
                "depth_source": self.config.depth_source,
                "fusion_object_points": int(object_xyz.shape[0]),
                "fusion_controller_points": int(controller_xyz.shape[0]),
                "per_camera_point_counts": per_camera_counts,
                "object_controller_overlap_pixels_by_camera": overlap_by_camera,
                "fusion_mask_valid_ms": float(mask_valid_ms),
                "fusion_backproject_ms": float(backproject_ms),
                "fusion_transform_ms": float(transform_ms),
                "fusion_concat_ms": float(concat_ms),
                "fusion_debug_ms": float(debug_ms),
                "fusion_debug_per_camera_color": bool(self.config.debug_per_camera_colors),
                "fusion_quality_guard_enabled": False,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "depth_source": self.config.depth_source,
            "fuse_count": int(self.fuse_count),
            "latest_group_id": self.latest_group_id,
            **self.grid_cache.snapshot(),
        }

    def _validate_frame(self, camera_idx: int, frame: CameraSemanticFrame) -> None:
        if int(frame.camera_idx) != int(camera_idx):
            raise ValueError(f"frame camera_idx {frame.camera_idx} does not match mapping key {camera_idx}")
        c2w = np.asarray(frame.c2w)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera {camera_idx} c2w must be 4x4")
        depth = np.asarray(frame.depth_m)
        if depth.ndim != 2:
            raise ValueError(f"camera {camera_idx} depth must be HxW")
        depth_shape = depth.shape[:2]
        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"camera {camera_idx} rgb must be HxWx3")
        if rgb.shape[:2] != depth_shape:
            raise ValueError(f"camera {camera_idx} rgb/depth shape mismatch")
        object_mask = np.asarray(frame.object_mask)
        controller_mask = np.asarray(frame.controller_mask)
        if object_mask.ndim != 2 or object_mask.shape != depth_shape:
            raise ValueError(f"camera {camera_idx} depth/object mask shape mismatch")
        if controller_mask.ndim != 2 or controller_mask.shape != depth_shape:
            raise ValueError(f"camera {camera_idx} depth/controller mask shape mismatch")

    @staticmethod
    def _backproject_mask(
        *,
        valid_mask: np.ndarray,
        depth_flat: np.ndarray,
        rgb_flat: np.ndarray,
        x_norm_flat: np.ndarray,
        y_norm_flat: np.ndarray,
        c2w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        backproject_start_s = time.perf_counter()
        idx = np.flatnonzero(valid_mask.reshape(-1))
        if idx.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=rgb_flat.dtype),
                float((time.perf_counter() - backproject_start_s) * 1000.0),
                0.0,
            )
        z = depth_flat[idx].astype(np.float32, copy=False)
        xyz_cam = np.empty((idx.size, 3), dtype=np.float32)
        xyz_cam[:, 0] = x_norm_flat[idx] * z
        xyz_cam[:, 1] = y_norm_flat[idx] * z
        xyz_cam[:, 2] = z
        colors = rgb_flat[idx]
        backproject_ms = float((time.perf_counter() - backproject_start_s) * 1000.0)

        transform_start_s = time.perf_counter()
        rotation = c2w[:3, :3].astype(np.float32, copy=False)
        translation = c2w[:3, 3].astype(np.float32, copy=False)
        xyz_world = xyz_cam @ rotation.T + translation[None, :]
        transform_ms = float((time.perf_counter() - transform_start_s) * 1000.0)
        return xyz_world.astype(np.float32, copy=False), colors, backproject_ms, transform_ms

    @staticmethod
    def _concat_xyz(chunks: list[np.ndarray]) -> np.ndarray:
        non_empty = [np.asarray(chunk, dtype=np.float32) for chunk in chunks if int(chunk.shape[0]) > 0]
        if not non_empty:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(non_empty, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _concat_rgb(chunks: list[np.ndarray]) -> np.ndarray:
        non_empty = [np.asarray(chunk) for chunk in chunks if int(chunk.shape[0]) > 0]
        if not non_empty:
            return np.empty((0, 3), dtype=np.uint8)
        return np.concatenate(non_empty, axis=0)


def compare_fused_semantic_pcds(
    reference: FusedSemanticPcd,
    candidate: FusedSemanticPcd,
    *,
    voxel_size_m: float = 0.005,
    origin_world: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return offline quality-guard metrics for two fused semantic outputs."""

    origin = np.zeros(3, dtype=np.float32) if origin_world is None else np.asarray(origin_world, dtype=np.float32).reshape(3)
    object_stats = _compare_layer(reference.object_xyz, candidate.object_xyz, voxel_size_m=voxel_size_m, origin_world=origin)
    controller_stats = _compare_layer(
        reference.controller_xyz,
        candidate.controller_xyz,
        voxel_size_m=voxel_size_m,
        origin_world=origin,
    )
    return {
        "fusion_quality_guard_enabled": True,
        "fusion_quality_voxel_size_m": float(voxel_size_m),
        "fusion_object_count_reference": int(reference.object_xyz.shape[0]),
        "fusion_object_count_candidate": int(candidate.object_xyz.shape[0]),
        "fusion_controller_count_reference": int(reference.controller_xyz.shape[0]),
        "fusion_controller_count_candidate": int(candidate.controller_xyz.shape[0]),
        "fusion_object_voxel_iou_5mm": float(object_stats["voxel_iou"]),
        "fusion_controller_voxel_iou_5mm": float(controller_stats["voxel_iou"]),
        "fusion_object_bbox_delta_mm": float(object_stats["bbox_delta_mm"]),
        "fusion_controller_bbox_delta_mm": float(controller_stats["bbox_delta_mm"]),
        "fusion_object_centroid_delta_mm": float(object_stats["centroid_delta_mm"]),
        "fusion_controller_centroid_delta_mm": float(controller_stats["centroid_delta_mm"]),
    }


def _compare_layer(
    reference_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    origin_world: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(reference_xyz, dtype=np.float32).reshape(-1, 3)
    cand = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1, 3)
    return {
        "voxel_iou": _voxel_iou(ref, cand, voxel_size_m=voxel_size_m, origin_world=origin_world),
        "bbox_delta_mm": _bbox_delta_mm(ref, cand),
        "centroid_delta_mm": _centroid_delta_mm(ref, cand),
    }


def _voxel_iou(
    reference_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    origin_world: np.ndarray,
) -> float:
    ref_keys = _voxel_key_set(reference_xyz, voxel_size_m=voxel_size_m, origin_world=origin_world)
    cand_keys = _voxel_key_set(candidate_xyz, voxel_size_m=voxel_size_m, origin_world=origin_world)
    if not ref_keys and not cand_keys:
        return 1.0
    union_count = len(ref_keys | cand_keys)
    return float(len(ref_keys & cand_keys) / union_count) if union_count else 1.0


def _voxel_key_set(
    xyz: np.ndarray,
    *,
    voxel_size_m: float,
    origin_world: np.ndarray,
) -> set[tuple[int, int, int]]:
    if xyz.shape[0] == 0:
        return set()
    q = np.floor((xyz - origin_world[None, :]) / float(voxel_size_m)).astype(np.int64)
    return {tuple(int(v) for v in row) for row in q}


def _bbox_delta_mm(reference_xyz: np.ndarray, candidate_xyz: np.ndarray) -> float:
    if reference_xyz.shape[0] == 0 and candidate_xyz.shape[0] == 0:
        return 0.0
    if reference_xyz.shape[0] == 0 or candidate_xyz.shape[0] == 0:
        return float("inf")
    ref_bounds = np.concatenate([reference_xyz.min(axis=0), reference_xyz.max(axis=0)])
    cand_bounds = np.concatenate([candidate_xyz.min(axis=0), candidate_xyz.max(axis=0)])
    return float(np.max(np.abs(ref_bounds - cand_bounds)) * 1000.0)


def _centroid_delta_mm(reference_xyz: np.ndarray, candidate_xyz: np.ndarray) -> float:
    if reference_xyz.shape[0] == 0 and candidate_xyz.shape[0] == 0:
        return 0.0
    if reference_xyz.shape[0] == 0 or candidate_xyz.shape[0] == 0:
        return float("inf")
    return float(np.linalg.norm(reference_xyz.mean(axis=0) - candidate_xyz.mean(axis=0)) * 1000.0)


__all__ = [
    "CameraSemanticFrame",
    "FusedSemanticPcd",
    "IntrinsicsGridCache",
    "SemanticFusionConfig",
    "SemanticFusionService",
    "compare_fused_semantic_pcds",
]
