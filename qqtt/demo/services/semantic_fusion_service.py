from __future__ import annotations

from dataclasses import dataclass, field
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


class SemanticFusionService:
    """Shared service boundary for three-camera semantic PCD fusion.

    The full production backprojection path still lives in the legacy runtime;
    this class establishes the tested service contract for the follow-up move.
    """

    def __init__(self, config: SemanticFusionConfig) -> None:
        self.config = config
        self.fuse_count = 0
        self.latest_group_id: int | None = None

    def fuse(self, frames: Mapping[int, CameraSemanticFrame]) -> FusedSemanticPcd:
        if not frames:
            raise ValueError("at least one semantic frame is required")
        group_ids = {int(frame.group_id) for frame in frames.values()}
        if len(group_ids) != 1:
            raise ValueError(f"semantic frames must share a group_id, got {sorted(group_ids)}")
        for camera_idx, frame in frames.items():
            c2w = np.asarray(frame.c2w)
            if c2w.shape != (4, 4):
                raise ValueError(f"camera {camera_idx} c2w must be 4x4")
            if np.asarray(frame.depth_m).shape[:2] != np.asarray(frame.object_mask).shape[:2]:
                raise ValueError(f"camera {camera_idx} depth/object mask shape mismatch")
            if np.asarray(frame.depth_m).shape[:2] != np.asarray(frame.controller_mask).shape[:2]:
                raise ValueError(f"camera {camera_idx} depth/controller mask shape mismatch")
        group_id = int(next(iter(group_ids)))
        self.fuse_count += 1
        self.latest_group_id = group_id
        return FusedSemanticPcd(
            group_id=group_id,
            timestamp_s=float(max(frame.timestamp_s for frame in frames.values())),
            object_xyz=np.empty((0, 3), dtype=np.float32),
            object_rgb=np.empty((0, 3), dtype=np.uint8),
            controller_xyz=np.empty((0, 3), dtype=np.float32),
            controller_rgb=np.empty((0, 3), dtype=np.uint8),
            stats={
                "fusion_camera_count": int(len(frames)),
                "depth_source": self.config.depth_source,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "depth_source": self.config.depth_source,
            "fuse_count": int(self.fuse_count),
            "latest_group_id": self.latest_group_id,
        }


__all__ = [
    "CameraSemanticFrame",
    "FusedSemanticPcd",
    "SemanticFusionConfig",
    "SemanticFusionService",
]
