from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RenderLayer:
    name: str
    points_xyz: np.ndarray
    colors_rgb: np.ndarray
    point_size: float | None = None
    visible: bool = True
    source_group_id: int | None = None
    source_timestamp_s: float | None = None


@dataclass(frozen=True)
class RenderPacket:
    group_id: int
    timestamp_s: float
    layers: tuple[RenderLayer, ...]
    overlay_layers: tuple[RenderLayer, ...] = ()
    label: str = ""
    created_perf_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


__all__ = [
    "CameraIntrinsics",
    "RenderLayer",
    "RenderPacket",
]
