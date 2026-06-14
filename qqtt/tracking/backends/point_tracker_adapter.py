from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

from qqtt.tracking.base import BackendAvailability, TrackingResult


TRACKER_BACKEND_NONE = "none"
TRACKER_BACKEND_TAPNEXTPP = "tapnextpp"
TRACKER_BACKENDS = (TRACKER_BACKEND_NONE, TRACKER_BACKEND_TAPNEXTPP)


@dataclass(frozen=True)
class PointTrackerBackendSpec:
    name: str
    family: str
    supports_batch_views: bool
    supports_online: bool = True
    supports_prewarm: bool = True
    query_format: str = "yx"
    batch_support_status: str = "true_online_serial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "family": str(self.family),
            "supports_batch_views": bool(self.supports_batch_views),
            "supports_online": bool(self.supports_online),
            "supports_prewarm": bool(self.supports_prewarm),
            "query_format": str(self.query_format),
            "batch_support_status": str(self.batch_support_status),
        }


class PointTrackerAdapter(Protocol):
    spec: PointTrackerBackendSpec
    name: str

    def availability(self) -> BackendAvailability:
        ...

    def warmup(self) -> dict[str, Any]:
        ...

    def initialize(self, frames: list[np.ndarray], query_points_yx: np.ndarray, masks: list[np.ndarray] | None = None) -> None:
        ...

    def update(self, frame: np.ndarray) -> TrackingResult:
        ...


def normalize_tracker_backend(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "tapnext++": TRACKER_BACKEND_TAPNEXTPP,
        "tapnext_pp": TRACKER_BACKEND_TAPNEXTPP,
        "tap_next_pp": TRACKER_BACKEND_TAPNEXTPP,
        "tapnext_plus_plus": TRACKER_BACKEND_TAPNEXTPP,
        "tap_next_plus_plus": TRACKER_BACKEND_TAPNEXTPP,
        "off": TRACKER_BACKEND_NONE,
        "disabled": TRACKER_BACKEND_NONE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRACKER_BACKENDS:
        raise ValueError(f"unsupported tracker backend {value!r}; expected one of {TRACKER_BACKENDS}")
    return normalized


def tracker_backend_spec(backend: str) -> PointTrackerBackendSpec:
    normalized = normalize_tracker_backend(backend)
    if normalized == TRACKER_BACKEND_NONE:
        return PointTrackerBackendSpec(
            name=TRACKER_BACKEND_NONE,
            family="none",
            supports_batch_views=False,
            supports_online=False,
            supports_prewarm=False,
            batch_support_status="disabled",
        )
    return PointTrackerBackendSpec(
        name=TRACKER_BACKEND_TAPNEXTPP,
        family="tapnext",
        supports_batch_views=False,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="single_camera_serial",
    )


@dataclass(frozen=True)
class PointTrackerAdapterConfig:
    backend: str = TRACKER_BACKEND_TAPNEXTPP
    device: str = "cuda:1"
    repo_dir: str | None = None
    checkpoint: str | None = None
    tapnet_repo_dir: str | None = None
    tapnextpp_checkpoint: str | None = None
    tapnextpp_image_size: tuple[int, int] = (256, 256)
    tapnextpp_autocast_dtype: str = "fp16"
    tapnextpp_use_certainty: bool = False
    tapnextpp_certainty_radius: int = 8
    tapnextpp_certainty_threshold: float = 0.5
    tapnextpp_compile: bool = False
    tapnextpp_reset_on_reinitialize: bool = True
    tapnextpp_fast_postprocess: bool = True


def _prepend_repo_dir(repo_dir: str | None) -> None:
    if not repo_dir:
        return
    path = str(Path(repo_dir).expanduser())
    if path and path not in sys.path:
        sys.path.insert(0, path)


def build_point_tracker_adapter_factory(config: PointTrackerAdapterConfig) -> Callable[[int], PointTrackerAdapter]:
    backend = normalize_tracker_backend(config.backend)
    if backend == TRACKER_BACKEND_NONE:
        raise ValueError("tracker backend 'none' does not build an adapter")

    def factory(camera_idx: int) -> PointTrackerAdapter:
        from qqtt.tracking.backends.tapnextpp_adapter import TAPNextPPAdapter

        _prepend_repo_dir(config.tapnet_repo_dir or config.repo_dir)
        return TAPNextPPAdapter(
            device=str(config.device),
            camera_idx=None if int(camera_idx) < 0 else int(camera_idx),
            repo_dir=config.tapnet_repo_dir or config.repo_dir,
            checkpoint=config.tapnextpp_checkpoint or config.checkpoint,
            image_size=config.tapnextpp_image_size,
            autocast_dtype=str(config.tapnextpp_autocast_dtype),
            use_certainty=bool(config.tapnextpp_use_certainty),
            certainty_radius=int(config.tapnextpp_certainty_radius),
            certainty_threshold=float(config.tapnextpp_certainty_threshold),
            compile_model=bool(config.tapnextpp_compile),
            reset_on_reinitialize=bool(config.tapnextpp_reset_on_reinitialize),
            fast_postprocess=bool(config.tapnextpp_fast_postprocess),
        )

    return factory


__all__ = [
    "PointTrackerAdapter",
    "PointTrackerAdapterConfig",
    "PointTrackerBackendSpec",
    "TRACKER_BACKEND_NONE",
    "TRACKER_BACKEND_TAPNEXTPP",
    "TRACKER_BACKENDS",
    "build_point_tracker_adapter_factory",
    "normalize_tracker_backend",
    "tracker_backend_spec",
]
