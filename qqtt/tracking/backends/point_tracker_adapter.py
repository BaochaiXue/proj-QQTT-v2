from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult


TRACKER_BACKEND_COTRACKER3 = "cotracker3_online"
TRACKER_BACKEND_TRACKON2 = "trackon2"
TRACKER_BACKEND_LITETRACKER = "litetracker"
TRACKER_BACKENDS = (
    TRACKER_BACKEND_COTRACKER3,
    TRACKER_BACKEND_TRACKON2,
    TRACKER_BACKEND_LITETRACKER,
)

TRACKER_EXECUTION_MODE_AUTO = "auto"
TRACKER_EXECUTION_MODE_SERIAL = "serial"
TRACKER_EXECUTION_MODE_BATCH_VIEWS = "batch-views"
TRACKER_EXECUTION_MODES = (
    TRACKER_EXECUTION_MODE_AUTO,
    TRACKER_EXECUTION_MODE_SERIAL,
    TRACKER_EXECUTION_MODE_BATCH_VIEWS,
)

TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED = "fixed"
TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON = "min-common"
TRACKER_BATCH_QUERY_COUNT_POLICIES = (
    TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED,
    TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON,
)


@dataclass(frozen=True)
class PointTrackerBackendSpec:
    name: str
    family: str
    supports_batch_views: bool
    supports_online: bool = True
    supports_prewarm: bool = True
    query_format: str = "yx"
    batch_support_status: str = "true"

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


@dataclass(frozen=True)
class PointTrackerBatchResult:
    tracks_yx_by_camera: dict[int, np.ndarray]
    visibility_by_camera: dict[int, np.ndarray]
    query_points_yx_by_camera: dict[int, np.ndarray]
    stats: dict[str, Any]


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

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        ...

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        ...


def normalize_tracker_backend(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "cotracker3": TRACKER_BACKEND_COTRACKER3,
        "co_tracker3": TRACKER_BACKEND_COTRACKER3,
        "co_tracker3_online": TRACKER_BACKEND_COTRACKER3,
        "cotracker": TRACKER_BACKEND_COTRACKER3,
        "track_on2": TRACKER_BACKEND_TRACKON2,
        "track_on": TRACKER_BACKEND_TRACKON2,
        "lite_tracker": TRACKER_BACKEND_LITETRACKER,
        "lite-tracker": TRACKER_BACKEND_LITETRACKER,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRACKER_BACKENDS:
        raise ValueError(f"unsupported tracker backend {value!r}; expected one of {TRACKER_BACKENDS}")
    return normalized


def normalize_tracker_execution_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "batch": TRACKER_EXECUTION_MODE_BATCH_VIEWS,
        "batch_views": TRACKER_EXECUTION_MODE_BATCH_VIEWS,
        "batchviews": TRACKER_EXECUTION_MODE_BATCH_VIEWS,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRACKER_EXECUTION_MODES:
        raise ValueError(f"unsupported tracker execution mode {value!r}; expected one of {TRACKER_EXECUTION_MODES}")
    return normalized


def normalize_tracker_batch_query_count_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in TRACKER_BATCH_QUERY_COUNT_POLICIES:
        raise ValueError(
            f"unsupported tracker batch query count policy {value!r}; expected one of {TRACKER_BATCH_QUERY_COUNT_POLICIES}"
        )
    return normalized


def tracker_backend_spec(backend: str) -> PointTrackerBackendSpec:
    normalized = normalize_tracker_backend(backend)
    if normalized == TRACKER_BACKEND_COTRACKER3:
        return PointTrackerBackendSpec(
            name=TRACKER_BACKEND_COTRACKER3,
            family="cotracker",
            supports_batch_views=True,
            batch_support_status="true",
        )
    if normalized == TRACKER_BACKEND_TRACKON2:
        return PointTrackerBackendSpec(
            name=TRACKER_BACKEND_TRACKON2,
            family="trackon",
            supports_batch_views=True,
            batch_support_status="declared",
        )
    return PointTrackerBackendSpec(
        name=TRACKER_BACKEND_LITETRACKER,
        family="litetracker",
        supports_batch_views=False,
        batch_support_status="serial_only",
    )


def tracker_backend_family(backend: str) -> str:
    return tracker_backend_spec(backend).family


def effective_legacy_update_mode(execution_mode: str) -> str:
    normalized = normalize_tracker_execution_mode(execution_mode)
    if normalized == TRACKER_EXECUTION_MODE_BATCH_VIEWS:
        return "batch"
    return normalized


@dataclass(frozen=True)
class PointTrackerAdapterConfig:
    backend: str = TRACKER_BACKEND_COTRACKER3
    device: str = "cuda"
    repo_dir: str | None = None
    checkpoint: str | None = None
    config_path: str | None = None
    litetracker_weights: str | None = None
    litetracker_repo_dir: str | None = None
    trackon2_checkpoint: str | None = None
    trackon2_config: str | None = None
    trackon2_repo_dir: str | None = None


def _prepend_repo_dir(repo_dir: str | None) -> None:
    if not repo_dir:
        return
    path = str(Path(repo_dir).expanduser())
    if path and path not in sys.path:
        sys.path.insert(0, path)


def build_point_tracker_adapter_factory(config: PointTrackerAdapterConfig) -> Callable[[int], PointTrackerAdapter]:
    backend = normalize_tracker_backend(config.backend)

    def factory(camera_idx: int) -> PointTrackerAdapter:
        if backend == TRACKER_BACKEND_COTRACKER3:
            from qqtt.tracking.backends.cotracker3_adapter import CoTracker3Adapter

            return CoTracker3Adapter(device=str(config.device), camera_idx=None if int(camera_idx) < 0 else int(camera_idx))
        if backend == TRACKER_BACKEND_TRACKON2:
            from qqtt.tracking.backends.trackon2_adapter import TrackOn2Adapter

            _prepend_repo_dir(config.trackon2_repo_dir or config.repo_dir)
            return TrackOn2Adapter(
                device=str(config.device),
                camera_idx=None if int(camera_idx) < 0 else int(camera_idx),
                checkpoint=config.trackon2_checkpoint or config.checkpoint,
                config_path=config.trackon2_config or config.config_path,
                repo_dir=config.trackon2_repo_dir or config.repo_dir,
            )
        from qqtt.tracking.backends.litetracker_adapter import LiteTrackerAdapter

        _prepend_repo_dir(config.litetracker_repo_dir or config.repo_dir)
        return LiteTrackerAdapter(
            device=str(config.device),
            camera_idx=None if int(camera_idx) < 0 else int(camera_idx),
            weights=config.litetracker_weights or config.checkpoint,
            repo_dir=config.litetracker_repo_dir or config.repo_dir,
        )

    return factory


class UnavailableExternalPointTrackerAdapter:
    """Explicit adapter shell for external trackers that need local repos/weights."""

    spec: PointTrackerBackendSpec
    name: str

    def __init__(
        self,
        *,
        spec: PointTrackerBackendSpec,
        device: str = "cuda",
        camera_idx: int | None = None,
        reason: str,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.device = str(device)
        self.camera_idx = camera_idx
        self._reason = str(reason)

    def availability(self) -> BackendAvailability:
        return BackendAvailability(self.name, False, self._reason)

    def is_available(self) -> bool:
        return False

    def availability_reason(self) -> str:
        return self._reason

    def _raise(self) -> None:
        raise BackendUnavailableError(self._reason)

    def warmup(self) -> dict[str, Any]:
        self._raise()

    def initialize(self, frames: list[np.ndarray], query_points_yx: np.ndarray, masks: list[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks
        self._raise()

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        self._raise()

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        _ = query_points_yx_by_camera
        self._raise()

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        _ = frames_by_camera
        self._raise()


__all__ = [
    "PointTrackerAdapter",
    "PointTrackerAdapterConfig",
    "PointTrackerBackendSpec",
    "PointTrackerBatchResult",
    "TRACKER_BACKEND_COTRACKER3",
    "TRACKER_BACKEND_LITETRACKER",
    "TRACKER_BACKEND_TRACKON2",
    "TRACKER_BACKENDS",
    "TRACKER_BATCH_QUERY_COUNT_POLICIES",
    "TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED",
    "TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON",
    "TRACKER_EXECUTION_MODE_AUTO",
    "TRACKER_EXECUTION_MODE_BATCH_VIEWS",
    "TRACKER_EXECUTION_MODE_SERIAL",
    "TRACKER_EXECUTION_MODES",
    "UnavailableExternalPointTrackerAdapter",
    "build_point_tracker_adapter_factory",
    "effective_legacy_update_mode",
    "normalize_tracker_backend",
    "normalize_tracker_batch_query_count_policy",
    "normalize_tracker_execution_mode",
    "tracker_backend_family",
    "tracker_backend_spec",
]
