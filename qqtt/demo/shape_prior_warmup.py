from __future__ import annotations

from dataclasses import dataclass, field, replace
import threading
import time
from typing import Any, Protocol

import numpy as np


SHAPE_BACKEND_SAM3D_OBJECTS = "sam3d-objects"
SHAPE_PRIOR_STATUS_DISABLED = "disabled"
SHAPE_PRIOR_STATUS_PENDING = "pending"
SHAPE_PRIOR_STATUS_READY = "ready"
SHAPE_PRIOR_STATUS_FAILED = "failed"
SHAPE_PRIOR_STATUS_UNAVAILABLE = "unavailable"
SHAPE_PRIOR_START_POLICY_ASYNC_AFTER_FIRST_STRICT_PAIR = "async-after-first-strict-pair"
SHAPE_PRIOR_START_POLICY_BLOCKING_BEFORE_FIRST_OUTPUT = "blocking-before-first-output"
SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN = "after-teardown"
SHAPE_PRIOR_START_POLICIES = (
    SHAPE_PRIOR_START_POLICY_ASYNC_AFTER_FIRST_STRICT_PAIR,
    SHAPE_PRIOR_START_POLICY_BLOCKING_BEFORE_FIRST_OUTPUT,
    SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
)
SHAPE_PRIOR_EXECUTION_REMOTE_WORKER = "remote-worker"
SHAPE_PRIOR_EXECUTION_LOCAL_SUBPROCESS = "local-subprocess"
SHAPE_PRIOR_EXECUTIONS = (
    SHAPE_PRIOR_EXECUTION_REMOTE_WORKER,
    SHAPE_PRIOR_EXECUTION_LOCAL_SUBPROCESS,
)
DEFAULT_SHAPE_PRIOR_ENDPOINT = "tcp://127.0.0.1:7100"
DEFAULT_SHAPE_PRIOR_DEVICE = "cuda:0"
DEFAULT_SHAPE_PRIOR_RENDER_RGB = (150, 150, 150)
DEFAULT_SHAPE_PRIOR_TIMEOUT_MS = 180_000


@dataclass(frozen=True)
class ShapePriorSnapshot:
    seq: int
    source_timestamp_s: float | None
    input_source: str
    depth_backend: str
    depth_source_internal: str
    rgb_u8: np.ndarray
    object_mask: np.ndarray
    controller_mask: np.ndarray
    depth_color_m: np.ndarray
    k_color: np.ndarray
    camera_to_world_c2w: np.ndarray | None
    table_z_m: float = 0.0
    table_z_above_direction: str = "positive"


@dataclass(frozen=True)
class ShapePriorResult:
    seq: int
    status: str
    points_m: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    colors_rgb_u8: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    source_timestamp_s: float | None = None
    source_seq: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return str(self.status) == SHAPE_PRIOR_STATUS_READY and len(self.points_m) > 0


class ShapePriorClient(Protocol):
    def request_shape_prior(self, snapshot: ShapePriorSnapshot) -> ShapePriorResult:
        ...


def replace_snapshot(snapshot: ShapePriorSnapshot, **changes: Any) -> ShapePriorSnapshot:
    return replace(snapshot, **changes)


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb_u8 must be HxWx3, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _as_mask(name: str, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} must match RGB/depth shape {shape}")
    return np.ascontiguousarray(arr, dtype=bool)


def validate_shape_prior_snapshot(snapshot: ShapePriorSnapshot) -> None:
    rgb = _as_rgb_u8(snapshot.rgb_u8)
    shape = tuple(int(v) for v in rgb.shape[:2])
    depth = np.asarray(snapshot.depth_color_m, dtype=np.float32)
    if depth.shape != shape:
        raise ValueError(f"depth_color_m shape {depth.shape} must match RGB shape {shape}")
    if not np.isfinite(depth).any():
        raise ValueError("depth_color_m contains no finite values")
    object_mask = _as_mask("object mask", snapshot.object_mask, shape)
    _as_mask("controller mask", snapshot.controller_mask, shape)
    if int(np.count_nonzero(object_mask)) <= 0:
        raise ValueError("shape-prior snapshot requires a non-empty object mask")
    k_color = np.asarray(snapshot.k_color, dtype=np.float32)
    if k_color.shape != (3, 3):
        raise ValueError(f"k_color must be 3x3, got {k_color.shape}")
    if snapshot.camera_to_world_c2w is None:
        raise ValueError("shape-prior snapshot requires camera_to_world_c2w")
    c2w = np.asarray(snapshot.camera_to_world_c2w, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world_c2w must be 4x4, got {c2w.shape}")
    if str(snapshot.table_z_above_direction) not in {"positive", "negative"}:
        raise ValueError("shape-prior snapshot table_z_above_direction must be positive or negative")


def normalize_snapshot(snapshot: ShapePriorSnapshot) -> ShapePriorSnapshot:
    validate_shape_prior_snapshot(snapshot)
    rgb = _as_rgb_u8(snapshot.rgb_u8)
    shape = tuple(int(v) for v in rgb.shape[:2])
    return ShapePriorSnapshot(
        seq=int(snapshot.seq),
        source_timestamp_s=(
            None if snapshot.source_timestamp_s is None else float(snapshot.source_timestamp_s)
        ),
        input_source=str(snapshot.input_source),
        depth_backend=str(snapshot.depth_backend),
        depth_source_internal=str(snapshot.depth_source_internal),
        rgb_u8=rgb,
        object_mask=_as_mask("object mask", snapshot.object_mask, shape),
        controller_mask=_as_mask("controller mask", snapshot.controller_mask, shape),
        depth_color_m=np.ascontiguousarray(snapshot.depth_color_m, dtype=np.float32),
        k_color=np.ascontiguousarray(snapshot.k_color, dtype=np.float32).reshape(3, 3),
        camera_to_world_c2w=np.ascontiguousarray(snapshot.camera_to_world_c2w, dtype=np.float32).reshape(4, 4),
        table_z_m=float(snapshot.table_z_m),
        table_z_above_direction=str(snapshot.table_z_above_direction),
    )


def default_profile(*, enabled: bool) -> dict[str, Any]:
    status = SHAPE_PRIOR_STATUS_PENDING if bool(enabled) else SHAPE_PRIOR_STATUS_DISABLED
    return {
        "shape_prior_enabled": bool(enabled),
        "shape_prior_status": status,
        "shape_backend": SHAPE_BACKEND_SAM3D_OBJECTS if bool(enabled) else None,
        "input_source": None,
        "depth_backend": None,
        "depth_source_internal": None,
        "shape_prior_source_seq": None,
        "shape_prior_source_time_s": None,
        "shape_prior_ready_seq": None,
        "first_rgb_ms": 0.0,
        "first_depth_ms": 0.0,
        "first_mask_ms": 0.0,
        "first_strict_pair_ms": 0.0,
        "snapshot_copy_ms": 0.0,
        "snapshot_write_ms": 0.0,
        "request_upload_ms": 0.0,
        "worker_queue_ms": 0.0,
        "sam3d_model_load_ms": 0.0,
        "image_upscale_ms": 0.0,
        "mask_refinement_ms": 0.0,
        "sam3d_inference_ms": 0.0,
        "geometry_export_ms": 0.0,
        "single_view_alignment_ms": 0.0,
        "sampling_ms": 0.0,
        "response_download_ms": 0.0,
        "shape_prior_total_ms": 0.0,
        "time_to_first_track_ms": 0.0,
        "time_to_first_render_ms": 0.0,
        "time_to_shape_prior_ready_ms": 0.0,
        "shape_prior_error": None,
    }


class ShapePriorWarmupManager:
    def __init__(
        self,
        *,
        enabled: bool,
        client: ShapePriorClient | None,
        start_policy: str = SHAPE_PRIOR_START_POLICY_ASYNC_AFTER_FIRST_STRICT_PAIR,
        created_perf_s: float | None = None,
    ) -> None:
        if start_policy not in SHAPE_PRIOR_START_POLICIES:
            raise ValueError(f"unsupported shape-prior start policy: {start_policy}")
        self.enabled = bool(enabled)
        self.client = client
        self.start_policy = str(start_policy)
        self.created_perf_s = time.perf_counter() if created_perf_s is None else float(created_perf_s)
        self._lock = threading.Lock()
        self._status = SHAPE_PRIOR_STATUS_PENDING if self.enabled else SHAPE_PRIOR_STATUS_DISABLED
        self._submitted = False
        self._result: ShapePriorResult | None = None
        self._profile = default_profile(enabled=self.enabled)
        self._thread: threading.Thread | None = None

    @property
    def status(self) -> str:
        with self._lock:
            return str(self._status)

    @staticmethod
    def _snapshot_profile_fields(snapshot: ShapePriorSnapshot) -> dict[str, Any]:
        return {
            "input_source": str(snapshot.input_source),
            "depth_backend": str(snapshot.depth_backend),
            "depth_source_internal": str(snapshot.depth_source_internal),
            "shape_prior_source_seq": int(snapshot.seq),
            "shape_prior_source_time_s": snapshot.source_timestamp_s,
            "table_z_m": float(snapshot.table_z_m),
            "table_z_above_direction": str(snapshot.table_z_above_direction),
        }

    def maybe_submit(self, snapshot: ShapePriorSnapshot) -> bool:
        if not self.enabled:
            return False
        normalized = normalize_snapshot(snapshot)
        with self._lock:
            if self._submitted:
                return False
            self._submitted = True
            self._status = SHAPE_PRIOR_STATUS_PENDING
            self._profile.update(self._snapshot_profile_fields(normalized))
        if self.client is None:
            self._mark_failed(normalized, "shape-prior client is unavailable")
            return True
        thread = threading.Thread(target=self._run_request, args=(normalized,), daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        return True

    def wait(self, timeout_s: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=None if timeout_s is None else float(timeout_s))

    def _run_request(self, snapshot: ShapePriorSnapshot) -> None:
        start_s = time.perf_counter()
        try:
            assert self.client is not None
            result = self.client.request_shape_prior(snapshot)
            metadata = dict(result.metadata)
            metadata.setdefault("shape_prior_total_ms", (time.perf_counter() - start_s) * 1000.0)
            result = replace(result, metadata=metadata)
            self._mark_result(snapshot, result)
        except Exception as exc:
            self._mark_failed(snapshot, str(exc))

    def _mark_result(self, snapshot: ShapePriorSnapshot, result: ShapePriorResult) -> None:
        status = SHAPE_PRIOR_STATUS_READY if result.ready else str(result.status or SHAPE_PRIOR_STATUS_FAILED)
        with self._lock:
            self._status = status
            self._result = result if status == SHAPE_PRIOR_STATUS_READY else None
            self._profile.update(dict(result.metadata))
            self._profile.update(self._snapshot_profile_fields(snapshot))
            self._profile.update(
                {
                    "shape_prior_status": status,
                    "shape_prior_ready_seq": int(result.seq),
                    "time_to_shape_prior_ready_ms": (
                        time.perf_counter() - float(self.created_perf_s)
                    )
                    * 1000.0,
                    "shape_prior_error": result.error,
                }
            )

    def _mark_failed(self, snapshot: ShapePriorSnapshot, error: str) -> None:
        with self._lock:
            self._status = SHAPE_PRIOR_STATUS_FAILED
            self._result = None
            self._profile.update(self._snapshot_profile_fields(snapshot))
            self._profile.update(
                {
                    "shape_prior_status": SHAPE_PRIOR_STATUS_FAILED,
                    "shape_prior_error": str(error),
                }
            )

    def ready_result(self) -> ShapePriorResult | None:
        with self._lock:
            return self._result

    def profile(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._profile)
            payload["shape_prior_status"] = str(self._status)
            return payload
