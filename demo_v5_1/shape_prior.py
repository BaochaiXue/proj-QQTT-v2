from __future__ import annotations

from dataclasses import dataclass, field, replace
import io
import json
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np


SHAPE_BACKEND_SAM3D_OBJECTS = "sam3d-objects"
SHAPE_PRIOR_STATUS_DISABLED = "disabled"
SHAPE_PRIOR_STATUS_PENDING = "pending"
SHAPE_PRIOR_STATUS_READY = "ready"
SHAPE_PRIOR_STATUS_FAILED = "failed"
DEFAULT_SHAPE_PRIOR_ENDPOINT = "tcp://127.0.0.1:7100"
DEFAULT_SHAPE_PRIOR_DEVICE = "cuda:0"
DEFAULT_SHAPE_PRIOR_RENDER_RGB = (150, 150, 150)
DEFAULT_SHAPE_PRIOR_TIMEOUT_MS = 180_000


@dataclass(frozen=True)
class ShapePriorFrame0Request:
    seq: int
    source_timestamp_s: float | None
    input_source: str
    depth_backend: str
    depth_source_internal: str
    rgb_u8: np.ndarray
    object_mask: np.ndarray
    object_observation_mask: np.ndarray | None
    controller_mask: np.ndarray
    depth_color_m: np.ndarray
    k_color: np.ndarray
    camera_to_world_c2w: np.ndarray
    table_z_m: float = 0.0
    table_z_above_direction: str = "negative"


@dataclass(frozen=True)
class ShapePriorResult:
    seq: int
    status: str
    points_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    colors_rgb_u8: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.uint8)
    )
    surface_points_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    interior_points_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    source_timestamp_s: float | None = None
    source_seq: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.status == SHAPE_PRIOR_STATUS_READY and len(self.points_m) > 0


@dataclass(frozen=True)
class ShapePriorSamples:
    surface_points_m: np.ndarray
    interior_points_m: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _empty_points(dtype: np.dtype | type = np.float32) -> np.ndarray:
    return np.empty((0, 3), dtype=dtype)


def _points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return _empty_points()
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)


def _metadata_json(metadata: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(metadata, sort_keys=True))


def _read_metadata(value: np.ndarray) -> dict[str, Any]:
    return json.loads(str(np.asarray(value).item()))


def _npz_bytes(**arrays: Any) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _load_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pack_shape_prior_request(frame0: ShapePriorFrame0Request) -> list[bytes]:
    observation_mask = frame0.object_observation_mask
    if observation_mask is None:
        observation_mask = frame0.object_mask
    metadata = {
        "request_id": f"shape-prior-{int(frame0.seq)}-{uuid4().hex[:8]}",
        "seq": int(frame0.seq),
        "source_timestamp_s": frame0.source_timestamp_s,
        "input_source": str(frame0.input_source),
        "depth_backend": str(frame0.depth_backend),
        "depth_source_internal": str(frame0.depth_source_internal),
        "table_z_m": float(frame0.table_z_m),
        "table_z_above_direction": str(frame0.table_z_above_direction),
        "shape_backend": SHAPE_BACKEND_SAM3D_OBJECTS,
    }
    return [
        _npz_bytes(
            metadata_json=_metadata_json(metadata),
            rgb_u8=frame0.rgb_u8,
            object_mask=frame0.object_mask,
            object_observation_mask=observation_mask,
            controller_mask=frame0.controller_mask,
            depth_color_m=frame0.depth_color_m,
            k_color=frame0.k_color,
            camera_to_world_c2w=frame0.camera_to_world_c2w,
        )
    ]


def unpack_shape_prior_request(parts: list[bytes]) -> ShapePriorFrame0Request:
    if len(parts) != 1:
        raise ValueError(f"shape-prior request expected 1 npz frame, got {len(parts)}")
    data = _load_npz(parts[0])
    metadata = _read_metadata(data["metadata_json"])
    return ShapePriorFrame0Request(
        seq=int(metadata.get("seq", -1)),
        source_timestamp_s=metadata.get("source_timestamp_s"),
        input_source=str(metadata.get("input_source", "")),
        depth_backend=str(metadata.get("depth_backend", "")),
        depth_source_internal=str(metadata.get("depth_source_internal", "")),
        rgb_u8=data["rgb_u8"],
        object_mask=data["object_mask"],
        object_observation_mask=data["object_observation_mask"],
        controller_mask=data["controller_mask"],
        depth_color_m=data["depth_color_m"],
        k_color=data["k_color"],
        camera_to_world_c2w=data["camera_to_world_c2w"],
        table_z_m=float(metadata.get("table_z_m", 0.0)),
        table_z_above_direction=str(metadata.get("table_z_above_direction", "negative")),
    )


def pack_shape_prior_result(result: ShapePriorResult) -> list[bytes]:
    metadata = dict(result.metadata)
    metadata.update(
        {
            "seq": int(result.seq),
            "status": str(result.status),
            "error": result.error,
            "source_seq": result.source_seq,
            "source_timestamp_s": result.source_timestamp_s,
            "point_count": int(len(result.points_m)),
            "surface_point_count": int(len(result.surface_points_m)),
            "interior_point_count": int(len(result.interior_points_m)),
        }
    )
    return [
        _npz_bytes(
            metadata_json=_metadata_json(metadata),
            points_m=result.points_m,
            colors_rgb_u8=result.colors_rgb_u8,
            surface_points_m=result.surface_points_m,
            interior_points_m=result.interior_points_m,
        )
    ]


def unpack_shape_prior_result(parts: list[bytes]) -> ShapePriorResult:
    if len(parts) != 1:
        raise ValueError(f"shape-prior response expected 1 npz frame, got {len(parts)}")
    data = _load_npz(parts[0])
    metadata = _read_metadata(data["metadata_json"])
    return ShapePriorResult(
        seq=int(metadata.get("seq", -1)),
        source_seq=metadata.get("source_seq"),
        source_timestamp_s=metadata.get("source_timestamp_s"),
        status=str(metadata.get("status", SHAPE_PRIOR_STATUS_FAILED)),
        points_m=_points(data.get("points_m", _empty_points())),
        colors_rgb_u8=np.asarray(data.get("colors_rgb_u8", _empty_points(np.uint8))),
        surface_points_m=_points(data.get("surface_points_m", _empty_points())),
        interior_points_m=_points(data.get("interior_points_m", _empty_points())),
        metadata=metadata,
        error=metadata.get("error"),
    )


def default_profile(*, enabled: bool) -> dict[str, Any]:
    return {
        "shape_prior_enabled": bool(enabled),
        "shape_prior_status": (
            SHAPE_PRIOR_STATUS_PENDING if enabled else SHAPE_PRIOR_STATUS_DISABLED
        ),
        "shape_backend": SHAPE_BACKEND_SAM3D_OBJECTS if enabled else None,
        "shape_prior_error": None,
    }


class ShapePriorRemoteClient:
    def __init__(self, *, endpoint: str, timeout_ms: int) -> None:
        self.endpoint = str(endpoint)
        self.timeout_ms = int(timeout_ms)
        self._socket: Any | None = None

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None

    def _connect(self) -> Any:
        if self._socket is not None:
            return self._socket
        import zmq

        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket
        return socket

    def request_shape_prior(self, frame0: ShapePriorFrame0Request) -> ShapePriorResult:
        socket = self._connect()
        start_s = time.perf_counter()
        try:
            socket.send_multipart(pack_shape_prior_request(frame0))
            result = unpack_shape_prior_result(socket.recv_multipart())
        except Exception:
            self.close()
            raise
        metadata = dict(result.metadata)
        metadata.setdefault("response_download_ms", (time.perf_counter() - start_s) * 1000.0)
        return replace(
            result,
            source_seq=int(frame0.seq),
            source_timestamp_s=frame0.source_timestamp_s,
            metadata=metadata,
        )


class ShapePriorWarmupManager:
    def __init__(self, *, enabled: bool, client: Any | None) -> None:
        self.enabled = bool(enabled)
        self.client = client
        self.created_perf_s = time.perf_counter()
        self._lock = threading.Lock()
        self._submitted = False
        self._result: ShapePriorResult | None = None
        self._profile = default_profile(enabled=self.enabled)
        self._thread: threading.Thread | None = None

    def maybe_submit(self, frame0: ShapePriorFrame0Request) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._submitted:
                return False
            self._submitted = True
            self._profile.update(
                {
                    "shape_prior_status": SHAPE_PRIOR_STATUS_PENDING,
                    "shape_prior_source_seq": int(frame0.seq),
                    "shape_prior_source_time_s": frame0.source_timestamp_s,
                    "shape_prior_submit_ms": (
                        time.perf_counter() - self.created_perf_s
                    )
                    * 1000.0,
                }
            )
        thread = threading.Thread(target=self._run, args=(frame0,), daemon=True)
        self._thread = thread
        thread.start()
        return True

    def _run(self, frame0: ShapePriorFrame0Request) -> None:
        try:
            if self.client is None:
                raise RuntimeError("shape-prior client is unavailable")
            result = self.client.request_shape_prior(frame0)
            status = SHAPE_PRIOR_STATUS_READY if result.ready else result.status
            with self._lock:
                self._result = result if result.ready else None
                self._profile.update(result.metadata)
                self._profile.update(
                    {
                        "shape_prior_status": status,
                        "shape_prior_ready_seq": int(result.seq),
                        "shape_prior_error": result.error,
                        "time_to_shape_prior_ready_ms": (
                            time.perf_counter() - self.created_perf_s
                        )
                        * 1000.0,
                    }
                )
        except Exception as exc:
            with self._lock:
                self._profile.update(
                    {
                        "shape_prior_status": SHAPE_PRIOR_STATUS_FAILED,
                        "shape_prior_error": str(exc),
                    }
                )

    def wait(self, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout_s)

    def ready_result(self) -> ShapePriorResult | None:
        with self._lock:
            return self._result

    def profile(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._profile)


def observation_points_world(frame0: ShapePriorFrame0Request, *, max_points: int) -> np.ndarray:
    mask = frame0.object_observation_mask
    if mask is None:
        mask = frame0.object_mask
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(frame0.depth_color_m, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > np.float32(0.0))
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return _empty_points()
    z = depth[rows, cols]
    k = np.asarray(frame0.k_color, dtype=np.float32).reshape(3, 3)
    x = (cols.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (rows.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    camera_points = np.stack([x, y, z], axis=1)
    c2w = np.asarray(frame0.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)
    world = (c2w @ np.c_[camera_points, np.ones(len(camera_points))].T).T[:, :3]
    return _take_evenly(_points(world), max_points)


def _take_evenly(points: np.ndarray, count: int) -> np.ndarray:
    points = _points(points)
    if count <= 0 or len(points) <= count:
        return points
    return points[np.linspace(0, len(points) - 1, count, dtype=np.int64)]


def align_mesh_to_observation(
    mesh: Any, observation_points: np.ndarray
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    canonical = _take_evenly(np.asarray(mesh.vertices), len(observation_points))
    observation = _points(observation_points)
    if len(canonical) < 3 or len(observation) < 3:
        raise RuntimeError("shape prior alignment requires at least 3 points")
    can_center = np.mean(canonical, axis=0)
    obs_center = np.mean(observation, axis=0)
    can_centered = canonical - can_center
    obs_centered = observation - obs_center
    can_basis = np.linalg.svd(can_centered, full_matrices=False)[2].T
    obs_basis = np.linalg.svd(obs_centered, full_matrices=False)[2].T
    rotation = obs_basis @ can_basis.T
    if np.linalg.det(rotation) < 0.0:
        obs_basis[:, -1] *= -1.0
        rotation = obs_basis @ can_basis.T
    can_radius = float(np.sqrt(np.mean(np.sum(can_centered * can_centered, axis=1))))
    obs_radius = float(np.sqrt(np.mean(np.sum(obs_centered * obs_centered, axis=1))))
    scale = obs_radius / max(can_radius, 1e-6)
    aligned = mesh.copy()
    vertices = np.asarray(aligned.vertices, dtype=np.float32)
    aligned.vertices = scale * ((vertices - can_center) @ rotation.T) + obs_center
    metadata = {"single_view_alignment_ms": 0.0, "shape_prior_align_scale": scale}
    return aligned, _points(aligned.vertices), metadata


def sample_shape_prior_points(
    mesh: Any,
    reference_points_m: np.ndarray,
    *,
    target_surface_points: int = 700,
    target_interior_points: int = 1000,
    max_dist_m: float = 0.05,
) -> ShapePriorSamples:
    from scipy.spatial import cKDTree
    import trimesh

    reference = _points(reference_points_m)
    if len(reference) == 0:
        raise RuntimeError("shape prior sampling requires observation points")
    tree = cKDTree(reference)

    def near(points: np.ndarray, count: int) -> np.ndarray:
        candidates = _points(points)
        if len(candidates) == 0:
            return candidates
        dist, _ = tree.query(candidates, k=1)
        return _take_evenly(candidates[dist <= float(max_dist_m)], count)

    surface, _ = trimesh.sample.sample_surface(mesh, max(4096, target_surface_points * 8))
    surface_points = near(surface, target_surface_points)
    try:
        interior = trimesh.sample.volume_mesh(mesh, max(4096, target_interior_points * 8))
    except Exception:
        interior = _empty_points()
    interior_points = near(interior, target_interior_points)
    return ShapePriorSamples(
        surface_points_m=surface_points,
        interior_points_m=interior_points,
        metadata={
            "shape_prior_sampling_backend": "sam3d-single-view",
            "shape_prior_surface_points": int(len(surface_points)),
            "shape_prior_interior_points": int(len(interior_points)),
        },
    )
