from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_TRACKON2,
    PointTrackerBackendSpec,
)


class TrackOn2Adapter:
    """Live Track-On2 adapter over the external `gorkaydemir/track_on` repo.

    Track-On2's public streaming API is frame-by-frame for one sequence. For
    Demo 3.1 batch-view calls we keep one shared model and swap the lightweight
    per-camera streaming state around it. That avoids three copies of the
    DINOv3/Track-On2 weights while still producing one tracker result per camera
    input group.
    """

    name = TRACKER_BACKEND_TRACKON2
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_TRACKON2,
        family="trackon",
        supports_batch_views=True,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="single_model_state_swap",
    )
    _STATE_ATTRS = (
        "t",
        "point_memory",
        "temporal_mask",
        "q_init",
        "N",
        "capacity",
        "initial_capacity",
        "H",
        "W",
        "device",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        checkpoint: str | None = None,
        config_path: str | None = None,
        repo_dir: str | None = None,
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.checkpoint = str(Path(checkpoint).expanduser()) if checkpoint else None
        self.config_path = str(Path(config_path).expanduser()) if config_path else None
        self.repo_dir = str(Path(repo_dir).expanduser()) if repo_dir else None
        self._model: Any | None = None
        self._model_load_ms = 0.0
        self._stream_query_points_yx: np.ndarray | None = None
        self._stream_state: dict[str, Any] | None = None
        self._stream_total_frames = 0
        self._stream_camera_idx: int | None = camera_idx
        self._batch_camera_ids: tuple[int, ...] = ()
        self._batch_query_points_yx_by_camera: dict[int, np.ndarray] = {}
        self._batch_states_by_camera: dict[int, dict[str, Any] | None] = {}
        self._batch_frame_counts_by_camera: dict[int, int] = {}

    def availability(self) -> BackendAvailability:
        missing: list[str] = []
        if not self.checkpoint:
            missing.append("--trackon2-checkpoint")
        elif not Path(self.checkpoint).is_file():
            missing.append(f"--trackon2-checkpoint {self.checkpoint!r} does not exist")
        if self.config_path and not Path(self.config_path).is_file():
            missing.append(f"--trackon2-config {self.config_path!r} does not exist")
        if self.repo_dir and not Path(self.repo_dir).is_dir():
            missing.append(f"--trackon2-repo-dir {self.repo_dir!r} does not exist")
        if missing:
            return BackendAvailability(self.name, False, "; ".join(missing))
        try:
            import torch  # noqa: F401
            from model.trackon_predictor import Predictor  # noqa: F401
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                f"Track-On2 runtime import failed: {type(exc).__name__}: {exc}",
            )
        return BackendAvailability(self.name, True, "Track-On2 Predictor import and checkpoint path are available")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        import torch
        from model.trackon_predictor import Predictor

        model_args = None
        if self.config_path:
            from utils.train_utils import load_args_from_yaml

            model_args = load_args_from_yaml(self.config_path)
        started_s = time.perf_counter()
        model = Predictor(model_args, checkpoint_path=self.checkpoint, support_grid_size=0).to(self.device).eval()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._model_load_ms = float((time.perf_counter() - started_s) * 1000.0)
        self._model = model
        return model

    def warmup(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        self._load_model()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "batch_support_status": self.spec.batch_support_status,
        }

    @staticmethod
    def _validate_query_points(query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            prefix = "query_points_yx" if camera_idx is None else f"query_points_yx for camera {camera_idx}"
            raise ValueError(f"{prefix} must have shape (N,2); got {points.shape}")
        if len(points) == 0:
            raise ValueError("Track-On2 requires at least one query point.")
        return np.ascontiguousarray(points)

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray, *, device: str):
        import torch

        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Track-On2 frame must be HxWx3 RGB uint8; got {arr.shape}")
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].contiguous().float().to(device)

    @staticmethod
    def _queries_yx_to_xy_tensor(query_points_yx: np.ndarray, *, device: str):
        import torch

        points_yx = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
        points_xy = np.ascontiguousarray(points_yx[:, ::-1])
        return torch.from_numpy(points_xy).float().to(device)

    def _capture_state(self) -> dict[str, Any]:
        model = self._load_model()
        return {name: getattr(model, name) for name in self._STATE_ATTRS}

    def _restore_state(self, state: dict[str, Any] | None) -> None:
        model = self._load_model()
        if state is None:
            model.reset()
            return
        for name, value in state.items():
            setattr(model, name, value)

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        _ = masks
        self._load_model()
        self._stream_query_points_yx = self._validate_query_points(query_points_yx, camera_idx=self.camera_idx)
        self._stream_total_frames = 0
        self._stream_state = None
        for frame in frames:
            self.update(frame)

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self._stream_camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        self._load_model()
        query_points: dict[int, np.ndarray] = {}
        for camera_idx, points in sorted(query_points_yx_by_camera.items()):
            query_points[int(camera_idx)] = self._validate_query_points(points, camera_idx=int(camera_idx))
        if not query_points:
            raise ValueError("Track-On2 batch stream requires at least one camera.")
        self._batch_camera_ids = tuple(sorted(query_points))
        self._batch_query_points_yx_by_camera = query_points
        self._batch_states_by_camera = {int(camera_idx): None for camera_idx in self._batch_camera_ids}
        self._batch_frame_counts_by_camera = {int(camera_idx): 0 for camera_idx in self._batch_camera_ids}

    def _update_stateful_camera(
        self,
        *,
        camera_idx: int | None,
        frame: np.ndarray,
        query_points_yx: np.ndarray,
        state: dict[str, Any] | None,
        frame_count: int,
    ) -> tuple[TrackingResult, dict[str, Any], int, float]:
        import torch

        model = self._load_model()
        self._restore_state(state)
        frame_tensor = self._frame_to_tensor(frame, device=self.device)
        new_queries = self._queries_yx_to_xy_tensor(query_points_yx, device=self.device) if state is None else None
        started_s = time.perf_counter()
        with torch.no_grad():
            points_xy, visibility = model.forward_frame(frame_tensor, new_queries=new_queries)
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        run_ms = float((time.perf_counter() - started_s) * 1000.0)
        next_state = self._capture_state()
        frame_idx = int(frame_count)
        tracks_xy = points_xy.detach().cpu().numpy().astype(np.float32)
        visibility_np = visibility.detach().cpu().numpy().astype(np.float32).reshape(-1)
        tracks_yx = tracks_xy[:, ::-1][None, :, :].astype(np.float32)
        visibility_t = visibility_np[None, :].astype(np.float32)
        result = TrackingResult(
            tracks_yx=tracks_yx,
            visibility=visibility_t,
            backend=self.name,
            camera_idx=camera_idx,
            query_points_yx=query_points_yx,
            stats={
                "backend": self.name,
                "tracker_backend": self.name,
                "adapter": type(self).__name__,
                "mode": "trackon2_forward_frame",
                "stream_status": "published",
                "chunk_start_idx": frame_idx,
                "chunk_end_idx": frame_idx,
                "frames_seen": frame_idx + 1,
                "num_query_points": int(len(query_points_yx)),
                "model_run_ms": float(run_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                "device": self.device,
            },
        )
        return result, next_state, frame_idx + 1, run_ms

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._stream_query_points_yx is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        result, state, frame_count, _run_ms = self._update_stateful_camera(
            camera_idx=self._stream_camera_idx,
            frame=frame,
            query_points_yx=self._stream_query_points_yx,
            state=self._stream_state,
            frame_count=self._stream_total_frames,
        )
        self._stream_state = state
        self._stream_total_frames = frame_count
        result.stats["update_mode"] = "serial"
        return result

    def update_camera(self, camera_idx: int, frame_rgb: np.ndarray) -> TrackingResult:
        self.camera_idx = int(camera_idx)
        self._stream_camera_idx = int(camera_idx)
        return self.update(frame_rgb)

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        if not self._batch_camera_ids:
            raise RuntimeError("Call initialize_batch(...) before update_batch().")
        results: dict[int, TrackingResult] = {}
        per_camera_ms: dict[int, float] = {}
        total_started_s = time.perf_counter()
        for camera_idx in self._batch_camera_ids:
            idx = int(camera_idx)
            result, state, frame_count, run_ms = self._update_stateful_camera(
                camera_idx=idx,
                frame=frames_by_camera[idx],
                query_points_yx=self._batch_query_points_yx_by_camera[idx],
                state=self._batch_states_by_camera.get(idx),
                frame_count=int(self._batch_frame_counts_by_camera.get(idx, 0)),
            )
            self._batch_states_by_camera[idx] = state
            self._batch_frame_counts_by_camera[idx] = frame_count
            results[idx] = result
            per_camera_ms[idx] = float(run_ms)
        total_ms = float((time.perf_counter() - total_started_s) * 1000.0)
        for idx, result in results.items():
            result.stats.update(
                {
                    "update_mode": "batch",
                    "batch_size": int(len(self._batch_camera_ids)),
                    "batch_camera_ids": [int(item) for item in self._batch_camera_ids],
                    "batch_impl": "trackon2_single_model_state_swap",
                    "model_run_ms": float(total_ms),
                    "per_camera_model_run_ms": dict(per_camera_ms),
                }
            )
        return results


__all__ = ["TrackOn2Adapter"]
