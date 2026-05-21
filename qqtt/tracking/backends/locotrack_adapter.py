from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
import os
import sys
import time
from typing import Any

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_LOCOTRACK,
    PointTrackerBackendSpec,
)


INSTALL_HINT = "Run scripts/env/install_locotrack_s_demo_3_1_max.sh"


class LocoTrackAdapter:
    """Windowed LocoTrack-S adapter for Demo 3.1 point tracking.

    LocoTrack is not a frame-by-frame online tracker. This adapter keeps a
    rolling RGB window and republishes the latest frame's 2D tracks through the
    existing QQTT point-tracker contract. Depth, intrinsics, camera poses, and
    world lift stay in the Demo 3.1 main process.
    """

    name = TRACKER_BACKEND_LOCOTRACK
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_LOCOTRACK,
        family="locotrack",
        supports_batch_views=True,
        supports_online=False,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="windowed_batch_views",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        repo_dir: str | None = None,
        checkpoint: str | None = None,
        model_size: str = "small",
        window_frames: int = 8,
        resolution: tuple[int, int] | Sequence[int] | str = (256, 256),
        query_chunk_size: int = 256,
        autocast_dtype: str = "bf16",
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.repo_dir = self._expand_path(
            repo_dir
            or os.environ.get("QQTT_LOCOTRACK_REPO_DIR")
            or os.environ.get("LOCOTRACK_REPO_DIR")
        )
        self.checkpoint = self._expand_path(
            checkpoint
            or os.environ.get("QQTT_LOCOTRACK_CHECKPOINT")
            or os.environ.get("LOCOTRACK_CHECKPOINT")
        )
        self.model_size = self._normalize_model_size(model_size)
        self.window_frames = self._normalize_positive_int(window_frames, "locotrack_window_frames")
        self.resolution = self._normalize_resolution(resolution)
        self.query_chunk_size = self._normalize_positive_int(query_chunk_size, "locotrack_query_chunk_size")
        self.autocast_dtype = self._normalize_autocast_dtype(autocast_dtype)
        self._model: Any | None = None
        self._model_load_ms = 0.0
        self._query_points_yx: np.ndarray | None = None
        self._frame_buffer: list[np.ndarray] = []
        self._stream_total_frames = 0
        self._batch_camera_ids: tuple[int, ...] = ()
        self._batch_query_points_yx_by_camera: dict[int, np.ndarray] = {}
        self._batch_frame_buffers_by_camera: dict[int, list[np.ndarray]] = {}
        self._batch_total_frames = 0

    @staticmethod
    def _expand_path(value: str | None) -> str | None:
        return str(Path(value).expanduser()) if value else None

    @staticmethod
    def _normalize_positive_int(value: int, name: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        return parsed

    @staticmethod
    def _normalize_model_size(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"small", "base"}:
            raise ValueError("--locotrack-model-size must be one of {'small', 'base'}.")
        return normalized

    @staticmethod
    def _normalize_autocast_dtype(value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {"float16": "fp16", "half": "fp16", "float32": "fp32", "bfloat16": "bf16"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"bf16", "fp16", "fp32"}:
            raise ValueError("--locotrack-autocast-dtype must be one of {'bf16', 'fp16', 'fp32'}.")
        return normalized

    @staticmethod
    def _normalize_resolution(value: tuple[int, int] | Sequence[int] | str) -> tuple[int, int]:
        if isinstance(value, str):
            raw = value.strip().lower().replace("x", ",")
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if len(parts) == 1:
                height = width = int(parts[0])
            elif len(parts) == 2:
                height, width = (int(parts[0]), int(parts[1]))
            else:
                raise ValueError("--locotrack-resolution must be HxW, H,W, or a single square size.")
        else:
            items = tuple(int(item) for item in value)
            if len(items) == 1:
                height = width = items[0]
            elif len(items) == 2:
                height, width = items
            else:
                raise ValueError("--locotrack-resolution must contain one or two integers.")
        if height <= 0 or width <= 0:
            raise ValueError("--locotrack-resolution dimensions must be positive.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("--locotrack-resolution dimensions must be multiples of 8.")
        return (int(height), int(width))

    @staticmethod
    def _resolve_pytorch_repo_dir(repo_dir: str | None) -> Path | None:
        if not repo_dir:
            return None
        path = Path(repo_dir).expanduser()
        if (path / "models" / "locotrack_model.py").is_file():
            return path
        nested = path / "locotrack_pytorch"
        if (nested / "models" / "locotrack_model.py").is_file():
            return nested
        return path

    def _prepend_repo_dir(self) -> Path | None:
        repo = self._resolve_pytorch_repo_dir(self.repo_dir)
        if repo is None:
            return None
        path = str(repo)
        if path not in sys.path:
            sys.path.insert(0, path)
        return repo

    def availability(self) -> BackendAvailability:
        if self._model is not None:
            return BackendAvailability(self.name, True, "Injected LocoTrack model is available")
        missing: list[str] = []
        repo = self._resolve_pytorch_repo_dir(self.repo_dir)
        if repo is None:
            missing.append("--locotrack-repo-dir")
        elif not (repo / "models" / "locotrack_model.py").is_file():
            missing.append(
                f"--locotrack-repo-dir {self.repo_dir!r} must point to locotrack_pytorch "
                "or a parent containing locotrack_pytorch"
            )
        if not self.checkpoint:
            missing.append("--locotrack-checkpoint")
        elif not Path(self.checkpoint).is_file():
            missing.append(f"--locotrack-checkpoint {self.checkpoint!r} does not exist")
        if missing:
            return BackendAvailability(self.name, False, "; ".join(missing) + f". {INSTALL_HINT}.")
        self._prepend_repo_dir()
        try:
            import torch  # noqa: F401
            from models.locotrack_model import load_model  # noqa: F401
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                f"LocoTrack runtime import failed: {type(exc).__name__}: {exc}. {INSTALL_HINT}.",
            )
        return BackendAvailability(self.name, True, "LocoTrack import and checkpoint path are available")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def is_initialized(self) -> bool:
        return self._query_points_yx is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        import torch
        from models.locotrack_model import load_model

        assert self.checkpoint is not None
        started_s = time.perf_counter()
        model = load_model(str(self.checkpoint), model_size=self.model_size)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self._sync_cuda_if_needed()
        self._model_load_ms = float((time.perf_counter() - started_s) * 1000.0)
        self._model = model
        return model

    def _sync_cuda_if_needed(self) -> None:
        if not str(self.device).startswith("cuda"):
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    def warmup(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        self._load_model()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "model_size": self.model_size,
            "window_frames": int(self.window_frames),
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "query_chunk_size": int(self.query_chunk_size),
            "autocast_dtype": self.autocast_dtype,
            "batch_support_status": self.spec.batch_support_status,
        }

    @staticmethod
    def _validate_query_points(query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            prefix = "query_points_yx" if camera_idx is None else f"query_points_yx for camera {camera_idx}"
            raise ValueError(f"{prefix} must have shape (N,2); got {points.shape}")
        if len(points) == 0:
            raise ValueError("LocoTrack requires at least one query point.")
        return np.ascontiguousarray(points)

    @staticmethod
    def _validate_frame(frame: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            prefix = "frame" if camera_idx is None else f"frame for camera {camera_idx}"
            raise ValueError(f"LocoTrack {prefix} must be HxWx3 RGB uint8; got {arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _queries_yx_to_tyx(query_points_yx: np.ndarray) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
        t = np.zeros((len(points), 1), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([t, points], axis=1))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            return np.asarray(numpy())
        return np.asarray(value)

    @staticmethod
    def _extract_output(output: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(output, Mapping):
            if "tracks" not in output or "occlusion" not in output:
                raise ValueError("LocoTrack inference output must contain 'tracks' and 'occlusion'.")
            tracks = output["tracks"]
            occlusion = output["occlusion"]
        elif isinstance(output, tuple) and len(output) >= 2:
            tracks, occlusion = output[0], output[1]
        else:
            tracks = getattr(output, "tracks", None)
            occlusion = getattr(output, "occlusion", None)
            if tracks is None or occlusion is None:
                raise ValueError("LocoTrack inference output must expose tracks and occlusion.")
        return LocoTrackAdapter._to_numpy(tracks), LocoTrackAdapter._to_numpy(occlusion)

    def _autocast_context(self):
        if not str(self.device).startswith("cuda") or self.autocast_dtype == "fp32":
            return nullcontext()
        import torch

        dtype = torch.bfloat16 if self.autocast_dtype == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

    def _run_inference(self, video: np.ndarray, query_points_tyx: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        model = self._load_model()
        import torch

        queries = torch.from_numpy(np.ascontiguousarray(query_points_tyx, dtype=np.float32)).to(self.device)
        video_np = np.ascontiguousarray(np.asarray(video, dtype=np.uint8))
        self._sync_cuda_if_needed()
        started_s = time.perf_counter()
        with torch.no_grad(), self._autocast_context():
            output = model.inference(
                video_np,
                queries,
                query_chunk_size=int(self.query_chunk_size),
                resolution=tuple(self.resolution),
                query_format="tyx",
            )
        self._sync_cuda_if_needed()
        run_ms = float((time.perf_counter() - started_s) * 1000.0)
        tracks_xy, occlusion = self._extract_output(output)
        return {"tracks": tracks_xy, "occlusion": occlusion}, run_ms

    @staticmethod
    def _parse_window_output(
        output: Mapping[str, np.ndarray],
        *,
        batch_size: int,
        query_count: int,
        window_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        tracks_xy = np.asarray(output["tracks"], dtype=np.float32)
        occlusion = np.asarray(output["occlusion"])
        if tracks_xy.ndim == 3:
            tracks_xy = tracks_xy[None]
        if occlusion.ndim == 2:
            occlusion = occlusion[None]
        expected_tracks = (int(batch_size), int(query_count), int(window_frames), 2)
        expected_occ = (int(batch_size), int(query_count), int(window_frames))
        if tracks_xy.shape != expected_tracks:
            raise ValueError(f"LocoTrack tracks shape {tracks_xy.shape} does not match expected {expected_tracks}")
        if occlusion.shape != expected_occ:
            raise ValueError(f"LocoTrack occlusion shape {occlusion.shape} does not match expected {expected_occ}")
        tracks_yx_btn = np.transpose(tracks_xy[..., ::-1], (0, 2, 1, 3)).astype(np.float32)
        visibility_btn = np.transpose((~occlusion.astype(bool)).astype(np.float32), (0, 2, 1))
        return np.ascontiguousarray(tracks_yx_btn), np.ascontiguousarray(visibility_btn)

    def _append_serial_frame(self, frame: np.ndarray) -> tuple[int, int, int]:
        frame_idx = int(self._stream_total_frames)
        self._frame_buffer.append(self._validate_frame(frame, camera_idx=self.camera_idx))
        if len(self._frame_buffer) > self.window_frames:
            self._frame_buffer = self._frame_buffer[-self.window_frames :]
        self._stream_total_frames += 1
        effective = len(self._frame_buffer)
        return frame_idx - effective + 1, frame_idx, effective

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        _ = masks
        self._load_model()
        self._query_points_yx = self._validate_query_points(query_points_yx, camera_idx=self.camera_idx)
        self._frame_buffer = []
        self._stream_total_frames = 0
        for frame in frames:
            self.update(frame)

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._query_points_yx is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        chunk_start, chunk_end, effective = self._append_serial_frame(frame)
        video = np.stack(self._frame_buffer, axis=0)[None]
        queries = self._queries_yx_to_tyx(self._query_points_yx)[None]
        output, run_ms = self._run_inference(video, queries)
        tracks_yx_b, visibility_b = self._parse_window_output(
            output,
            batch_size=1,
            query_count=len(self._query_points_yx),
            window_frames=effective,
        )
        return TrackingResult(
            tracks_yx=tracks_yx_b[0],
            visibility=visibility_b[0],
            backend=self.name,
            camera_idx=self.camera_idx,
            query_points_yx=self._query_points_yx,
            stats={
                "backend": self.name,
                "tracker_backend": self.name,
                "adapter": type(self).__name__,
                "mode": "locotrack_windowed_serial",
                "stream_status": "published",
                "update_mode": "serial",
                "chunk_start_idx": int(chunk_start),
                "chunk_end_idx": int(chunk_end),
                "frames_seen": int(self._stream_total_frames),
                "model_size": self.model_size,
                "locotrack_model_size": self.model_size,
                "window_frames": int(self.window_frames),
                "locotrack_window_frames": int(self.window_frames),
                "effective_window_frames": int(effective),
                "num_query_points": int(len(self._query_points_yx)),
                "model_run_ms": float(run_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0.0 else 0.0,
                "query_chunk_size": int(self.query_chunk_size),
                "locotrack_query_chunk_size": int(self.query_chunk_size),
                "resolution": [int(self.resolution[0]), int(self.resolution[1])],
                "locotrack_resolution": [int(self.resolution[0]), int(self.resolution[1])],
                "autocast_dtype": self.autocast_dtype,
                "locotrack_autocast_dtype": self.autocast_dtype,
                "device": self.device,
            },
        )

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        self._load_model()
        query_points: dict[int, np.ndarray] = {}
        for camera_idx, points in sorted(query_points_yx_by_camera.items()):
            query_points[int(camera_idx)] = self._validate_query_points(points, camera_idx=int(camera_idx))
        if not query_points:
            raise ValueError("LocoTrack batch-views requires at least one camera.")
        counts = {int(camera_idx): int(len(points)) for camera_idx, points in query_points.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                "LocoTrack batch-views requires equal query counts per camera; "
                f"got {counts}. Use --tracker-batch-query-count-policy min-common."
            )
        self._batch_camera_ids = tuple(sorted(query_points))
        self._batch_query_points_yx_by_camera = query_points
        self._batch_frame_buffers_by_camera = {int(camera_idx): [] for camera_idx in self._batch_camera_ids}
        self._batch_total_frames = 0

    def _append_batch_frames(self, frames_by_camera: Mapping[int, np.ndarray]) -> tuple[int, int, int]:
        if not self._batch_camera_ids:
            raise RuntimeError("Call initialize_batch(...) before update_batch().")
        expected = set(int(item) for item in self._batch_camera_ids)
        received = set(int(item) for item in frames_by_camera)
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        if missing or extra:
            raise ValueError(f"LocoTrack batch-views camera set mismatch; missing={missing}, extra={extra}")
        frame_idx = int(self._batch_total_frames)
        for camera_idx in self._batch_camera_ids:
            idx = int(camera_idx)
            buffer = self._batch_frame_buffers_by_camera[idx]
            buffer.append(self._validate_frame(frames_by_camera[idx], camera_idx=idx))
            if len(buffer) > self.window_frames:
                self._batch_frame_buffers_by_camera[idx] = buffer[-self.window_frames :]
        self._batch_total_frames += 1
        lengths = {idx: len(self._batch_frame_buffers_by_camera[idx]) for idx in self._batch_camera_ids}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"LocoTrack batch-views rolling windows are not aligned: {lengths}")
        effective = next(iter(lengths.values()))
        return frame_idx - effective + 1, frame_idx, effective

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        chunk_start, chunk_end, effective = self._append_batch_frames(frames_by_camera)
        frames: list[np.ndarray] = []
        queries: list[np.ndarray] = []
        shape: tuple[int, ...] | None = None
        for camera_idx in self._batch_camera_ids:
            idx = int(camera_idx)
            window = self._batch_frame_buffers_by_camera[idx]
            if shape is None:
                shape = window[0].shape
            for frame in window:
                if frame.shape != shape:
                    raise ValueError(
                        "LocoTrack batch-views requires equal frame shapes; "
                        f"expected {shape}, got {frame.shape} for camera {idx}"
                    )
            frames.append(np.stack(window, axis=0))
            queries.append(self._queries_yx_to_tyx(self._batch_query_points_yx_by_camera[idx]))
        video = np.stack(frames, axis=0)
        query_points = np.stack(queries, axis=0)
        query_count = int(query_points.shape[1])
        output, run_ms = self._run_inference(video, query_points)
        tracks_yx_b, visibility_b = self._parse_window_output(
            output,
            batch_size=len(self._batch_camera_ids),
            query_count=query_count,
            window_frames=effective,
        )
        results: dict[int, TrackingResult] = {}
        camera_ids_list = [int(item) for item in self._batch_camera_ids]
        for batch_idx, camera_idx in enumerate(self._batch_camera_ids):
            idx = int(camera_idx)
            query_points_yx = self._batch_query_points_yx_by_camera[idx]
            results[idx] = TrackingResult(
                tracks_yx=tracks_yx_b[batch_idx],
                visibility=visibility_b[batch_idx],
                backend=self.name,
                camera_idx=idx,
                query_points_yx=query_points_yx,
                stats={
                    "backend": self.name,
                    "tracker_backend": self.name,
                    "adapter": type(self).__name__,
                    "mode": "locotrack_windowed_batch_views",
                    "stream_status": "published",
                    "update_mode": "batch",
                    "chunk_start_idx": int(chunk_start),
                    "chunk_end_idx": int(chunk_end),
                    "frames_seen": int(self._batch_total_frames),
                    "model_size": self.model_size,
                    "locotrack_model_size": self.model_size,
                    "window_frames": int(self.window_frames),
                    "locotrack_window_frames": int(self.window_frames),
                    "effective_window_frames": int(effective),
                    "num_query_points": int(len(query_points_yx)),
                    "model_run_ms": float(run_ms),
                    "fps_model_only": float(1000.0 / run_ms) if run_ms > 0.0 else 0.0,
                    "query_chunk_size": int(self.query_chunk_size),
                    "locotrack_query_chunk_size": int(self.query_chunk_size),
                    "resolution": [int(self.resolution[0]), int(self.resolution[1])],
                    "locotrack_resolution": [int(self.resolution[0]), int(self.resolution[1])],
                    "autocast_dtype": self.autocast_dtype,
                    "locotrack_autocast_dtype": self.autocast_dtype,
                    "batch_size": int(len(self._batch_camera_ids)),
                    "locotrack_batch_size": int(len(self._batch_camera_ids)),
                    "batch_camera_ids": camera_ids_list,
                    "batch_index": int(batch_idx),
                    "device": self.device,
                },
            )
        return results


__all__ = ["LocoTrackAdapter"]
