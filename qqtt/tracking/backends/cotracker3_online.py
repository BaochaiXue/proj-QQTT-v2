from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult


class CoTracker3OnlineBackend:
    name = "cotracker3_online"

    def __init__(self, *, device: str = "cuda", repo_or_dir: str = "facebookresearch/co-tracker", hub_model: str = "cotracker3_online", model: Any | None = None) -> None:
        self.device = str(device)
        self.repo_or_dir = str(repo_or_dir)
        self.hub_model = str(hub_model)
        self._model = model
        self._model_load_ms = 0.0
        self._stream_query_points_yx: np.ndarray | None = None
        self._stream_frames: list[np.ndarray] = []
        self._stream_total_frames = 0
        self._stream_last_processed_frame_count = 0
        self._stream_initialized = False
        self._stream_camera_idx: int | None = None
        self._batch_camera_ids: tuple[int, ...] = ()
        self._batch_query_points_yx_by_camera: dict[int, np.ndarray] = {}
        self._batch_query_counts_by_camera: dict[int, int] = {}
        self._batch_stream_frames: list[np.ndarray] = []
        self._batch_total_frames = 0
        self._batch_last_processed_frame_count = 0
        self._batch_initialized = False

    def availability(self) -> BackendAvailability:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            return BackendAvailability(self.name, False, f"torch is not importable: {exc}")
        return BackendAvailability(self.name, True, "torch is importable; CoTracker3 model loads lazily through torch.hub or injected model")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def _load_model(self):
        if self._model is not None:
            return self._model
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        import torch

        start = time.perf_counter()
        model = torch.hub.load(self.repo_or_dir, self.hub_model)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model = model.eval()
        self._model = model
        self._model_load_ms = (time.perf_counter() - start) * 1000.0
        return model

    def warmup(self) -> dict[str, float]:
        start = time.perf_counter()
        self._load_model()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - start) * 1000.0),
        }

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = masks
        self._load_model()
        self._reset_stream(query_points_yx)
        for frame in frames:
            self.update(frame)

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        self._load_model()
        self._reset_batch_stream(query_points_yx_by_camera)

    @staticmethod
    def _frames_to_torch_video(frames: Sequence[np.ndarray], *, device: str):
        import torch

        if not frames:
            raise ValueError("CoTracker3 requires at least one frame.")
        arr = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"frames_rgb must be HxWx3 RGB arrays; got {arr.shape}")
        return torch.from_numpy(arr).permute(0, 3, 1, 2)[None].contiguous().float().to(device)

    @staticmethod
    def _batch_frames_to_torch_video(frames: Sequence[np.ndarray], *, device: str):
        import torch

        if not frames:
            raise ValueError("CoTracker3 batch update requires at least one frame stack.")
        arr = np.stack([np.asarray(frame_stack, dtype=np.uint8) for frame_stack in frames], axis=1)
        if arr.ndim != 5 or arr.shape[-1] != 3:
            raise ValueError(f"batch frame stacks must be BxHxWx3 RGB arrays; got {arr.shape}")
        return torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float().to(device)

    @staticmethod
    def _queries_yx_to_torch(query_points_yx: np.ndarray, *, device: str):
        import torch

        query_points_yx = np.asarray(query_points_yx, dtype=np.float32)
        if query_points_yx.ndim != 2 or query_points_yx.shape[1] != 2:
            raise ValueError(f"query_points_yx must have shape (N,2); got {query_points_yx.shape}")
        query_pixels_xy = query_points_yx[:, ::-1]
        query_frame = np.zeros((query_pixels_xy.shape[0], 1), dtype=np.float32)
        return torch.from_numpy(np.concatenate([query_frame, query_pixels_xy], axis=1))[None].to(device)

    @staticmethod
    def _batch_queries_yx_to_torch(
        query_points_yx_by_camera: Mapping[int, np.ndarray],
        *,
        camera_ids: Sequence[int],
        device: str,
    ):
        import torch

        counts = [int(len(np.asarray(query_points_yx_by_camera[int(camera_idx)]).reshape(-1, 2))) for camera_idx in camera_ids]
        max_count = max(counts) if counts else 0
        if max_count <= 0:
            raise ValueError("CoTracker3 batch update requires at least one query point.")
        queries = np.zeros((len(tuple(camera_ids)), max_count, 3), dtype=np.float32)
        for batch_idx, camera_idx in enumerate(camera_ids):
            points_yx = np.asarray(query_points_yx_by_camera[int(camera_idx)], dtype=np.float32).reshape(-1, 2)
            if len(points_yx) == 0:
                continue
            queries[batch_idx, : len(points_yx), 1:] = points_yx[:, ::-1]
            if len(points_yx) < max_count:
                queries[batch_idx, len(points_yx) :, 1:] = points_yx[-1, ::-1]
        return torch.from_numpy(queries).contiguous().to(device)

    @staticmethod
    def _extract_prediction(output: Any) -> tuple[Any, Any]:
        if isinstance(output, dict):
            tracks = output.get("tracks")
            if tracks is None:
                tracks = output.get("pred_tracks")
            visibility = output.get("visibility")
            if visibility is None:
                visibility = output.get("pred_visibility")
            if tracks is None or visibility is None:
                raise ValueError(f"Could not find tracks/visibility in CoTracker output keys: {sorted(output)}")
            return tracks, visibility
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            return output[0], output[1]
        raise ValueError(f"Unsupported CoTracker output type: {type(output).__name__}")

    @staticmethod
    def _prediction_to_numpy(tracks_xy: Any, visibility: Any) -> tuple[np.ndarray, np.ndarray]:
        tracks_np = tracks_xy.detach().cpu().numpy() if hasattr(tracks_xy, "detach") else np.asarray(tracks_xy)
        visibility_np = visibility.detach().cpu().numpy() if hasattr(visibility, "detach") else np.asarray(visibility)
        if tracks_np.ndim == 4:
            tracks_np = tracks_np[0]
        if visibility_np.ndim in {3, 4} and visibility_np.shape[0] == 1:
            visibility_np = visibility_np[0]
        return tracks_np.astype(np.float32)[:, :, ::-1], visibility_np.astype(np.float32)

    @staticmethod
    def _prediction_to_numpy_batch(tracks_xy: Any, visibility: Any) -> tuple[np.ndarray, np.ndarray]:
        tracks_np = tracks_xy.detach().cpu().numpy() if hasattr(tracks_xy, "detach") else np.asarray(tracks_xy)
        visibility_np = visibility.detach().cpu().numpy() if hasattr(visibility, "detach") else np.asarray(visibility)
        if tracks_np.ndim != 4:
            raise ValueError(f"batch CoTracker tracks must be BxTxNx2; got {tracks_np.shape}")
        if visibility_np.ndim == 4 and visibility_np.shape[-1] == 1:
            visibility_np = visibility_np[..., 0]
        if visibility_np.ndim != 3:
            raise ValueError(f"batch CoTracker visibility must be BxTxN; got {visibility_np.shape}")
        return tracks_np.astype(np.float32)[:, :, :, ::-1], visibility_np.astype(np.float32)

    @staticmethod
    def _run_online_model(model: Any, video: Any, queries: Any) -> tuple[Any, Any]:
        """Run CoTrackerOnlinePredictor with its required init/step protocol."""
        step = int(getattr(model, "step", 0) or 0)
        if step <= 0:
            raise AttributeError("CoTracker online predictor does not expose a positive step size.")
        model(video_chunk=video, is_first_step=True, queries=queries, grid_size=0, add_support_grid=False)
        pred_tracks = pred_visibility = None
        total_frames = int(video.shape[1])
        for ind in range(0, max(total_frames - step, 0), step):
            pred_tracks, pred_visibility = model(
                video_chunk=video[:, ind : ind + step * 2],
                is_first_step=False,
                grid_size=0,
                add_support_grid=False,
            )
        if pred_tracks is None or pred_visibility is None:
            pred_tracks, pred_visibility = model(
                video_chunk=video,
                is_first_step=False,
                grid_size=0,
                add_support_grid=False,
            )
        return pred_tracks, pred_visibility

    @staticmethod
    def _online_step_and_window(model: Any) -> tuple[int, int]:
        step = int(getattr(model, "step", 0) or 0)
        if step <= 0:
            raise AttributeError("CoTracker online predictor does not expose a positive step size.")
        return step, step * 2

    def _reset_stream(self, query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> None:
        query_points = np.asarray(query_points_yx, dtype=np.float32)
        if query_points.ndim != 2 or query_points.shape[1] != 2:
            raise ValueError(f"query_points_yx must have shape (N,2); got {query_points.shape}")
        self._stream_query_points_yx = query_points
        self._stream_frames = []
        self._stream_total_frames = 0
        self._stream_last_processed_frame_count = 0
        self._stream_initialized = False
        self._stream_camera_idx = camera_idx

    def _reset_batch_stream(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        query_points: dict[int, np.ndarray] = {}
        for camera_idx, points in sorted(query_points_yx_by_camera.items()):
            arr = np.asarray(points, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"query_points_yx for camera {camera_idx} must have shape (N,2); got {arr.shape}")
            if len(arr) == 0:
                raise ValueError(f"query_points_yx for camera {camera_idx} is empty.")
            query_points[int(camera_idx)] = arr
        if not query_points:
            raise ValueError("batch CoTracker stream requires at least one camera.")
        self._batch_camera_ids = tuple(sorted(query_points))
        self._batch_query_points_yx_by_camera = query_points
        self._batch_query_counts_by_camera = {
            int(camera_idx): int(len(points))
            for camera_idx, points in query_points.items()
        }
        self._batch_stream_frames = []
        self._batch_total_frames = 0
        self._batch_last_processed_frame_count = 0
        self._batch_initialized = False

    def _empty_stream_result(self, *, status: str, step: int, window_len: int) -> TrackingResult:
        query_points = self._stream_query_points_yx
        if query_points is None:
            raise RuntimeError("CoTracker3 online stream is not initialized with query points.")
        return TrackingResult(
            tracks_yx=np.empty((0, len(query_points), 2), dtype=np.float32),
            visibility=np.empty((0, len(query_points)), dtype=np.float32),
            backend=self.name,
            camera_idx=self._stream_camera_idx,
            query_points_yx=query_points,
            stats={
                "backend": self.name,
                "mode": "cotracker3_online_streaming_update",
                "stream_status": status,
                "online_step": int(step),
                "online_window_len": int(window_len),
                "frames_buffered": int(len(self._stream_frames)),
                "frames_seen": int(self._stream_total_frames),
            },
        )

    def _empty_batch_results(self, *, status: str, step: int, window_len: int) -> dict[int, TrackingResult]:
        results: dict[int, TrackingResult] = {}
        for camera_idx in self._batch_camera_ids:
            query_points = self._batch_query_points_yx_by_camera[int(camera_idx)]
            results[int(camera_idx)] = TrackingResult(
                tracks_yx=np.empty((0, len(query_points), 2), dtype=np.float32),
                visibility=np.empty((0, len(query_points)), dtype=np.float32),
                backend=self.name,
                camera_idx=int(camera_idx),
                query_points_yx=query_points,
                stats={
                    "backend": self.name,
                    "mode": "cotracker3_online_batch_update",
                    "update_mode": "batch",
                    "stream_status": status,
                    "online_step": int(step),
                    "online_window_len": int(window_len),
                    "frames_buffered": int(len(self._batch_stream_frames)),
                    "frames_seen": int(self._batch_total_frames),
                    "batch_size": int(len(self._batch_camera_ids)),
                    "batch_camera_ids": [int(item) for item in self._batch_camera_ids],
                },
            )
        return results

    def _tracks_to_result(
        self,
        *,
        tracks_xy: Any,
        visibility: Any,
        run_ms: float,
        step: int,
        window_len: int,
    ) -> TrackingResult:
        query_points = self._stream_query_points_yx
        if query_points is None:
            raise RuntimeError("CoTracker3 online stream is not initialized with query points.")
        tracks_yx, visibility_np = self._prediction_to_numpy(tracks_xy, visibility)
        chunk_end = self._stream_total_frames - 1
        chunk_start = max(0, chunk_end - int(tracks_yx.shape[0]) + 1)
        return TrackingResult(
            tracks_yx=tracks_yx,
            visibility=visibility_np,
            backend=self.name,
            camera_idx=self._stream_camera_idx,
            query_points_yx=query_points,
            stats={
                "backend": self.name,
                "mode": "cotracker3_online_streaming_update",
                "stream_status": "published",
                "online_step": int(step),
                "online_window_len": int(window_len),
                "chunk_start_idx": int(chunk_start),
                "chunk_end_idx": int(chunk_end),
                "frames_buffered": int(len(self._stream_frames)),
                "frames_seen": int(self._stream_total_frames),
                "num_query_points": int(len(query_points)),
                "model_run_ms": float(run_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                "device": self.device,
            },
        )

    def _tracks_to_batch_results(
        self,
        *,
        tracks_xy: Any,
        visibility: Any,
        run_ms: float,
        step: int,
        window_len: int,
    ) -> dict[int, TrackingResult]:
        tracks_yx_batch, visibility_batch = self._prediction_to_numpy_batch(tracks_xy, visibility)
        chunk_end = self._batch_total_frames - 1
        chunk_start = max(0, chunk_end - int(tracks_yx_batch.shape[1]) + 1)
        results: dict[int, TrackingResult] = {}
        for batch_idx, camera_idx in enumerate(self._batch_camera_ids):
            count = int(self._batch_query_counts_by_camera[int(camera_idx)])
            query_points = self._batch_query_points_yx_by_camera[int(camera_idx)]
            results[int(camera_idx)] = TrackingResult(
                tracks_yx=tracks_yx_batch[batch_idx, :, :count, :],
                visibility=visibility_batch[batch_idx, :, :count],
                backend=self.name,
                camera_idx=int(camera_idx),
                query_points_yx=query_points,
                stats={
                    "backend": self.name,
                    "mode": "cotracker3_online_batch_update",
                    "update_mode": "batch",
                    "stream_status": "published",
                    "online_step": int(step),
                    "online_window_len": int(window_len),
                    "chunk_start_idx": int(chunk_start),
                    "chunk_end_idx": int(chunk_end),
                    "frames_buffered": int(len(self._batch_stream_frames)),
                    "frames_seen": int(self._batch_total_frames),
                    "num_query_points": int(count),
                    "model_run_ms": float(run_ms),
                    "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                    "device": self.device,
                    "batch_size": int(len(self._batch_camera_ids)),
                    "batch_camera_ids": [int(item) for item in self._batch_camera_ids],
                },
            )
        return results

    def track_sequence(
        self,
        frames: Sequence[np.ndarray] | None = None,
        query_points_yx: np.ndarray | None = None,
        *,
        frames_rgb: Sequence[np.ndarray] | None = None,
        camera_idx: int | None = None,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> TrackingResult:
        _ = output_shape_hw
        video_frames = list(frames_rgb if frames_rgb is not None else frames or [])
        if query_points_yx is None:
            raise ValueError("query_points_yx is required.")
        load_start = time.perf_counter()
        model = self._load_model()
        load_ms = self._model_load_ms if self._model_load_ms else (time.perf_counter() - load_start) * 1000.0
        if hasattr(model, "step"):
            step, window_len = self._online_step_and_window(model)
            query_points = np.asarray(query_points_yx, dtype=np.float32)
            self._reset_stream(query_points, camera_idx=camera_idx)
            frame_count = int(len(video_frames))
            tracks_yx = np.zeros((frame_count, len(query_points), 2), dtype=np.float32)
            visibility = np.zeros((frame_count, len(query_points)), dtype=np.float32)
            model_run_ms = 0.0
            published_chunks = 0
            published_end_idx = -1
            replay_start = time.perf_counter()
            for frame in video_frames:
                update_result = self.update(frame)
                if str(update_result.stats.get("stream_status", "")) != "published":
                    continue
                chunk_start = int(update_result.stats.get("chunk_start_idx", 0))
                chunk_end = min(int(update_result.stats.get("chunk_end_idx", chunk_start - 1)), frame_count - 1)
                chunk_len = max(0, chunk_end - chunk_start + 1)
                if chunk_len <= 0:
                    continue
                chunk_visibility = update_result.visibility
                if chunk_visibility.ndim == 3 and chunk_visibility.shape[-1] == 1:
                    chunk_visibility = chunk_visibility[..., 0]
                tracks_yx[chunk_start : chunk_end + 1] = update_result.tracks_yx[:chunk_len]
                visibility[chunk_start : chunk_end + 1] = chunk_visibility[:chunk_len]
                model_run_ms += float(update_result.stats.get("model_run_ms", 0.0))
                published_chunks += 1
                published_end_idx = max(published_end_idx, chunk_end)
            replay_ms = (time.perf_counter() - replay_start) * 1000.0
            if published_chunks == 0:
                tracks_yx = np.empty((0, len(query_points), 2), dtype=np.float32)
                visibility = np.empty((0, len(query_points)), dtype=np.float32)
            published_frame_count = int(published_end_idx + 1) if published_end_idx >= 0 else 0
            return TrackingResult(
                tracks_yx=tracks_yx,
                visibility=visibility,
                backend=self.name,
                camera_idx=camera_idx,
                query_points_yx=query_points,
                stats={
                    "backend": self.name,
                    "camera_idx": None if camera_idx is None else int(camera_idx),
                    "num_frames": frame_count,
                    "num_published_frames": published_frame_count,
                    "num_query_points": int(len(query_points)),
                    "model_load_ms": float(load_ms),
                    "model_run_ms": float(model_run_ms),
                    "stream_replay_ms": float(replay_ms),
                    "fps_model_only": float(published_frame_count * 1000.0 / model_run_ms) if model_run_ms > 0 else 0.0,
                    "device": self.device,
                    "mode": "cotracker3_online_streaming_replay",
                    "online_step": int(step),
                    "online_window_len": int(window_len),
                    "published_chunks": int(published_chunks),
                    "stream_tail_unpublished_frames": int(max(0, frame_count - published_frame_count)),
                },
            )
        import torch

        video = self._frames_to_torch_video(video_frames, device=self.device)
        queries = self._queries_yx_to_torch(query_points_yx, device=self.device)
        run_start = time.perf_counter()
        with torch.no_grad():
            if hasattr(model, "step"):
                tracks_xy, visibility = self._run_online_model(model, video, queries)
            else:
                try:
                    output = model(video, queries=queries, is_online=False)
                except TypeError:
                    output = model(video, queries=queries)
                tracks_xy, visibility = self._extract_prediction(output)
        run_ms = (time.perf_counter() - run_start) * 1000.0
        tracks_yx, visibility_np = self._prediction_to_numpy(tracks_xy, visibility)
        step = int(getattr(model, "step", 0) or 0)
        return TrackingResult(
            tracks_yx=tracks_yx,
            visibility=visibility_np,
            backend=self.name,
            camera_idx=camera_idx,
            query_points_yx=np.asarray(query_points_yx, dtype=np.float32),
            stats={
                "backend": self.name,
                "camera_idx": None if camera_idx is None else int(camera_idx),
                "num_frames": int(len(video_frames)),
                "num_query_points": int(len(query_points_yx)),
                "model_load_ms": float(load_ms),
                "model_run_ms": float(run_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                "device": self.device,
                "mode": "cotracker3_online_sequence_wrapper",
                "online_step": step,
                "online_window_len": step * 2 if step > 0 else 0,
            },
        )

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._stream_query_points_yx is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        model = self._load_model()
        step, window_len = self._online_step_and_window(model)
        self._stream_frames.append(np.asarray(frame, dtype=np.uint8))
        self._stream_total_frames += 1
        if len(self._stream_frames) > window_len:
            self._stream_frames = self._stream_frames[-window_len:]
        if len(self._stream_frames) < window_len:
            return self._empty_stream_result(status="buffering", step=step, window_len=window_len)
        if self._stream_initialized and self._stream_total_frames - self._stream_last_processed_frame_count < step:
            return self._empty_stream_result(status="waiting_for_step", step=step, window_len=window_len)

        import torch

        video = self._frames_to_torch_video(self._stream_frames, device=self.device)
        run_start = time.perf_counter()
        with torch.no_grad():
            if not self._stream_initialized:
                queries = self._queries_yx_to_torch(self._stream_query_points_yx, device=self.device)
                model(video_chunk=video, is_first_step=True, queries=queries, grid_size=0, add_support_grid=False)
                self._stream_initialized = True
            tracks_xy, visibility = model(
                video_chunk=video,
                is_first_step=False,
                grid_size=0,
                add_support_grid=False,
            )
        run_ms = (time.perf_counter() - run_start) * 1000.0
        self._stream_last_processed_frame_count = self._stream_total_frames
        return self._tracks_to_result(
            tracks_xy=tracks_xy,
            visibility=visibility,
            run_ms=run_ms,
            step=step,
            window_len=window_len,
        )

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        if not self._batch_camera_ids:
            raise RuntimeError("Call initialize_batch(...) before update_batch().")
        model = self._load_model()
        step, window_len = self._online_step_and_window(model)
        frame_stack = np.stack(
            [
                np.asarray(frames_by_camera[int(camera_idx)], dtype=np.uint8)
                for camera_idx in self._batch_camera_ids
            ],
            axis=0,
        )
        if frame_stack.ndim != 4 or frame_stack.shape[-1] != 3:
            raise ValueError(f"batch frames must be BxHxWx3 RGB arrays; got {frame_stack.shape}")
        self._batch_stream_frames.append(frame_stack)
        self._batch_total_frames += 1
        if len(self._batch_stream_frames) > window_len:
            self._batch_stream_frames = self._batch_stream_frames[-window_len:]
        if len(self._batch_stream_frames) < window_len:
            return self._empty_batch_results(status="buffering", step=step, window_len=window_len)
        if self._batch_initialized and self._batch_total_frames - self._batch_last_processed_frame_count < step:
            return self._empty_batch_results(status="waiting_for_step", step=step, window_len=window_len)

        import torch

        video = self._batch_frames_to_torch_video(self._batch_stream_frames, device=self.device)
        run_start = time.perf_counter()
        with torch.no_grad():
            if not self._batch_initialized:
                queries = self._batch_queries_yx_to_torch(
                    self._batch_query_points_yx_by_camera,
                    camera_ids=self._batch_camera_ids,
                    device=self.device,
                )
                model(video_chunk=video, is_first_step=True, queries=queries, grid_size=0, add_support_grid=False)
                self._batch_initialized = True
            tracks_xy, visibility = model(
                video_chunk=video,
                is_first_step=False,
                grid_size=0,
                add_support_grid=False,
            )
        run_ms = (time.perf_counter() - run_start) * 1000.0
        self._batch_last_processed_frame_count = self._batch_total_frames
        return self._tracks_to_batch_results(
            tracks_xy=tracks_xy,
            visibility=visibility,
            run_ms=run_ms,
            step=step,
            window_len=window_len,
        )
