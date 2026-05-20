from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any
import types

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_LITETRACKER,
    PointTrackerBackendSpec,
)


class LiteTrackerAdapter:
    """Live LiteTracker adapter over the external LiteTracker repo."""

    name = TRACKER_BACKEND_LITETRACKER
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_LITETRACKER,
        family="litetracker",
        supports_batch_views=True,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="experimental_batch_views",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        weights: str | None = None,
        repo_dir: str | None = None,
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.weights = str(Path(weights).expanduser()) if weights else None
        self.repo_dir = str(Path(repo_dir).expanduser()) if repo_dir else None
        self._tracker: Any | None = None
        self._model_load_ms = 0.0
        self._query_points_yx: np.ndarray | None = None
        self._frame_count = 0
        self._batch_camera_ids: tuple[int, ...] = ()
        self._batch_query_points_yx_by_camera: dict[int, np.ndarray] = {}
        self._batch_queries_xyf: Any | None = None
        self._batch_frame_count = 0
        self._batch_is_first_frame = True

    def availability(self) -> BackendAvailability:
        missing: list[str] = []
        if not self.weights:
            missing.append("--litetracker-weights")
        elif not Path(self.weights).is_file():
            missing.append(f"--litetracker-weights {self.weights!r} does not exist")
        if self.repo_dir and not Path(self.repo_dir).is_dir():
            missing.append(f"--litetracker-repo-dir {self.repo_dir!r} does not exist")
        if missing:
            return BackendAvailability(self.name, False, "; ".join(missing))
        try:
            import torch  # noqa: F401
            from src.eval.lite_tracker_wrapper import LiteTrackerWrapper  # noqa: F401
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                f"LiteTracker runtime import failed: {type(exc).__name__}: {exc}",
            )
        return BackendAvailability(self.name, True, "LiteTracker wrapper import and weights path are available")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def is_initialized(self) -> bool:
        return self._query_points_yx is not None

    def _load_tracker(self) -> Any:
        if self._tracker is not None:
            return self._tracker
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        import torch
        from src.eval.lite_tracker_wrapper import LiteTrackerWrapper

        started_s = time.perf_counter()
        tracker = LiteTrackerWrapper(Path(str(self.weights)), return_vis=True)
        self._patch_model_for_batch_views(getattr(tracker, "model", None))
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._model_load_ms = float((time.perf_counter() - started_s) * 1000.0)
        self._tracker = tracker
        return tracker

    @staticmethod
    def _with_time_axis(value: Any, reference: Any) -> Any:
        """Keep LiteTracker online buffers broadcastable for B>1 camera batches."""
        if getattr(value, "ndim", None) == getattr(reference, "ndim", None) - 1:
            return value[:, None]
        return value

    @classmethod
    def _patch_model_for_batch_views(cls, model: Any | None) -> None:
        if model is None or bool(getattr(model, "_qqtt_batch_view_forward_patch", False)):
            return
        required = ("fnet", "forward_window", "get_track_feat", "corr_levels", "stride", "model_resolution")
        if not all(hasattr(model, name) for name in required):
            return

        def patched_forward(self: Any, frame: Any, queries: Any):  # type: ignore[no-untyped-def]
            import torch
            import torch.nn.functional as F

            original_shape = frame.shape
            frame = F.interpolate(
                frame, size=self.model_resolution, mode="bilinear", align_corners=True
            )
            queries_scaled = queries.clone()
            queries_scaled[:, :, 1] *= (self.model_resolution[1] - 1) / (original_shape[3] - 1)
            queries_scaled[:, :, 2] *= (self.model_resolution[0] - 1) / (original_shape[2] - 1)

            batch_size, _channels, height, width = frame.shape
            device = queries_scaled.device
            batch_size, query_count, _ = queries_scaled.shape
            frame = 2 * (frame / 255.0) - 1.0
            window_size = 1
            dtype = frame.dtype

            queried_frames = queries_scaled[:, :, 0].long()
            queried_coords = queries_scaled[..., 1:3].clone() / self.stride

            fmaps = self.fnet(frame)
            fmaps = fmaps.permute(0, 2, 3, 1)
            fmaps = fmaps / torch.sqrt(
                torch.maximum(
                    torch.sum(torch.square(fmaps), dim=-1, keepdim=True),
                    torch.tensor(1e-12, device=fmaps.device),
                )
            )
            fmaps = fmaps.permute(0, 3, 1, 2).reshape(
                batch_size,
                -1,
                self.latent_dim,
                height // self.stride,
                width // self.stride,
            )
            fmaps = fmaps.to(dtype)

            fmaps_pyramid = [fmaps]
            track_feat_support_pyramid = []
            for _level in range(self.corr_levels - 1):
                pooled = fmaps.reshape(
                    batch_size * window_size,
                    self.latent_dim,
                    fmaps.shape[-2],
                    fmaps.shape[-1],
                )
                pooled = F.avg_pool2d(pooled, 2, stride=2)
                fmaps = pooled.reshape(
                    batch_size,
                    window_size,
                    self.latent_dim,
                    pooled.shape[-2],
                    pooled.shape[-1],
                )
                fmaps_pyramid.append(fmaps)

            initialized_now = (queried_frames == self.online_ind)[:, None, :, None]
            for level in range(self.corr_levels):
                if self.online_ind == 0:
                    _, track_feat_support = self.get_track_feat(
                        fmaps_pyramid[level],
                        queried_coords / 2**level,
                        support_radius=self.corr_radius,
                    )
                    self.track_feat_cache[level] = torch.zeros_like(track_feat_support, device=device)
                    self.track_feat_cache[level] += track_feat_support * initialized_now
                elif initialized_now.any():
                    _, track_feat_support = self.get_track_feat(
                        fmaps_pyramid[level],
                        queried_coords / 2**level,
                        support_radius=self.corr_radius,
                    )
                    self.track_feat_cache[level] += track_feat_support * initialized_now
                track_feat_support_pyramid.append(self.track_feat_cache[level].unsqueeze(1))

            vis_init = torch.zeros((batch_size, window_size, query_count, 1), device=device).float()
            conf_init = torch.zeros((batch_size, window_size, query_count, 1), device=device).float()
            coords_init = queried_coords.reshape(batch_size, window_size, query_count, 2).float()

            vis_init = torch.where(
                initialized_now.expand_as(vis_init),
                self.inv_sigmoid_true_val,
                vis_init,
            )
            conf_init = torch.where(
                initialized_now.expand_as(conf_init),
                self.inv_sigmoid_true_val,
                conf_init,
            )

            if self.online_ind == 0:
                self.ema_flow_buffer = torch.zeros_like(coords_init)

            previously_initialized = (queried_frames < self.online_ind)[:, None, :, None]
            if self.online_ind > 0:
                previous_vis = cls._with_time_axis(self.vis_buffer[:, -1], vis_init)
                previous_conf = cls._with_time_axis(self.conf_buffer[:, -1], conf_init)
                vis_init = torch.where(
                    previously_initialized.expand_as(vis_init),
                    previous_vis,
                    vis_init,
                )
                conf_init = torch.where(
                    previously_initialized.expand_as(conf_init),
                    previous_conf,
                    conf_init,
                )
                if self.online_ind == 1:
                    previous_coords = cls._with_time_axis(self.coords_buffer[:, -1], coords_init)
                    coords_init = torch.where(
                        previously_initialized.expand_as(coords_init),
                        previous_coords,
                        coords_init,
                    )
                else:
                    last_flow = cls._with_time_axis(
                        self.coords_buffer[:, -1] - self.coords_buffer[:, -2],
                        coords_init,
                    )
                    cached_flow = cls._with_time_axis(self.ema_flow_buffer, coords_init)
                    alpha = 0.8
                    accumulated_flow = alpha * last_flow + (1 - alpha) * cached_flow
                    self.ema_flow_buffer = accumulated_flow
                    previous_coords = cls._with_time_axis(self.coords_buffer[:, -1], coords_init)
                    coords_init = torch.where(
                        previously_initialized.expand_as(coords_init),
                        previous_coords + accumulated_flow,
                        coords_init,
                    )

            coords, viss, confs = self.forward_window(
                fmaps_pyramid=fmaps_pyramid,
                coords=coords_init,
                track_feat_support_pyramid=track_feat_support_pyramid,
                queried_frames=queried_frames,
                vis=vis_init,
                conf=conf_init,
                iters=self.iters,
                is_track_previsouly_initialized=previously_initialized,
            )

            coords[:, :, :, 0] *= (original_shape[3] - 1) / (self.model_resolution[1] - 1)
            coords[:, :, :, 1] *= (original_shape[2] - 1) / (self.model_resolution[0] - 1)

            viss = torch.sigmoid(viss)
            confs = torch.sigmoid(confs)
            viss = (viss * confs) > 0.6
            self.online_ind += 1
            return coords, viss, confs

        model.forward = types.MethodType(patched_forward, model)
        model._qqtt_batch_view_forward_patch = True

    def warmup(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        tracker = self._load_tracker()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "device": str(getattr(tracker, "device", self.device)),
            "dtype": str(getattr(tracker, "dtype", "")),
            "batch_support_status": self.spec.batch_support_status,
        }

    @staticmethod
    def _validate_query_points(query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            prefix = "query_points_yx" if camera_idx is None else f"query_points_yx for camera {camera_idx}"
            raise ValueError(f"{prefix} must have shape (N,2); got {points.shape}")
        if len(points) == 0:
            raise ValueError("LiteTracker requires at least one query point.")
        return np.ascontiguousarray(points)

    @staticmethod
    def _frame_to_torch_uint8(frame: np.ndarray):
        import torch

        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"LiteTracker frame must be HxWx3 RGB uint8; got {arr.shape}")
        return torch.from_numpy(np.ascontiguousarray(arr))[None]

    @staticmethod
    def _frames_to_torch_chw_batch(frames: Sequence[np.ndarray], *, device: str, dtype: Any):
        import torch

        arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
        if not arrays:
            raise ValueError("LiteTracker batch update requires at least one frame.")
        shape = arrays[0].shape
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"LiteTracker frame must be HxWx3 RGB uint8; got {shape}")
        for idx, arr in enumerate(arrays):
            if arr.shape != shape:
                raise ValueError(
                    "LiteTracker batch-views requires equal frame shapes; "
                    f"frame 0 has {shape}, frame {idx} has {arr.shape}"
                )
        stacked = np.stack([np.ascontiguousarray(arr) for arr in arrays], axis=0)
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)

    @staticmethod
    def _queries_yx_to_xyf_batch(
        query_points_yx_by_camera: Mapping[int, np.ndarray],
        *,
        camera_ids: Sequence[int],
        device: str,
    ):
        import torch

        counts = {
            int(camera_idx): int(len(np.asarray(query_points_yx_by_camera[int(camera_idx)], dtype=np.float32).reshape(-1, 2)))
            for camera_idx in camera_ids
        }
        if not counts:
            raise ValueError("LiteTracker batch stream requires at least one camera.")
        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            raise ValueError(
                "LiteTracker batch-views requires equal query counts per camera; "
                f"got {counts}. Use --tracker-batch-query-count-policy min-common."
            )
        count = next(iter(unique_counts))
        if count <= 0:
            raise ValueError("LiteTracker batch-views requires at least one query point per camera.")
        queries = []
        for camera_idx in camera_ids:
            points_yx = np.asarray(query_points_yx_by_camera[int(camera_idx)], dtype=np.float32).reshape(-1, 2)
            points_xy = np.ascontiguousarray(points_yx[:, ::-1])
            frame_index = np.zeros((len(points_xy), 1), dtype=np.float32)
            queries.append(np.concatenate([frame_index, points_xy], axis=1))
        return torch.from_numpy(np.stack(queries, axis=0)).float().contiguous().to(device=device)

    @staticmethod
    def _autocast_context(*, device: str, dtype: Any):
        if not str(device).startswith("cuda"):
            return nullcontext()
        import torch

        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        _ = masks
        self._load_tracker()
        self._query_points_yx = self._validate_query_points(query_points_yx, camera_idx=self.camera_idx)
        self._frame_count = 0
        for frame in frames:
            self.update(frame)

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        tracker = self._load_tracker()
        query_points: dict[int, np.ndarray] = {}
        for camera_idx, points in sorted(query_points_yx_by_camera.items()):
            query_points[int(camera_idx)] = self._validate_query_points(points, camera_idx=int(camera_idx))
        if not query_points:
            raise ValueError("LiteTracker batch stream requires at least one camera.")
        camera_ids = tuple(sorted(query_points))
        self._batch_queries_xyf = self._queries_yx_to_xyf_batch(
            query_points,
            camera_ids=camera_ids,
            device=str(getattr(tracker, "device", self.device)),
        )
        self._batch_camera_ids = camera_ids
        self._batch_query_points_yx_by_camera = query_points
        self._batch_frame_count = 0
        self._batch_is_first_frame = True
        model = getattr(tracker, "model", None)
        reset = getattr(model, "init_video_online_processing", None)
        if callable(reset):
            reset()
        if hasattr(tracker, "queries"):
            tracker.queries = self._batch_queries_xyf
        if hasattr(tracker, "is_first_frame"):
            tracker.is_first_frame = False

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        if self._batch_queries_xyf is None or not self._batch_camera_ids:
            raise RuntimeError("Call initialize_batch(...) before update_batch().")
        missing = [int(camera_idx) for camera_idx in self._batch_camera_ids if int(camera_idx) not in frames_by_camera]
        if missing:
            raise ValueError(f"LiteTracker batch-views missing frame(s) for camera(s): {missing}")
        tracker = self._load_tracker()
        model = getattr(tracker, "model", None)
        if model is None:
            raise BackendUnavailableError("LiteTracker batch-views requires LiteTrackerWrapper.model.")
        import torch

        device = str(getattr(tracker, "device", self.device))
        dtype = getattr(tracker, "dtype", torch.float32)
        frames = [np.asarray(frames_by_camera[int(camera_idx)], dtype=np.uint8) for camera_idx in self._batch_camera_ids]
        frame_tensor = self._frames_to_torch_chw_batch(frames, device=device, dtype=dtype)
        started_s = time.perf_counter()
        with torch.no_grad(), self._autocast_context(device=device, dtype=dtype):
            if self._batch_is_first_frame:
                model(frame_tensor, queries=self._batch_queries_xyf)
                self._batch_is_first_frame = False
            coords, visibility, *rest = model(frame_tensor, queries=self._batch_queries_xyf)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_ms = float((time.perf_counter() - started_s) * 1000.0)

        tracks_xy = np.asarray(coords.detach().cpu().numpy(), dtype=np.float32)
        visibility_np = np.asarray(visibility.detach().cpu().numpy(), dtype=np.float32)
        confidence_np = None
        if rest:
            confidence_np = np.asarray(rest[0].detach().cpu().numpy(), dtype=np.float32)
        if tracks_xy.ndim != 4 or tracks_xy.shape[0] != len(self._batch_camera_ids) or tracks_xy.shape[-1] != 2:
            raise ValueError(f"LiteTracker batch returned invalid coords shape {tracks_xy.shape}")
        if visibility_np.ndim == 4 and visibility_np.shape[-1] == 1:
            visibility_np = visibility_np[..., 0]
        if visibility_np.ndim != 3 or visibility_np.shape[:3] != tracks_xy.shape[:3]:
            raise ValueError(
                f"LiteTracker batch returned visibility shape {visibility_np.shape}, expected {tracks_xy.shape[:-1]}"
            )
        if confidence_np is not None and confidence_np.ndim == 4 and confidence_np.shape[-1] == 1:
            confidence_np = confidence_np[..., 0]

        frame_idx = int(self._batch_frame_count)
        self._batch_frame_count += 1
        results: dict[int, TrackingResult] = {}
        for batch_idx, camera_idx in enumerate(self._batch_camera_ids):
            idx = int(camera_idx)
            tracks_yx = tracks_xy[batch_idx, ..., ::-1].astype(np.float32)
            visibility_t = visibility_np[batch_idx].astype(np.float32)
            confidence_t = None if confidence_np is None else confidence_np[batch_idx].astype(np.float32)
            query_points = self._batch_query_points_yx_by_camera[idx]
            results[idx] = TrackingResult(
                tracks_yx=tracks_yx,
                visibility=visibility_t,
                confidence=confidence_t,
                backend=self.name,
                camera_idx=idx,
                query_points_yx=query_points,
                stats={
                    "backend": self.name,
                    "tracker_backend": self.name,
                    "adapter": type(self).__name__,
                    "mode": "litetracker_batch_views",
                    "stream_status": "published",
                    "update_mode": "batch",
                    "chunk_start_idx": frame_idx,
                    "chunk_end_idx": frame_idx,
                    "frames_seen": self._batch_frame_count,
                    "num_query_points": int(len(query_points)),
                    "model_run_ms": float(run_ms),
                    "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                    "device": device,
                    "dtype": str(dtype),
                    "batch_size": int(len(self._batch_camera_ids)),
                    "batch_camera_ids": [int(item) for item in self._batch_camera_ids],
                    "batch_index": int(batch_idx),
                    "lite_batch_size": int(len(self._batch_camera_ids)),
                    "lite_effective_query_count": int(len(query_points)),
                    "lite_model_ms": float(run_ms),
                },
            )
        return results

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._query_points_yx is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        import torch

        tracker = self._load_tracker()
        frame_tensor = self._frame_to_torch_uint8(frame)
        queries_xy = np.ascontiguousarray(self._query_points_yx[:, ::-1])
        started_s = time.perf_counter()
        points_xy, visibility = tracker.trackpoints2D(queries_xy, [frame_tensor, frame_tensor])
        if str(getattr(tracker, "device", self.device)).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_ms = float((time.perf_counter() - started_s) * 1000.0)
        tracks_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        tracks_yx = tracks_xy[:, ::-1][None, :, :].astype(np.float32)
        visibility_np = np.asarray(visibility, dtype=np.float32).reshape(-1)
        visibility_t = visibility_np[None, :].astype(np.float32)
        frame_idx = int(self._frame_count)
        self._frame_count += 1
        return TrackingResult(
            tracks_yx=tracks_yx,
            visibility=visibility_t,
            backend=self.name,
            camera_idx=self.camera_idx,
            query_points_yx=self._query_points_yx,
            stats={
                "backend": self.name,
                "tracker_backend": self.name,
                "adapter": type(self).__name__,
                "mode": "litetracker_trackpoints2D",
                "stream_status": "published",
                "update_mode": "serial",
                "chunk_start_idx": frame_idx,
                "chunk_end_idx": frame_idx,
                "frames_seen": self._frame_count,
                "num_query_points": int(len(self._query_points_yx)),
                "model_run_ms": float(run_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                "device": str(getattr(tracker, "device", self.device)),
                "dtype": str(getattr(tracker, "dtype", "")),
            },
        )


__all__ = ["LiteTrackerAdapter"]
