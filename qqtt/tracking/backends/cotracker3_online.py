from __future__ import annotations

import time
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

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks
        self._load_model()

    @staticmethod
    def _frames_to_torch_video(frames: Sequence[np.ndarray], *, device: str):
        import torch

        if not frames:
            raise ValueError("CoTracker3 requires at least one frame.")
        arr = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"frames_rgb must be HxWx3 RGB arrays; got {arr.shape}")
        return torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float().to(device)

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
        tracks_yx = tracks_xy.detach().cpu().numpy()[0].astype(np.float32)[:, :, ::-1]
        visibility_np = visibility.detach().cpu().numpy()[0].astype(np.float32)
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
            },
        )

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        raise NotImplementedError("Streaming update is reserved for the live Demo 3 integration slice.")
