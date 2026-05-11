from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult


DEFAULT_EXTERNAL_ROOT = Path("/home/zhangxinjie/external_tracking_backends")


class NvofaBackend:
    """NVOFA frame-to-frame flow propagation probe.

    This backend is intentionally dependency-gated. It is not long-term TAP;
    it propagates query points with adjacent-frame optical flow and should be
    periodically re-anchored in live use.
    """

    name = "nvofa"

    def __init__(
        self,
        *,
        external_root: str | Path | None = None,
        helper_binary: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.external_root = Path(external_root or os.environ.get("DEMO3_TRACKING_EXTERNAL_ROOT", DEFAULT_EXTERNAL_ROOT))
        self.helper_binary = Path(helper_binary or os.environ.get("DEMO3_NVOFA_HELPER", "")) if (helper_binary or os.environ.get("DEMO3_NVOFA_HELPER")) else None
        self.device = str(device)

    def _repo_exists(self) -> bool:
        return (self.external_root / "NVIDIAOpticalFlowSDK").exists()

    def _helper_path(self) -> Path | None:
        candidates = []
        if self.helper_binary is not None:
            candidates.append(self.helper_binary)
        candidates.extend(
            [
                self.external_root / "NVIDIAOpticalFlowSDK" / "run_nvofa_flow_helper",
                self.external_root / "NVIDIAOpticalFlowSDK" / "build" / "run_nvofa_flow_helper",
                self.external_root / "NVIDIAOpticalFlowSDK" / "build" / "NvOFTracker",
            ]
        )
        for path in candidates:
            if path.exists() and os.access(path, os.X_OK):
                return path
        found = shutil.which("run_nvofa_flow_helper")
        return Path(found) if found else None

    @staticmethod
    def _opencv_nvofa_available() -> bool:
        return any(hasattr(cv2, name) for name in ("cuda_NvidiaOpticalFlow_1_0", "cuda_NvidiaOpticalFlow_2_0"))

    def availability(self) -> BackendAvailability:
        if self._helper_path() is not None:
            return BackendAvailability(self.name, True, "NVOFA helper binary found")
        if self._opencv_nvofa_available():
            return BackendAvailability(self.name, True, "OpenCV CUDA NVIDIA Optical Flow binding found")
        if self._repo_exists():
            return BackendAvailability(self.name, False, "NVIDIA Optical Flow SDK repo found but helper/binding is not built")
        return BackendAvailability(self.name, False, "NVIDIA Optical Flow SDK repo/binary not found")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks
        if not self.is_available():
            raise BackendUnavailableError(self.availability_reason())

    def _flow_with_helper(self, prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
        helper = self._helper_path()
        if helper is None:
            raise BackendUnavailableError(self.availability_reason())
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            prev_path = tmp / "prev.png"
            next_path = tmp / "next.png"
            flow_path = tmp / "flow.npy"
            cv2.imwrite(str(prev_path), cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(next_path), cv2.cvtColor(next_rgb, cv2.COLOR_RGB2BGR))
            subprocess.run(
                [str(helper), "--prev", str(prev_path), "--next", str(next_path), "--out", str(flow_path)],
                check=True,
                text=True,
                capture_output=True,
            )
            flow = np.load(flow_path)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError(f"NVOFA helper flow must have shape H,W,2 xy; got {flow.shape}")
        return flow.astype(np.float32)

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
        if len(video_frames) < 1:
            raise ValueError("NVOFA requires at least one frame.")
        if query_points_yx is None:
            raise ValueError("query_points_yx is required.")
        if not self.is_available():
            raise BackendUnavailableError(self.availability_reason())
        queries = np.asarray(query_points_yx, dtype=np.float32)
        tracks = np.zeros((len(video_frames), queries.shape[0], 2), dtype=np.float32)
        tracks[0] = queries
        visibility = np.ones((len(video_frames), queries.shape[0]), dtype=np.float32)
        for frame_idx in range(1, len(video_frames)):
            flow = self._flow_with_helper(video_frames[frame_idx - 1], video_frames[frame_idx])
            yy = np.rint(tracks[frame_idx - 1, :, 0]).astype(np.int64)
            xx = np.rint(tracks[frame_idx - 1, :, 1]).astype(np.int64)
            in_bounds = (yy >= 0) & (yy < flow.shape[0]) & (xx >= 0) & (xx < flow.shape[1])
            visibility[frame_idx] = visibility[frame_idx - 1] * in_bounds.astype(np.float32)
            next_points = tracks[frame_idx - 1].copy()
            next_points[in_bounds, 1] += flow[yy[in_bounds], xx[in_bounds], 0]
            next_points[in_bounds, 0] += flow[yy[in_bounds], xx[in_bounds], 1]
            tracks[frame_idx] = next_points
        return TrackingResult(
            tracks_yx=tracks,
            visibility=visibility,
            backend=self.name,
            camera_idx=camera_idx,
            query_points_yx=queries,
            stats={"backend": self.name, "mode": "frame_to_frame_flow_propagation", "reanchor_required": True},
        )

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        raise NotImplementedError("NVOFA live update is reserved for a later realtime integration slice.")
