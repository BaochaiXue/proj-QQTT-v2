"""Realtime tracking-driven gaussian deformation + rendering (FORMAL).

Consumed by the staged runtime's gaussian worker thread: per displayed
frame, the tracked OBJECT markers (world coords, identity-stable via
query_indices) act as control points ("bones"); the aligned world-frame
splats ride their motion via the vendored sparse motion interpolation
(gaussian_dynamics), and gsplat rasterizes the result over the live RGB.

Identity handling (the mis-association trap from the design review):
marker rows are a per-frame, variable-width subset — positions are
scattered by ``query_indices`` into a fixed (Q,3) buffer holding each
query's last-known position; a query missing this frame simply keeps its
previous position (zero motion), matching ASAP's occlusion semantics.
The bone set is frozen on the FIRST frame (queries seen then); relations
are computed once over that set.

Fail-soft: every public call catches internally-raised errors and flips
``self.failed`` — the worker then stops publishing the channel instead of
taking the pipeline down (display-only feature).
"""

from __future__ import annotations

import numpy as np

from demo_v7.service import gaussian_dynamics
from demo_v7.service.gaussian_utils import (
    load_gaussian_ply,
    render_gaussians,
    splats_to_tensors,
)

_BONE_RELATION_K = 8
_SKIN_K = 16
_MIN_BONES = 12
_OVERLAY_ALPHA = 0.85


class GaussianLiveRenderer:
    """Stateful deform+render loop over the aligned world splats."""

    def __init__(self, world_ply_path: str, *, device: str = "cuda") -> None:
        import torch

        self.device = device
        self.failed = False
        splats = load_gaussian_ply(world_ply_path)
        self._tensors = splats_to_tensors(splats, device=device)
        self._torch = torch
        self._bone_ids: np.ndarray | None = None  # frozen query-id subset
        self._relations = None  # (B, K) torch
        self._ctrl_prev = None  # (B, 3) torch, last-known bone positions
        self._buffer: dict[int, np.ndarray] = {}  # query id -> last position
        # Warm the gsplat CUDA extension before the live loop (a cold cache
        # would stall the first frame for minutes).
        try:
            render_gaussians(
                self._tensors,
                viewmat=np.eye(4),
                intrinsics=np.array([[300.0, 0, 32], [0, 300.0, 32], [0, 0, 1]]),
                width=64,
                height=64,
                device=device,
            )
        except Exception:
            self.failed = True
            raise

    def _scatter(self, marker_xyz: np.ndarray, query_ids: np.ndarray) -> None:
        for row, query_id in enumerate(query_ids):
            self._buffer[int(query_id)] = marker_xyz[row]

    def step(
        self,
        marker_xyz: np.ndarray,
        query_ids: np.ndarray,
        is_object: np.ndarray,
    ) -> None:
        """Advance the splats by this frame's object-marker motion."""
        if self.failed:
            return
        try:
            self._step(marker_xyz, query_ids, is_object)
        except Exception as exc:
            self.failed = True
            print(f"[gaussian-live] deform failed: {exc}", flush=True)

    def _step(
        self,
        marker_xyz: np.ndarray,
        query_ids: np.ndarray,
        is_object: np.ndarray,
    ) -> None:
        torch = self._torch
        keep = np.asarray(is_object, dtype=bool)
        self._scatter(
            np.asarray(marker_xyz, dtype=np.float32)[keep],
            np.asarray(query_ids, dtype=np.int64)[keep],
        )
        if self._bone_ids is None:
            if len(self._buffer) < _MIN_BONES:
                return  # not enough object markers yet
            self._bone_ids = np.array(sorted(self._buffer), dtype=np.int64)
            positions = np.stack([self._buffer[i] for i in self._bone_ids])
            self._ctrl_prev = torch.as_tensor(positions, device=self.device)
            self._relations = gaussian_dynamics.get_topk_indices(
                self._ctrl_prev, K=min(_BONE_RELATION_K, len(self._bone_ids) - 1)
            )
            return  # first frame defines the rest pose; no motion yet
        cur_np = np.stack([self._buffer[i] for i in self._bone_ids])
        ctrl_cur = torch.as_tensor(cur_np, device=self.device)
        motions = ctrl_cur - self._ctrl_prev
        if float(motions.abs().max()) < 1e-6:
            return  # nothing moved; skip the solve
        weights, indices = gaussian_dynamics.knn_weights_sparse(
            self._ctrl_prev, self._tensors["means"], K=_SKIN_K
        )
        new_means, new_quats = gaussian_dynamics.interpolate_motions_sparse(
            self._ctrl_prev,
            motions,
            self._relations,
            self._tensors["means"],
            self._tensors["quats"],
            weights,
            indices,
            device=self.device,
        )
        self._tensors["means"] = new_means
        if new_quats is not None:
            self._tensors["quats"] = torch.nn.functional.normalize(new_quats, dim=-1)
        self._ctrl_prev = ctrl_cur

    def render_over(
        self,
        frame_bgr: np.ndarray,
        *,
        viewmat: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray | None:
        """Render the current splats over the live frame (None on failure)."""
        if self.failed:
            return None
        try:
            height, width = frame_bgr.shape[:2]
            rgb, alpha = render_gaussians(
                self._tensors,
                viewmat=viewmat,
                intrinsics=intrinsics,
                width=int(width),
                height=int(height),
                background=(0.0, 0.0, 0.0),
                device=self.device,
            )
            blend = (alpha[..., None] * _OVERLAY_ALPHA).astype(np.float32)
            composed = (
                frame_bgr.astype(np.float32) * (1.0 - blend)
                + rgb[..., ::-1].astype(np.float32) * blend
            )
            return composed.astype(np.uint8)
        except Exception as exc:
            self.failed = True
            print(f"[gaussian-live] render failed: {exc}", flush=True)
            return None
