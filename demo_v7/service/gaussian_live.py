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

Rest pose (the stuck-in-frame-0-pose trap, root-caused 2026-08-07): the
tracker seeds its queries on FORMAL frame seq 0, but this worker starts
late on a latest-wins slot — freezing bones on the first packet it happens
to see silently discards all object motion since seq 0. The caller
therefore seeds the seq-0 world positions of the SAME query ids via
``seed_rest_positions`` (reconstructed from prepared_phystwin/000000.npz),
and the first packet becomes a substepped catch-up deformation instead of
a new rest pose. Without a seed the old first-packet behavior remains.

Fail-soft: every public call catches internally-raised errors and flips
``self.failed`` — the worker then stops publishing the channel instead of
taking the pipeline down (display-only feature).
"""

from __future__ import annotations

import math

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
# Catch-up substepping: the interpolation's per-bone rigid estimation is
# built for small per-frame motions, so a large one-shot displacement
# (seq-0 rest -> current pose after this worker's slow start) is applied
# as a sequence of small steps along the straight-line control path.
_SUBSTEP_MAX_M = 0.02
_MAX_SUBSTEPS = 25
# knn_weights_sparse chunk: ~4k object bones x 16k splats per chunk keeps
# the per-step distance matrix around 256MB on the shared camera GPU.
_KNN_CHUNK = 16384


def load_formal_frame0_rest_positions(
    npz_path,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """(query id -> seq-0 world xyz, seq-0 object cloud) from a prepared frame.

    ``prepared_phystwin/000000.npz`` stores the FULL per-query tracker
    arrays plus the world-frame pcd grid; sampling the grid at the rounded
    seq-0 track pixel reproduces the tracker's own ``lift_tracks_yx_to_world``
    backprojection exactly (verified to 4.3e-8 m on a real run). Only
    queries that are visible AND land on the object mask become bones.
    """
    data = np.load(npz_path)
    tracks = np.asarray(data["tracks_yx"], dtype=np.float64)
    visibility = np.asarray(data["visibility"], dtype=bool)
    pcd_grid = np.asarray(data["pcd_points"][0], dtype=np.float32)
    mask_object = np.asarray(data["mask_object"], dtype=bool)
    height, width = mask_object.shape
    rows = np.clip(np.rint(tracks[:, 0]).astype(np.int64), 0, height - 1)
    cols = np.clip(np.rint(tracks[:, 1]).astype(np.int64), 0, width - 1)
    world = pcd_grid[rows, cols]
    valid = visibility & mask_object[rows, cols] & np.isfinite(world).all(axis=1)
    rest = {int(q): world[q].copy() for q in np.flatnonzero(valid)}
    object_cloud = pcd_grid[mask_object]
    object_cloud = object_cloud[np.isfinite(object_cloud).all(axis=1)]
    return rest, object_cloud


def whiten_background(frame_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Blend a frame toward white (amount in [0,1]) as float32 BGR."""
    base = frame_bgr.astype(np.float32)
    amount = float(min(max(amount, 0.0), 1.0))
    if amount <= 0.0:
        return base
    return base * (1.0 - amount) + 255.0 * amount


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
        self._rest_positions: dict[int, np.ndarray] = {}  # seq-0 world xyz
        self.rest_seeded = False  # bones initialized from seq-0 rest pose
        self._seed_grace_left = 25  # packets to wait for a seedable set
        self.frames_stepped = 0
        self.last_substeps = 0
        # Cumulative follow telemetry (world meters), read by the worker.
        self.bones_moved_m = 0.0
        self.splats_moved_m = 0.0
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

    def seed_rest_positions(self, rest: dict[int, np.ndarray]) -> None:
        """Provide the queries' FORMAL seq-0 world positions (pre-loop)."""
        self._rest_positions = {int(k): np.asarray(v) for k, v in rest.items()}

    def apply_rigid_transform(self, transform: np.ndarray) -> None:
        """Rigidly move all splats (means AND orientations) in world frame."""
        torch = self._torch
        try:
            rotation = torch.as_tensor(
                np.asarray(transform[:3, :3], dtype=np.float32), device=self.device
            )
            translation = torch.as_tensor(
                np.asarray(transform[:3, 3], dtype=np.float32), device=self.device
            )
            new_means = self._tensors["means"] @ rotation.T + translation
            rot_quat = gaussian_dynamics.mat2quat(rotation[None])
            rot_quat = rot_quat.expand(self._tensors["quats"].shape[0], 4)
            new_quats = torch.nn.functional.normalize(
                gaussian_dynamics.quaternion_multiply(
                    rot_quat, self._tensors["quats"]
                ),
                dim=-1,
            )
            # Commit both or neither: a mid-way failure must not leave
            # moved positions with pre-transform orientations.
            self._tensors["means"] = new_means
            self._tensors["quats"] = new_quats
        except Exception as exc:
            # A skipped rigid catch-up degrades alignment, not availability.
            print(f"[gaussian-live] rigid catch-up skipped: {exc}", flush=True)

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

    def _init_bones(self) -> bool:
        """Freeze the bone set; True once bones exist (rest-seeded or not)."""
        torch = self._torch
        if len(self._buffer) < _MIN_BONES:
            return False
        if self._rest_positions:
            ids = sorted(set(self._buffer) & set(self._rest_positions))
            if len(ids) >= _MIN_BONES:
                self._bone_ids = np.array(ids, dtype=np.int64)
                rest = np.stack(
                    [self._rest_positions[i] for i in ids]
                ).astype(np.float32)
                self._ctrl_prev = torch.as_tensor(rest, device=self.device)
                self.rest_seeded = True
            elif self._seed_grace_left > 0:
                # A marginal first packet (occlusion, depth dropouts) must
                # not permanently forfeit rest seeding — the buffer is a
                # growing union, so give later packets a chance to satisfy
                # the intersection before freezing an unseeded bone set.
                self._seed_grace_left -= 1
                return False
            else:
                print(
                    f"[gaussian-live] only {len(ids)} buffered queries have a "
                    "seq-0 rest position; falling back to first-packet rest "
                    "pose",
                    flush=True,
                )
        if self._bone_ids is None:
            self._bone_ids = np.array(sorted(self._buffer), dtype=np.int64)
            positions = np.stack([self._buffer[i] for i in self._bone_ids])
            self._ctrl_prev = torch.as_tensor(
                positions.astype(np.float32), device=self.device
            )
        self._relations = gaussian_dynamics.get_topk_indices(
            self._ctrl_prev, K=min(_BONE_RELATION_K, len(self._bone_ids) - 1)
        )
        return True

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
        self.frames_stepped += 1
        if self._bone_ids is None:
            if not self._init_bones():
                return  # not enough object markers yet
            if not self.rest_seeded:
                return  # rest pose = this packet; no motion to apply yet
        cur_np = np.stack([self._buffer[i] for i in self._bone_ids])
        ctrl_target = torch.as_tensor(
            cur_np.astype(np.float32), device=self.device
        )
        self._move_to(ctrl_target)

    def _move_to(self, ctrl_target) -> None:
        """Deform splats along the control path to ``ctrl_target``.

        Motions beyond _SUBSTEP_MAX_M per bone are applied in straight-line
        substeps (skinning weights recomputed each substep) so the per-bone
        rigid estimation always sees a small motion — this is what makes
        the one-shot seq-0 catch-up safe.
        """
        torch = self._torch
        total = ctrl_target - self._ctrl_prev
        max_disp = float(total.norm(dim=1).max())
        if max_disp < 1e-6:
            self.last_substeps = 0
            return  # nothing moved; skip the solve
        substeps = int(min(_MAX_SUBSTEPS, max(1, math.ceil(max_disp / _SUBSTEP_MAX_M))))
        self.last_substeps = substeps
        self.bones_moved_m += float(total.norm(dim=1).mean())
        start = self._ctrl_prev
        centroid_before = self._tensors["means"].mean(dim=0)
        for index in range(1, substeps + 1):
            target = start + total * (index / substeps)
            motions = target - self._ctrl_prev
            weights, indices = gaussian_dynamics.knn_weights_sparse(
                self._ctrl_prev,
                self._tensors["means"],
                K=_SKIN_K,
                chunk_size=_KNN_CHUNK,
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
                self._tensors["quats"] = torch.nn.functional.normalize(
                    new_quats, dim=-1
                )
            self._ctrl_prev = target
        self.splats_moved_m += float(
            (self._tensors["means"].mean(dim=0) - centroid_before).norm()
        )

    def follow_stats(self, *, max_bones: int = 512) -> dict | None:
        """Bone->nearest-splat distances (cm): the 'is it following' metric.

        Bones ride the REAL object surface; if the splats track it, every
        bone has splats nearby (p50 ~1-2cm). A detached limb shows up as an
        exploding p90 long before a human squints at the overlay.
        """
        if self._ctrl_prev is None:
            return None
        torch = self._torch
        with torch.no_grad():
            bones = self._ctrl_prev
            if bones.shape[0] > max_bones:
                stride = -(-bones.shape[0] // max_bones)  # ceil -> <= max_bones
                bones = bones[::stride]
            # Chunk over splats (running min) — an unchunked (B, N) cdist
            # would transiently allocate ~1GB on the shared camera GPU.
            means = self._tensors["means"]
            distances = torch.full(
                (bones.shape[0],), float("inf"), device=bones.device
            )
            for start in range(0, means.shape[0], _KNN_CHUNK):
                chunk = torch.cdist(bones, means[start : start + _KNN_CHUNK])
                distances = torch.minimum(distances, chunk.min(dim=1).values)
            quantiles = torch.quantile(
                distances, torch.tensor([0.5, 0.9], device=distances.device)
            )
        return {
            "bones": int(self._ctrl_prev.shape[0]),
            "rest_seeded": bool(self.rest_seeded),
            "failed": bool(self.failed),
            "frames_stepped": int(self.frames_stepped),
            "bones_moved_cm": round(self.bones_moved_m * 100.0, 2),
            "splats_moved_cm": round(self.splats_moved_m * 100.0, 2),
            "bone2splat_p50_cm": round(float(quantiles[0]) * 100.0, 2),
            "bone2splat_p90_cm": round(float(quantiles[1]) * 100.0, 2),
        }

    def render_over(
        self,
        frame_bgr: np.ndarray,
        *,
        viewmat: np.ndarray,
        intrinsics: np.ndarray,
        background_whiten: float = 0.0,
    ) -> np.ndarray | None:
        """Render the current splats over the live frame (None on failure).

        ``background_whiten`` blends the camera frame toward white first so
        the splats read clearly against the (busy) table surface.
        """
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
            base = whiten_background(frame_bgr, background_whiten)
            blend = (alpha[..., None] * _OVERLAY_ALPHA).astype(np.float32)
            # rgb is already alpha-premultiplied (composited over the black
            # background by gsplat) — multiplying it by blend again would
            # double-count alpha and draw dark halos around soft splat
            # edges. Over-composite: fg_premult * opacity + bg * (1 - a*o).
            composed = (
                base * (1.0 - blend)
                + rgb[..., ::-1].astype(np.float32) * _OVERLAY_ALPHA
            )
            return np.clip(composed, 0.0, 255.0).astype(np.uint8)
        except Exception as exc:
            self.failed = True
            print(f"[gaussian-live] render failed: {exc}", flush=True)
            return None
