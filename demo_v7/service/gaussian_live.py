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
and the first packet becomes a one-shot catch-up pose. Without a seed the
old first-packet behavior remains.

Rest-ANCHORED deformation (the erosion trap, root-caused 2026-08-07 on a
448-frame operator session): an incremental prev->cur scheme that re-binds
skinning to the already-deformed splats every frame lets tracker jitter
and binding churn accumulate — the splat cloud visibly thins/tears after
a few hundred frames of manipulation. Every frame here is instead solved
FROM THE REST STATE: bindings (weights + indices + bone relations) are
computed once against the rest bones, and per-bone rotations come from
the TOTAL rest->current displacement (exact weighted Kabsch, valid for
arbitrary magnitude) applied to the pristine rest splats. Nothing ever
compounds; frame N is bit-identical whether reached live or by replay.

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
# knn_weights_sparse chunk: ~4k object bones x 16k splats per chunk keeps
# the per-step distance matrix around 256MB on the shared camera GPU.
_KNN_CHUNK = 16384
# Bone hygiene (measured on a real 542-frame operator session: 71/3969
# bones went rogue — track slid onto the hand / depth-boundary lift, up to
# 17cm off-object — and up to 160 at a time sat occlusion-frozen >2s;
# with the frozen rest binding both classes drag their splats visibly).
# A bone whose rest->current displacement deviates from its rest-neighbor
# median by more than this is a rogue track this frame:
_BONE_RIGID_DEV_M = 0.05
# A bone unseen for this many packets (~2s at 5Hz) is stale — its
# last-known world position anchors old pose instead of riding the object:
_BONE_STALE_STEPS = 10
# Rogue/stale bones ride the mean displacement of their VALID rest
# neighbors instead; need at least this many valid neighbors, else the
# global valid-median displacement (fully-occluded patch fallback):
_BONE_MIN_VALID_NEIGHBORS = 3
# Rigid-aware healing (the drag fix, A/B'd on a 132-frame manipulation
# capture): a grabbed patch is OCCLUDED exactly while it ROTATES, and a
# translation-average of valid neighbors collapses rotation — the healed
# patch lags and its splats smear ("拖拽"). Extrapolating each invalid
# bone through the valid bones' local RIGID motion field (the same LBS the
# splats use) preserves rotation: worst-frame IoU 0.745 -> 0.770, peak
# spill -6%. Falls back to the mean consensus below this valid count.
_HEAL_RIGID_MIN_VALID = 24
_HEAL_RELATION_K = 8
_HEAL_SKIN_K = 8


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
        self._relations = None  # (B, K) torch, on the REST bones
        self._ctrl_rest = None  # (B, 3) torch, frozen rest bone positions
        self._ctrl_prev = None  # (B, 3) torch, last applied bone positions
        self._rest_means = None  # pristine rest splat positions
        self._rest_quats = None  # pristine rest splat orientations
        self._skin_weights = None  # (N, K) one-time rest binding
        self._skin_indices = None  # (N, K) one-time rest binding
        self._buffer: dict[int, np.ndarray] = {}  # query id -> last position
        self._rest_positions: dict[int, np.ndarray] = {}  # seq-0 world xyz
        self.rest_seeded = False  # bones initialized from seq-0 rest pose
        self._seed_grace_left = 25  # packets to wait for a seedable set
        self._last_seen_step: np.ndarray | None = None  # per-bone packet idx
        self.frames_stepped = 0
        # Cumulative follow telemetry (world meters), read by the worker.
        self.bones_moved_m = 0.0
        self.splats_moved_m = 0.0
        # Last hygiene-pass counts (rogue tracks / occlusion-stale bones).
        self.bone_outliers = 0
        self.bone_stale = 0
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
                self._ctrl_rest = torch.as_tensor(rest, device=self.device)
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
            self._ctrl_rest = torch.as_tensor(
                positions.astype(np.float32), device=self.device
            )
        # Freeze the WHOLE rest state once: pristine splats, bone graph and
        # skinning binding. Every later frame is solved from here — binding
        # churn and incremental error cannot accumulate by construction.
        self._ctrl_prev = self._ctrl_rest
        self._rest_means = self._tensors["means"].clone()
        self._rest_quats = self._tensors["quats"].clone()
        self._relations = gaussian_dynamics.get_topk_indices(
            self._ctrl_rest, K=min(_BONE_RELATION_K, len(self._bone_ids) - 1)
        )
        self._skin_weights, self._skin_indices = gaussian_dynamics.knn_weights_sparse(
            self._ctrl_rest,
            self._skin_targets(),
            K=_SKIN_K,
            chunk_size=_KNN_CHUNK,
        )
        return True

    def _skin_targets(self):
        """Rest points the bones bind to (subclass hook: mesh vertices)."""
        return self._rest_means

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
            self._last_seen_step = np.full(
                len(self._bone_ids), self.frames_stepped, dtype=np.int64
            )
            if not self.rest_seeded:
                return  # rest pose = this packet; no motion to apply yet
        packet_ids = np.asarray(query_ids, dtype=np.int64)[keep]
        seen = np.isin(self._bone_ids, packet_ids)
        self._last_seen_step[seen] = self.frames_stepped
        cur_np = np.stack([self._buffer[i] for i in self._bone_ids])
        ctrl_target = torch.as_tensor(
            cur_np.astype(np.float32), device=self.device
        )
        stale = torch.as_tensor(
            (self.frames_stepped - self._last_seen_step) > _BONE_STALE_STEPS,
            device=self.device,
        )
        self._pose_to(self._hygiene(ctrl_target, stale))

    def _hygiene(self, ctrl_target, stale):
        """Replace rogue/stale bone targets with neighbor consensus.

        Measured failure classes on a real operator session (both are
        amplified by the frozen rest binding, which gives every bone
        PERMANENT influence over its splats):
        - rogue tracks: the 2D track slides onto the hand or a depth
          boundary and the lift lands centimeters off the object — its
          splats get dragged into free space;
        - occlusion-stale bones: an unseen marker keeps its last-known
          world position and anchors its splats in the old pose while the
          object moves on.
        Detection is LOCAL RIGIDITY on the frozen rest-neighbor graph: a
        plush's rest neighborhood moves together, so a bone whose
        rest->current displacement deviates from the neighborhood median
        by >_BONE_RIGID_DEV_M is a rogue this frame. Rogue and stale bones
        ride the mean displacement of their VALID neighbors instead (whole
        patch occluded -> global valid-median fallback), so they follow
        the object rather than freezing or flying off.
        """
        torch = self._torch
        disp = ctrl_target - self._ctrl_rest
        nbr_disp = disp[self._relations]  # (B, K, 3)
        # Stale bones carry frozen last-known positions — known-bad data
        # that must not vote in the rigidity reference (a half-occluded
        # patch would otherwise drag the median and flag the GOOD bones).
        nbr_for_median = nbr_disp.clone()
        nbr_for_median[stale[self._relations]] = float("nan")
        median = nbr_for_median.nanmedian(dim=1).values  # NaN if all stale
        deviation = (disp - median).norm(dim=1)
        outlier = (
            ~stale & torch.isfinite(deviation) & (deviation > _BONE_RIGID_DEV_M)
        )
        invalid = outlier | stale
        self.bone_outliers = int(outlier.sum())
        self.bone_stale = int(stale.sum())
        if not bool(invalid.any()):
            return ctrl_target
        valid = ~invalid
        if not bool(valid.any()):
            return ctrl_target  # nothing trustworthy to lean on
        if int(valid.sum()) >= _HEAL_RIGID_MIN_VALID:
            try:
                return self._heal_rigid(ctrl_target, disp, invalid, valid)
            except Exception as exc:
                print(
                    f"[gaussian-live] rigid healing fell back to consensus "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
        valid_nbr = valid[self._relations]  # (B, K)
        valid_count = valid_nbr.sum(dim=1)
        consensus = (nbr_disp * valid_nbr[..., None]).sum(dim=1) / (
            valid_count.clamp(min=1)[:, None]
        )
        fallback = disp[valid].median(dim=0).values
        consensus = torch.where(
            valid_count[:, None] >= _BONE_MIN_VALID_NEIGHBORS,
            consensus,
            fallback.expand_as(consensus),
        )
        healed = torch.where(
            invalid[:, None], self._ctrl_rest + consensus, ctrl_target
        )
        return healed

    def _heal_rigid(self, ctrl_target, disp, invalid, valid):
        """Extrapolate invalid bones through the valid bones' rigid field."""
        rest_valid = self._ctrl_rest[valid]
        disp_valid = disp[valid]
        relations = gaussian_dynamics.get_topk_indices(
            rest_valid, K=min(_HEAL_RELATION_K, rest_valid.shape[0] - 1)
        )
        rest_invalid = self._ctrl_rest[invalid]
        weights, indices = gaussian_dynamics.knn_weights_sparse(
            rest_valid, rest_invalid, K=_HEAL_SKIN_K
        )
        healed_pos, _quats = gaussian_dynamics.interpolate_motions_sparse(
            rest_valid,
            disp_valid,
            relations,
            rest_invalid,
            None,
            weights,
            indices,
            device=self.device,
        )
        healed = ctrl_target.clone()
        healed[invalid] = healed_pos
        return healed

    def _pose_to(self, ctrl_target) -> None:
        """Solve the splat pose for ``ctrl_target`` from the REST state.

        Per-bone rotations come from the TOTAL rest->target displacement
        field (exact weighted Kabsch — valid for arbitrary magnitude, no
        substepping needed) applied to the pristine rest splats through the
        one-time rest binding. Frame N's pose depends only on frame N's
        bones: tracker jitter and binding churn cannot accumulate, which is
        what eroded the splat cloud over long sessions in the incremental
        prev->cur scheme this replaces.
        """
        torch = self._torch
        step = ctrl_target - self._ctrl_prev
        if float(step.norm(dim=1).max()) < 1e-6:
            return  # nothing moved since the last applied pose
        self.bones_moved_m += float(step.norm(dim=1).mean())
        centroid_before = self._tensors["means"].mean(dim=0)
        total = ctrl_target - self._ctrl_rest
        new_means, new_quats = gaussian_dynamics.interpolate_motions_sparse(
            self._ctrl_rest,
            total,
            self._relations,
            self._rest_means,
            self._rest_quats,
            self._skin_weights,
            self._skin_indices,
            device=self.device,
        )
        self._tensors["means"] = new_means
        if new_quats is not None:
            self._tensors["quats"] = torch.nn.functional.normalize(
                new_quats, dim=-1
            )
        self._ctrl_prev = ctrl_target
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
            "bone_outliers": int(self.bone_outliers),
            "bone_stale": int(self.bone_stale),
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


class MeshAnchoredGaussianRenderer(GaussianLiveRenderer):
    """mesh_surface backend: splats ride the mesh, the mesh rides the bones.

    The splats were DERIVED from the aligned world mesh (face_id +
    barycentric anchors, see mesh_surface_gaussian). Instead of skinning
    each splat to the bones directly, the bones deform the MESH VERTICES
    (same rest-anchored LBS + hygiene as the parent), and every splat is
    replayed from its triangle: center = barycentric combination, splat
    orientation = the deformed face's tangent frame. Splats therefore stay
    ON the deformed mesh surface by construction — the mesh remains the
    single geometry truth in motion, and a rogue bone can at worst deform
    the mesh, never tear splats off it.

    (v1 note: the deformed vertices come from the SAME bone LBS the
    triposplat path uses, not from ASAP — ASAP's per-frame vertices live in
    the orchestrator process on a re-cleaned topology, one chunk (~1.2s)
    behind live; reusing them is a recorded follow-up, not a v1 blocker.)
    """

    def __init__(
        self, world_ply_path: str, anchors_path: str, *, device: str = "cuda"
    ) -> None:
        super().__init__(world_ply_path, device=device)
        from demo_v7.service.mesh_surface_gaussian import load_anchors

        torch = self._torch
        anchors = load_anchors(anchors_path)
        if len(anchors) != self._tensors["means"].shape[0]:
            raise ValueError(
                f"anchors ({len(anchors)}) and world ply "
                f"({self._tensors['means'].shape[0]}) disagree on splat "
                "count — mixed artifacts from different generations"
            )
        self._verts = torch.as_tensor(anchors.rest_vertices, device=device)
        self._faces = torch.as_tensor(
            anchors.faces.astype(np.int64), device=device
        )
        self._anchor_face = torch.as_tensor(
            anchors.face_index.astype(np.int64), device=device
        )
        self._anchor_bary = torch.as_tensor(anchors.barycentric, device=device)
        self._face_quats_prev = None  # degenerate-face carryover, set below
        means, quats = self._replay(self._verts)
        drift = float((means - self._tensors["means"]).norm(dim=1).max())
        if drift > 1e-3:
            raise ValueError(
                f"anchor replay disagrees with the world ply by {drift:.4f} m "
                "— anchors were built against a different mesh/ply pair"
            )
        # Canonicalize both representations to the replay (float32-exact
        # binding from here on).
        self._tensors["means"] = means
        self._tensors["quats"] = quats

    def _skin_targets(self):
        """Bones bind to the mesh vertices, not to the splats."""
        return self._verts

    def _replay(self, verts):
        """(means, quats) for all splats from a (possibly deformed) verts."""
        torch = self._torch
        corners = verts[self._faces[self._anchor_face]]  # (N,3,3)
        means = (self._anchor_bary.unsqueeze(-1) * corners).sum(dim=1)
        tri = verts[self._faces]  # (F,3,3)
        edge1 = tri[:, 1] - tri[:, 0]
        normal = torch.cross(edge1, tri[:, 2] - tri[:, 0], dim=1)
        edge1_len = edge1.norm(dim=1)
        normal_len = normal.norm(dim=1)
        ok = (edge1_len > 1e-9) & (normal_len > 1e-9)
        t1 = edge1 / edge1_len.clamp(min=1e-9)[:, None]
        n_hat = normal / normal_len.clamp(min=1e-9)[:, None]
        frames = torch.stack([t1, torch.cross(n_hat, t1, dim=1), n_hat], dim=2)
        face_quats = gaussian_dynamics.mat2quat(frames)
        if self._face_quats_prev is not None:
            # A transiently-degenerate face keeps its last orientation for
            # the frame instead of emitting NaNs into the rasterizer.
            face_quats = torch.where(
                ok[:, None], face_quats, self._face_quats_prev
            )
        self._face_quats_prev = face_quats
        quats = torch.nn.functional.normalize(
            face_quats[self._anchor_face], dim=-1
        )
        return means, quats

    def apply_rigid_transform(self, transform: np.ndarray) -> None:
        """Rigidly move the MESH; splats follow through their anchors."""
        torch = self._torch
        try:
            rotation = torch.as_tensor(
                np.asarray(transform[:3, :3], dtype=np.float32), device=self.device
            )
            translation = torch.as_tensor(
                np.asarray(transform[:3, 3], dtype=np.float32), device=self.device
            )
            new_verts = self._verts @ rotation.T + translation
            means, quats = self._replay(new_verts)
            # Commit all three or none — a mid-way failure must not leave
            # splats detached from the vertices they are anchored to.
            self._verts = new_verts
            self._tensors["means"] = means
            self._tensors["quats"] = quats
        except Exception as exc:
            print(f"[gaussian-live] rigid catch-up skipped: {exc}", flush=True)

    def _pose_to(self, ctrl_target) -> None:
        """Deform the mesh vertices from REST, then replay the anchors."""
        step = ctrl_target - self._ctrl_prev
        if float(step.norm(dim=1).max()) < 1e-6:
            return
        self.bones_moved_m += float(step.norm(dim=1).mean())
        centroid_before = self._tensors["means"].mean(dim=0)
        total = ctrl_target - self._ctrl_rest
        new_verts, _ = gaussian_dynamics.interpolate_motions_sparse(
            self._ctrl_rest,
            total,
            self._relations,
            self._verts_rest,
            None,
            self._skin_weights,
            self._skin_indices,
            device=self.device,
        )
        means, quats = self._replay(new_verts)
        self._verts = new_verts
        self._tensors["means"] = means
        self._tensors["quats"] = quats
        self._ctrl_prev = ctrl_target
        self.splats_moved_m += float(
            (self._tensors["means"].mean(dim=0) - centroid_before).norm()
        )

    def _init_bones(self) -> bool:
        """Freeze the rest state; also freeze the rest VERTICES snapshot."""
        ready = super()._init_bones()
        if ready:
            self._verts_rest = self._verts.clone()
        return ready

    def follow_stats(self, *, max_bones: int = 512) -> dict | None:
        stats = super().follow_stats(max_bones=max_bones)
        if stats is not None:
            stats["mesh_anchored"] = True
        return stats
