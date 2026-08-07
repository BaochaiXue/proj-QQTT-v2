"""Control-point-driven gaussian deformation (realtime-suitable, pure torch).

Vendored + cleaned from ``FuturePhysTwin/gaussian_splatting/dynamic_utils.py``
(the reference implementation behind ``gs_render_dynamics.py``'s offline
rollout). Changes vs upstream: the kornia import is dropped (upstream only
references it in dead code — the live path uses the local ``mat2quat``), the
ipdb debug traps become identity-rotation fallbacks, the dense
(particles x bones) code path is removed in favor of the sparse K-nearest
formulation (``interpolate_motions_sparse`` = upstream
``interpolate_motions_speedup``), and the per-frame KNN is chunked on GPU
instead of upstream's CPU round-trip.

Semantics (unchanged): bones are control points; per frame each bone gets a
rigid transform estimated from its K-adjacent bones' motion (weighted Kabsch
with reflection fixes); particles blend the K nearest bones' transforms with
inverse-distance weights — positions move and orientations rotate
(quaternion composition), which is what keeps anisotropic splats looking
right under rotation.
"""

from __future__ import annotations

import torch


def quat2mat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices."""
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)
    rot = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def mat2quat(rot: torch.Tensor) -> torch.Tensor:
    """(N,3,3) rotation matrices -> (N,4) wxyz quaternions (branchless masks)."""
    t = torch.clamp(rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], min=-1)
    q = torch.zeros((rot.shape[0], 4), device=rot.device, dtype=rot.dtype)

    mask_0 = t > -1
    t_0 = torch.sqrt(t[mask_0] + 1)
    q[mask_0, 0] = 0.5 * t_0
    t_0 = 0.5 / t_0
    q[mask_0, 1] = (rot[mask_0, 2, 1] - rot[mask_0, 1, 2]) * t_0
    q[mask_0, 2] = (rot[mask_0, 0, 2] - rot[mask_0, 2, 0]) * t_0
    q[mask_0, 3] = (rot[mask_0, 1, 0] - rot[mask_0, 0, 1]) * t_0

    mask_1 = ~mask_0 & (rot[:, 0, 0] >= rot[:, 1, 1]) & (rot[:, 0, 0] >= rot[:, 2, 2])
    t_1 = torch.sqrt(1 + rot[mask_1, 0, 0] - rot[mask_1, 1, 1] - rot[mask_1, 2, 2])
    t_1 = 0.5 / t_1
    q[mask_1, 0] = (rot[mask_1, 2, 1] - rot[mask_1, 1, 2]) * t_1
    q[mask_1, 1] = 0.5 / (2 * t_1)
    q[mask_1, 2] = (rot[mask_1, 1, 0] + rot[mask_1, 0, 1]) * t_1
    q[mask_1, 3] = (rot[mask_1, 2, 0] + rot[mask_1, 0, 2]) * t_1

    mask_2 = ~mask_0 & (rot[:, 1, 1] >= rot[:, 2, 2]) & (rot[:, 1, 1] > rot[:, 0, 0])
    t_2 = torch.sqrt(1 + rot[mask_2, 1, 1] - rot[mask_2, 0, 0] - rot[mask_2, 2, 2])
    t_2 = 0.5 / t_2
    q[mask_2, 0] = (rot[mask_2, 0, 2] - rot[mask_2, 2, 0]) * t_2
    q[mask_2, 1] = (rot[mask_2, 2, 1] + rot[mask_2, 1, 2]) * t_2
    q[mask_2, 2] = 0.5 / (2 * t_2)
    q[mask_2, 3] = (rot[mask_2, 0, 1] + rot[mask_2, 1, 0]) * t_2

    mask_3 = ~mask_0 & (rot[:, 2, 2] > rot[:, 0, 0]) & (rot[:, 2, 2] > rot[:, 1, 1])
    t_3 = torch.sqrt(1 + rot[mask_3, 2, 2] - rot[mask_3, 0, 0] - rot[mask_3, 1, 1])
    t_3 = 0.5 / t_3
    q[mask_3, 0] = (rot[mask_3, 1, 0] - rot[mask_3, 0, 1]) * t_3
    q[mask_3, 1] = (rot[mask_3, 0, 2] + rot[mask_3, 2, 0]) * t_3
    q[mask_3, 2] = (rot[mask_3, 1, 2] + rot[mask_3, 2, 1]) * t_3
    q[mask_3, 3] = 0.5 / (2 * t_3)
    return q


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """(N,4) x (N,4) wxyz Hamilton product."""
    q = torch.zeros_like(q1)
    q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    return q


def get_topk_indices(points: torch.Tensor, K: int = 5) -> torch.Tensor:
    """(N,3) -> (N,K) indices of each point's K nearest neighbors (no self)."""
    dist_matrix = torch.cdist(points, points, p=2)
    return torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]


def compute_bone_transforms(
    bones: torch.Tensor,
    motions: torch.Tensor,
    relations: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Per-bone rigid transforms from adjacent-bone motion (weighted Kabsch).

    Returns (n_bones, 4, 4); rotation from the SVD of the local neighborhood
    covariance with reflection fixes, translation = the bone's own motion.
    Degenerate neighborhoods (rank <= 1 or SVD failure) fall back to
    identity rotation — upstream dropped into ipdb there, which a realtime
    worker must never do.
    """
    n_bones = bones.shape[0]
    bone_transforms = torch.zeros((n_bones, 4, 4), device=device, dtype=torch.float32)
    bone_transforms[:, :3, :3] = torch.eye(3, device=device)
    bone_transforms[:, 3, 3] = 1.0
    bones = bones.to(device=device, dtype=torch.float32)
    motions = motions.to(device=device, dtype=torch.float32)
    relations = relations.to(device=device, dtype=torch.long)

    adj_bones = bones[relations] - bones[:, None]
    adj_bones_new = (bones[relations] + motions[relations]) - (
        bones[:, None] + motions[:, None]
    )
    F = adj_bones_new.permute(0, 2, 1) @ adj_bones

    cov_rank = torch.linalg.matrix_rank(F)
    solvable = cov_rank >= 2
    if bool(solvable.any()):
        try:
            U, _S, V = torch.svd(F[solvable])
            S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(
                int(solvable.sum()), 1, 1
            )
            neg_det_mask = torch.linalg.det(F[solvable]) < 0
            S[neg_det_mask, -1, -1] = -1
            R = U @ S @ V.permute(0, 2, 1)
            # A residual reflection means the neighborhood degenerated
            # mid-solve; flipping the last axis restores a proper rotation.
            neg_1 = torch.abs(torch.linalg.det(R) + 1) < 1e-3
            R[neg_1, -1, -1] *= -1
            bone_transforms[solvable, :3, :3] = R
        except Exception:
            pass  # identity rotations already in place

    bone_transforms[:, :3, 3] = motions
    return bone_transforms


def knn_weights_sparse(
    bones: torch.Tensor,
    pts: torch.Tensor,
    K: int = 16,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-distance weights + indices of each point's K nearest bones.

    Returns ``(weights (N,K), indices (N,K))``. Chunked over points so the
    (chunk x n_bones) distance matrix stays small on GPU.
    """
    k = min(int(K), bones.shape[0])
    weights_out = torch.empty((pts.shape[0], k), device=pts.device, dtype=pts.dtype)
    indices_out = torch.empty((pts.shape[0], k), device=pts.device, dtype=torch.long)
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start : start + chunk_size]
        dist = torch.cdist(chunk, bones)
        vals, idx = torch.topk(dist, k, dim=-1, largest=False)
        w = 1.0 / (vals + 1e-6)
        weights_out[start : start + chunk_size] = w / w.sum(dim=-1, keepdim=True)
        indices_out[start : start + chunk_size] = idx
    return weights_out, indices_out


def interpolate_motions_sparse(
    bones: torch.Tensor,
    motions: torch.Tensor,
    relations: torch.Tensor,
    xyz: torch.Tensor,
    quat: torch.Tensor | None,
    weights: torch.Tensor,
    weights_indices: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move (and rotate) particles by their K nearest bones' rigid motion.

    bones/motions: (n_bones, 3); relations: (n_bones, k_adj);
    xyz: (n_particles, 3); quat: (n_particles, 4) wxyz or None;
    weights/weights_indices: from ``knn_weights_sparse``.
    Returns (new_xyz, new_quat_or_None).
    """
    bone_transforms = compute_bone_transforms(bones, motions, relations, device=device)

    selected_bones = bones[weights_indices]  # (N, k, 3)
    selected_transforms = bone_transforms[weights_indices]  # (N, k, 4, 4)
    xyz_local = xyz.unsqueeze(1) - selected_bones
    rotated_local = torch.einsum(
        "nkij,nkj->nki", selected_transforms[:, :, :3, :3], xyz_local
    )
    transformed = rotated_local + selected_transforms[:, :, :3, 3] + selected_bones
    new_xyz = torch.sum(transformed * weights[:, :, None], dim=1)

    new_quat = None
    if quat is not None:
        base_quats = mat2quat(
            selected_transforms[:, :, :3, :3].reshape(-1, 3, 3)
        ).reshape(xyz.shape[0], -1, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)
        blended = torch.sum(base_quats * weights[:, :, None], dim=1)
        blended = torch.nn.functional.normalize(blended, dim=-1)
        new_quat = quaternion_multiply(blended, quat)
    return new_xyz, new_quat
