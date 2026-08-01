"""Bone-driven gaussian skinning (clean reimplementation of the PhysTwin
``gaussian_splatting/dynamic_utils.py`` math, torch, self-contained).

Model — incremental rollout, as in FuturePhysTwin/gs_render_dynamics.py:
- bones[t] are the trajectory points (object_points from final_data.pkl);
- per frame, each bone gets a rigid transform whose rotation comes from a
  Procrustes fit of its neighborhood edge vectors (prev -> curr) and whose
  translation is its own motion;
- each gaussian blends the K nearest bones' transforms with inverse-distance
  weights: positions as LBS in position space, rotations as normalized
  quaternion blend applied on the LEFT of the gaussian quaternion (wxyz).

Differences from the original: no ipdb traps (degenerate neighborhoods fall
back to identity rotation), no dense (N, B) weight matrix (indices are frozen
at bind time, values refreshed per frame — the trainer_warp variant), and
bone-quaternion hemisphere fixing before the blend.
"""

from __future__ import annotations

import torch


def build_bone_relations(bones: torch.Tensor, k: int) -> torch.Tensor:
    """(B, 3) -> (B, k) indices of each bone's k nearest other bones."""
    b = bones.shape[0]
    k = min(k, b - 1)
    dist = torch.cdist(bones, bones)
    return dist.topk(k + 1, largest=False).indices[:, 1:]


def bind_gaussians(bones: torch.Tensor, points: torch.Tensor, k: int) -> torch.Tensor:
    """(B, 3), (N, 3) -> (N, k) indices of the k nearest bones per gaussian."""
    k = min(k, bones.shape[0])
    dist = torch.cdist(points, bones)
    return dist.topk(k, largest=False).indices


def skin_weights(
    bones: torch.Tensor, points: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Inverse-distance weights of each point to its bound bones, rows sum 1."""
    diff = points[:, None, :] - bones[indices]
    weights = 1.0 / (diff.norm(dim=-1) + 1e-6)
    return weights / weights.sum(dim=1, keepdim=True)


def matrix_to_quat_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> (B, 4) unit wxyz quaternions (vectorized, branch-free)."""
    m = rotation
    b = m.shape[0]
    quats = m.new_zeros(b, 4)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Four candidate constructions; pick the numerically best per row.
    q0 = torch.stack(
        [1.0 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]],
        dim=1,
    )
    q1 = torch.stack(
        [m[:, 2, 1] - m[:, 1, 2], 1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
         m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]],
        dim=1,
    )
    q2 = torch.stack(
        [m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0],
         1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]],
        dim=1,
    )
    q3 = torch.stack(
        [m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0],
         m[:, 1, 2] + m[:, 2, 1], 1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]],
        dim=1,
    )
    candidates = torch.stack([q0, q1, q2, q3], dim=1)  # (B, 4, 4)
    norms = candidates.norm(dim=2)
    best = norms.argmax(dim=1)
    quats = candidates[torch.arange(b, device=m.device), best]
    quats = quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # Hemisphere: w >= 0 so downstream blending never averages antipodes.
    return quats * torch.sign(quats[:, :1] + 1e-12)


def quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def compute_bone_transforms(
    prev_bones: torch.Tensor,  # (B, 3)
    curr_bones: torch.Tensor,  # (B, 3)
    relations: torch.Tensor,  # (B, k)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-bone rigid transforms prev -> curr.

    Returns (rotations (B, 3, 3), translations (B, 3)); translation is the
    bone's own motion, rotation the Procrustes fit of its neighborhood edges.
    Degenerate neighborhoods (near-zero or near-collinear edges) fall back to
    identity instead of trapping.
    """
    edges_prev = prev_bones[relations] - prev_bones[:, None, :]  # (B, k, 3)
    edges_curr = curr_bones[relations] - curr_bones[:, None, :]
    covariance = edges_curr.transpose(1, 2) @ edges_prev  # (B, 3, 3)
    u, singular, vt = torch.linalg.svd(covariance)
    det = torch.linalg.det(u @ vt)
    sign = torch.ones_like(u[:, :, 0])
    sign[:, 2] = torch.sign(det)
    rotations = (u * sign[:, None, :]) @ vt  # Kabsch with reflection fix
    # Degenerate: neighborhood collapsed to (near) a line or point.
    scale = singular[:, 0].clamp_min(1e-12)
    degenerate = (singular[:, 1] / scale) < 1e-4
    if degenerate.any():
        eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        rotations = torch.where(degenerate[:, None, None], eye.expand_as(rotations), rotations)
    translations = curr_bones - prev_bones
    return rotations, translations


def apply_bone_transforms(
    means: torch.Tensor,  # (N, 3)
    quats_wxyz: torch.Tensor,  # (N, 4)
    prev_bones: torch.Tensor,  # (B, 3)
    rotations: torch.Tensor,  # (B, 3, 3)
    translations: torch.Tensor,  # (B, 3)
    indices: torch.Tensor,  # (N, k) frozen bone bindings
    chunk_size: int = 131072,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend per-bone rigid transforms onto gaussians (LBS + quat blend)."""
    bone_quats = matrix_to_quat_wxyz(rotations)  # (B, 4)
    new_means = torch.empty_like(means)
    new_quats = torch.empty_like(quats_wxyz)
    for start in range(0, means.shape[0], chunk_size):
        end = min(start + chunk_size, means.shape[0])
        idx = indices[start:end]  # (n, k)
        weights = skin_weights(prev_bones, means[start:end], idx)  # (n, k)
        local = means[start:end, None, :] - prev_bones[idx]  # (n, k, 3)
        rotated = torch.einsum("nkij,nkj->nki", rotations[idx], local)
        moved = rotated + prev_bones[idx] + translations[idx]
        new_means[start:end] = (weights[..., None] * moved).sum(dim=1)

        blended = (weights[..., None] * bone_quats[idx]).sum(dim=1)  # (n, 4)
        blended = blended / blended.norm(dim=1, keepdim=True).clamp_min(1e-12)
        composed = quat_multiply_wxyz(blended, quats_wxyz[start:end])
        new_quats[start:end] = composed / composed.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return new_means, new_quats
