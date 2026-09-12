"""Soft rest-shape projection for mesh vertices after tracking-driven LBS.

The constraint is local and articulation-friendly: every triangle proposes
its closest rest-shape rigid copy at the triangle's current centroid and
orientation. Shared-vertex proposals are rest-area weighted and softly blended
back toward the unmodified LBS result. Rigid motion is an exact fixed point.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FaceRigidShapeConstraint:
    """Precomputed rest data for a triangle-rigid mesh projection."""

    faces: torch.Tensor
    rest_offsets: torch.Tensor
    rest_frames: torch.Tensor
    rest_double_area: torch.Tensor
    vertex_count: int


def _face_frames(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (right-handed frames, valid mask, doubled areas) per face."""
    tri = vertices[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    normal = torch.cross(edge1, tri[:, 2] - tri[:, 0], dim=1)
    edge1_len = edge1.norm(dim=1)
    double_area = normal.norm(dim=1)
    eps = torch.finfo(vertices.dtype).eps * 32.0
    valid = (edge1_len > eps) & (double_area > eps)
    t1 = edge1 / edge1_len.clamp_min(eps)[:, None]
    n_hat = normal / double_area.clamp_min(eps)[:, None]
    t2 = torch.cross(n_hat, t1, dim=1)
    frames = torch.stack([t1, t2, n_hat], dim=2)
    identity = torch.eye(3, dtype=vertices.dtype, device=vertices.device)
    frames = torch.where(valid[:, None, None], frames, identity[None])
    return frames, valid, double_area


def build_face_rigid_shape_constraint(
    rest_vertices: torch.Tensor, faces: torch.Tensor
) -> FaceRigidShapeConstraint:
    """Precompute rest triangle frames/offsets for the projection."""
    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        raise ValueError("rest_vertices must be (V,3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F,3)")
    if faces.numel() == 0:
        raise ValueError("faces must be non-empty")
    faces = faces.to(device=rest_vertices.device, dtype=torch.long)
    if int(faces.min()) < 0 or int(faces.max()) >= int(rest_vertices.shape[0]):
        raise ValueError("faces contain out-of-range vertex indices")
    tri = rest_vertices[faces]
    rest_centroids = tri.mean(dim=1)
    rest_offsets = tri - rest_centroids[:, None]
    rest_frames, valid, double_area = _face_frames(rest_vertices, faces)
    if not bool(valid.any()):
        raise ValueError("mesh has no non-degenerate triangle")
    # Degenerate rest faces carry zero weight and therefore cannot inject an
    # arbitrary identity-frame proposal into their vertices.
    rest_double_area = torch.where(valid, double_area, torch.zeros_like(double_area))
    return FaceRigidShapeConstraint(
        faces=faces,
        rest_offsets=rest_offsets,
        rest_frames=rest_frames,
        rest_double_area=rest_double_area,
        vertex_count=int(rest_vertices.shape[0]),
    )


def project_face_rigid_shape(
    lbs_vertices: torch.Tensor,
    constraint: FaceRigidShapeConstraint,
    *,
    iterations: int = 2,
    strength: float = 0.5,
    max_correction_m: float | None = 0.008,
) -> torch.Tensor:
    """Softly project an LBS mesh toward locally rigid rest triangles.

    The LBS vertices remain the data term on every iteration. Each face keeps
    its current centroid and orientation, but proposes its REST offsets under
    that rigid transform. The area-weighted incident proposals are blended
    against the original LBS result, so bending/articulation remains free while
    stretch/shear is damped. The final displacement from LBS is optionally
    bounded per vertex.
    """
    if lbs_vertices.ndim != 2 or lbs_vertices.shape != (constraint.vertex_count, 3):
        raise ValueError(
            f"lbs_vertices must be ({constraint.vertex_count},3), got "
            f"{tuple(lbs_vertices.shape)}"
        )
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0,1]")
    if max_correction_m is not None and float(max_correction_m) <= 0.0:
        raise ValueError("max_correction_m must be positive or None")
    if iterations == 0 or strength == 0.0:
        return lbs_vertices

    faces = constraint.faces
    rest_frames = constraint.rest_frames
    rest_offsets = constraint.rest_offsets
    face_weight = constraint.rest_double_area
    corner_index = faces.reshape(-1)
    verts = lbs_vertices
    for _ in range(int(iterations)):
        tri = verts[faces]
        centroids = tri.mean(dim=1)
        frames, valid, _double_area = _face_frames(verts, faces)
        # R maps a rest-frame vector into the current face frame.
        rotation = frames @ rest_frames.transpose(1, 2)
        target = centroids[:, None] + torch.einsum(
            "fij,fkj->fki", rotation, rest_offsets
        )
        weights = (face_weight * valid.to(face_weight.dtype))[:, None]
        corner_weight = weights.expand(-1, 3).reshape(-1, 1)
        accum = torch.zeros_like(lbs_vertices)
        weight_sum = torch.zeros(
            (constraint.vertex_count, 1),
            dtype=lbs_vertices.dtype,
            device=lbs_vertices.device,
        )
        accum.index_add_(0, corner_index, target.reshape(-1, 3) * corner_weight)
        weight_sum.index_add_(0, corner_index, corner_weight)
        supported = weight_sum[:, 0] > 0
        average = torch.where(
            supported[:, None],
            accum / weight_sum.clamp_min(torch.finfo(lbs_vertices.dtype).eps),
            lbs_vertices,
        )
        proposed = lbs_vertices + float(strength) * (average - lbs_vertices)
        if max_correction_m is not None:
            delta = proposed - lbs_vertices
            norm = delta.norm(dim=1, keepdim=True)
            scale = (float(max_correction_m) / norm.clamp_min(1e-12)).clamp(max=1.0)
            proposed = lbs_vertices + delta * scale
        verts = proposed
    return verts
