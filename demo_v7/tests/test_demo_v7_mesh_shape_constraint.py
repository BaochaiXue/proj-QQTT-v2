from __future__ import annotations

import math

import torch

from demo_v7.service.mesh_shape_constraint import (
    build_face_rigid_shape_constraint,
    project_face_rigid_shape,
)


def _grid_mesh(rows: int = 8, cols: int = 10, spacing: float = 0.015):
    yy, xx = torch.meshgrid(
        torch.arange(rows, dtype=torch.float32),
        torch.arange(cols, dtype=torch.float32),
        indexing="ij",
    )
    vertices = torch.stack(
        [xx * spacing, yy * spacing, torch.zeros_like(xx)], dim=-1
    ).reshape(-1, 3)
    faces = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = a + cols
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, torch.tensor(faces, dtype=torch.long)


def _unique_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def _strain_tail(
    rest: torch.Tensor, current: torch.Tensor, faces: torch.Tensor, threshold: float
) -> float:
    edges = _unique_edges(faces)
    rest_len = (rest[edges[:, 0]] - rest[edges[:, 1]]).norm(dim=1)
    cur_len = (current[edges[:, 0]] - current[edges[:, 1]]).norm(dim=1)
    ratio = cur_len / rest_len.clamp_min(1e-9)
    return float(((ratio > threshold) | (ratio < 1.0 / threshold)).float().mean())


def test_rigid_motion_is_exact_fixed_point() -> None:
    rest, faces = _grid_mesh()
    constraint = build_face_rigid_shape_constraint(rest, faces)
    angle = math.radians(37.0)
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rigid = rest @ rotation.T + torch.tensor([0.08, -0.03, 0.04])
    projected = project_face_rigid_shape(rigid, constraint)
    assert torch.allclose(projected, rigid, atol=2e-7, rtol=2e-6)


def test_articulated_hinge_is_not_flattened() -> None:
    rest = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.05, 0.05, 0.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    current = rest.clone()
    axis_a, axis_b = rest[1], rest[2]
    axis = (axis_b - axis_a) / (axis_b - axis_a).norm()
    angle = math.radians(65.0)
    k = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        torch.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * axis[:, None] * axis[None, :]
        + math.sin(angle) * k
    )
    current[3] = (rest[3] - axis_a) @ rotation.T + axis_a
    constraint = build_face_rigid_shape_constraint(rest, faces)
    projected = project_face_rigid_shape(current, constraint)
    assert torch.allclose(projected, current, atol=2e-7, rtol=2e-6)


def test_projection_reduces_shape_noise_and_strain() -> None:
    rest, faces = _grid_mesh(rows=12, cols=14)
    constraint = build_face_rigid_shape_constraint(rest, faces)
    current = rest.clone()
    x, y = current[:, 0], current[:, 1]
    current[:, 2] = 0.018 * torch.sin(13.0 * x) * torch.sin(9.0 * y)
    checker = ((torch.arange(len(current)) % 2) * 2 - 1).to(current.dtype)
    current[:, 0] += checker * 0.0045
    current[:, 1] += torch.roll(checker, 3) * 0.0025
    before_tail = _strain_tail(rest, current, faces, threshold=1.5)
    projected = project_face_rigid_shape(current, constraint)
    after_tail = _strain_tail(rest, projected, faces, threshold=1.5)
    assert before_tail > 0.10
    assert after_tail < before_tail * 0.75
    correction = (projected - current).norm(dim=1)
    assert float(correction.quantile(0.95)) < 0.0081


def test_per_vertex_correction_is_hard_bounded() -> None:
    rest, faces = _grid_mesh(rows=5, cols=5)
    constraint = build_face_rigid_shape_constraint(rest, faces)
    distorted = rest.clone()
    distorted[12] += torch.tensor([0.08, -0.04, 0.06])
    projected = project_face_rigid_shape(
        distorted,
        constraint,
        iterations=4,
        strength=1.0,
        max_correction_m=0.006,
    )
    correction = (projected - distorted).norm(dim=1)
    assert float(correction.max()) <= 0.006001


def test_same_lbs_frame_is_deterministic_and_history_free() -> None:
    rest, faces = _grid_mesh(rows=7, cols=8)
    constraint = build_face_rigid_shape_constraint(rest, faces)
    generator = torch.Generator().manual_seed(19260817)
    lbs = rest + torch.randn(rest.shape, generator=generator) * 0.002
    first = project_face_rigid_shape(lbs, constraint)
    second = project_face_rigid_shape(lbs, constraint)
    assert torch.equal(first, second)


def test_mesh_renderer_applies_projection_after_lbs(monkeypatch) -> None:
    """The production renderer must project, not merely expose a helper."""
    from demo_v7.service import gaussian_dynamics
    from demo_v7.service.gaussian_live import MeshAnchoredGaussianRenderer

    rest, faces = _grid_mesh(rows=7, cols=8)
    constraint = build_face_rigid_shape_constraint(rest, faces)
    distorted = rest.clone()
    checker = ((torch.arange(len(rest)) % 2) * 2 - 1).to(rest.dtype)
    distorted[:, 0] += checker * 0.0045
    distorted[:, 1] += torch.roll(checker, 3) * 0.0025
    before_tail = _strain_tail(rest, distorted, faces, threshold=1.5)

    live = object.__new__(MeshAnchoredGaussianRenderer)
    live.device = "cpu"
    live._torch = torch
    live._ctrl_rest = torch.zeros((2, 3))
    live._ctrl_prev = torch.zeros((2, 3))
    live._relations = torch.tensor([[1], [0]])
    live._verts_rest = rest
    live._verts = rest.clone()
    live._faces = faces
    live._skin_weights = torch.ones((len(rest), 1))
    live._skin_indices = torch.zeros((len(rest), 1), dtype=torch.long)
    live._shape_constraint = constraint
    live._shape_correction = torch.zeros(len(rest))
    identity_quats = torch.zeros((len(rest), 4))
    identity_quats[:, 0] = 1.0
    live._tensors = {"means": rest.clone(), "quats": identity_quats.clone()}
    live.bones_moved_m = 0.0
    live.splats_moved_m = 0.0
    live._replay = lambda vertices: (vertices.clone(), identity_quats.clone())

    monkeypatch.setattr(
        gaussian_dynamics,
        "interpolate_motions_sparse",
        lambda *args, **kwargs: (distorted.clone(), None),
    )
    live._pose_to(torch.full((2, 3), 0.01))

    after_tail = _strain_tail(rest, live._verts, faces, threshold=1.5)
    assert after_tail < before_tail * 0.75
    assert float(live._shape_correction.max()) <= 0.008001
    assert torch.equal(live._tensors["means"], live._verts)
