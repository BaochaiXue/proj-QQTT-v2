"""Unit tests for the gaussian-splats feature (CPU only; no GPU, no models).

Covers the vendored dynamics math (quaternion round trips incl. the
trace<=-1 branch upstream got wrong, rigid-motion transfer), the similarity
transform on splats, ply IO round trip, and the canonical-registration
helpers' invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from demo_v7.service import gaussian_dynamics  # noqa: E402
from demo_v7.service.gaussian_utils import (  # noqa: E402
    GaussianSplats,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_gaussians,
)


def _random_rotations(count: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return gaussian_dynamics.quat2mat(torch.as_tensor(quats, dtype=torch.float64))


class TestQuaternionMath:
    def test_roundtrip_random(self) -> None:
        mats = _random_rotations(256)
        back = gaussian_dynamics.quat2mat(gaussian_dynamics.mat2quat(mats))
        assert torch.allclose(back, mats, atol=1e-6)

    def test_roundtrip_near_pi_rotations(self) -> None:
        # trace <= -1 exercises the mask_1/2/3 branches (upstream's dead-code
        # bug lived here); 180-degree rotations about each principal axis.
        mats = []
        for axis in np.eye(3):
            mats.append(
                2.0 * np.outer(axis, axis) - np.eye(3)
            )  # rotation by pi about `axis`
        mats_t = torch.as_tensor(np.stack(mats), dtype=torch.float64)
        back = gaussian_dynamics.quat2mat(gaussian_dynamics.mat2quat(mats_t))
        assert torch.allclose(back, mats_t, atol=1e-6)


class TestMotionTransfer:
    def _grid_bones(self) -> torch.Tensor:
        xs = torch.linspace(0, 1, 4)
        return torch.stack(
            torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1
        ).reshape(-1, 3)

    def test_pure_translation_transfers_exactly(self) -> None:
        bones = self._grid_bones()
        motion = torch.tensor([0.05, -0.02, 0.11])
        motions = motion[None].repeat(len(bones), 1)
        relations = gaussian_dynamics.get_topk_indices(bones, K=6)
        particles = torch.rand(500, 3)
        quats = torch.zeros(500, 4)
        quats[:, 0] = 1.0
        weights, indices = gaussian_dynamics.knn_weights_sparse(
            bones, particles, K=8
        )
        new_xyz, new_quat = gaussian_dynamics.interpolate_motions_sparse(
            bones, motions, relations, particles, quats, weights, indices,
            device="cpu",
        )
        assert torch.allclose(new_xyz, particles + motion, atol=1e-4)
        # A pure translation must not rotate the splats.
        assert torch.allclose(new_quat, quats, atol=1e-4)

    def test_knn_weights_sum_to_one(self) -> None:
        bones = self._grid_bones()
        particles = torch.rand(100, 3)
        weights, indices = gaussian_dynamics.knn_weights_sparse(bones, particles, K=5)
        assert weights.shape == (100, 5)
        assert indices.shape == (100, 5)
        assert torch.allclose(weights.sum(dim=1), torch.ones(100), atol=1e-5)


class TestSplatTransforms:
    def _splats(self, count: int = 64) -> GaussianSplats:
        rng = np.random.default_rng(3)
        quats = rng.normal(size=(count, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        return GaussianSplats(
            means=rng.normal(size=(count, 3)).astype(np.float32),
            quats=quats,
            scales=rng.uniform(0.001, 0.01, size=(count, 3)).astype(np.float32),
            opacities=rng.uniform(0.1, 0.99, size=count).astype(np.float32),
            colors=rng.uniform(0, 1, size=(count, 3)).astype(np.float32),
        )

    def test_similarity_transform_means_and_scales(self) -> None:
        splats = self._splats()
        angle = np.radians(30)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation * 0.5
        transform[:3, 3] = [1.0, -2.0, 3.0]
        moved = transform_gaussians(splats, transform)
        expected = splats.means @ (rotation * 0.5).T + [1.0, -2.0, 3.0]
        assert np.allclose(moved.means, expected, atol=1e-5)
        assert np.allclose(moved.scales, splats.scales * 0.5, atol=1e-7)
        norms = np.linalg.norm(moved.quats, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_non_uniform_linear_part_rejected(self) -> None:
        splats = self._splats(8)
        transform = np.diag([1.0, 2.0, 3.0, 1.0])
        with pytest.raises(ValueError, match="similarity"):
            transform_gaussians(splats, transform)

    def test_ply_roundtrip(self, tmp_path) -> None:
        splats = self._splats()
        path = tmp_path / "roundtrip.ply"
        save_gaussian_ply(path, splats)
        loaded = load_gaussian_ply(path)
        assert np.allclose(loaded.means, splats.means, atol=1e-6)
        assert np.allclose(loaded.scales, splats.scales, rtol=1e-4)
        assert np.allclose(loaded.opacities, splats.opacities, atol=1e-4)
        assert np.allclose(loaded.colors, splats.colors, atol=1e-5)
        dots = np.abs(np.sum(loaded.quats * splats.quats, axis=1))
        assert np.allclose(dots, 1.0, atol=1e-5)


class TestRegistrationHelpers:
    def test_axis_rotations_are_24_proper(self) -> None:
        from demo_v7.service.gaussian_align import _axis_rotations

        rotations = _axis_rotations()
        assert len(rotations) == 24
        for rotation in rotations:
            assert np.isclose(np.linalg.det(rotation), 1.0)
        unique = {tuple(np.round(r.flatten()).astype(int)) for r in rotations}
        assert len(unique) == 24


class TestGaussianBackendSelector:
    """GUI selector vocabulary + session-level forcing rules."""

    def test_normalize_defaults_and_ids(self) -> None:
        from demo_v7.service import gaussian_options as go

        assert go.normalize_gaussian_backend(None) == go.GAUSSIAN_TRIPOSPLAT
        assert go.normalize_gaussian_backend("") == go.GAUSSIAN_TRIPOSPLAT
        assert go.normalize_gaussian_backend(" TripoSplat ") == "triposplat"
        assert go.normalize_gaussian_backend("none") == go.GAUSSIAN_NONE
        with pytest.raises(ValueError, match="unknown gaussian backend"):
            go.normalize_gaussian_backend("splatco")

    def _session(self, tmp_path, **kwargs):
        from demo_v7.orchestration.session import OrchestratorSession

        return OrchestratorSession(
            source="fake-live",
            fake_live_case="data_collect/fake",
            base_path=tmp_path / "run",
            **kwargs,
        )

    def test_session_default_is_triposplat(self, tmp_path) -> None:
        session = self._session(tmp_path)
        assert session.gaussian_backend == "triposplat"

    def test_shape_prior_none_forces_gaussian_none(self, tmp_path) -> None:
        session = self._session(
            tmp_path, shape_prior_backend="none", gaussian_backend="triposplat"
        )
        assert session.gaussian_backend == "none"

    def test_explicit_gaussian_none_sticks(self, tmp_path) -> None:
        session = self._session(tmp_path, gaussian_backend="none")
        assert session.gaussian_backend == "none"

    def test_camera_service_parser_consumes_flag(self) -> None:
        from demo_v7.service.camera_service import _build_v7_parser

        v7_args, rest = _build_v7_parser().parse_known_args(
            [
                "--socket-dir",
                "/tmp/x",
                "--gaussian-backend",
                "none",
                "--input-source",
                "fake-live",
            ]
        )
        assert v7_args.gaussian_backend == "none"
        assert "--gaussian-backend" not in rest
