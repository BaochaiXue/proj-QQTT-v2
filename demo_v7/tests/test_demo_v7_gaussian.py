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


class TestFirstGenerateFreshnessGate:
    """The first-generate waiter must ignore a stale masked_image from a
    previous run in the same base_path (mtime gate)."""

    def _manager(self, tmp_path):
        import demo_v7.service.gaussian_manager as gm

        manager = gm.GaussianManager(
            case_dir=tmp_path / "case",
            out_dir=tmp_path / "out",
            controller_name="hand",
            emit_progress=lambda *a, **k: None,
            emit_artifacts=lambda *a, **k: None,
            emit_error=lambda *a, **k: None,
        )
        calls = []
        manager.regenerate = lambda seed: calls.append(seed) or True
        return manager, calls

    def test_stale_image_rejected_fresh_accepted(self, tmp_path) -> None:
        import os
        import threading
        import time as time_mod

        manager, calls = self._manager(tmp_path)
        shape_dir = tmp_path / "case" / "shape"
        shape_dir.mkdir(parents=True)
        stale = shape_dir / "masked_image.png"
        stale.write_bytes(b"old-run-image")
        old = time_mod.time() - 3600.0
        os.utime(stale, (old, old))

        manager._submit_wall = time_mod.time()
        manager._submit_perf = time_mod.perf_counter()
        waiter = threading.Thread(
            target=manager._queue_first_generate, daemon=True
        )
        waiter.start()
        time_mod.sleep(0.8)
        assert calls == [], "stale image must not trigger a generation"

        stale.write_bytes(b"fresh-image-from-this-run")
        waiter.join(timeout=5.0)
        assert not waiter.is_alive()
        assert calls == [manager.seed]


class TestRegisterCanonicalArticulated:
    """Registration must survive articulated pose asymmetry (the sloth
    failure class: a near-flip coarse tie resolved by refining top-K
    candidates instead of only the coarse winner)."""

    def _limbed_body(self) -> np.ndarray:
        # Torso deliberately near-symmetric under 180-deg flips; the thin
        # limbs are the only disambiguators — exactly the geometry where
        # the coarse 24-rotation chamfer ranks a flip within a hair of the
        # truth.
        rng = np.random.default_rng(11)
        torso = rng.normal(size=(3000, 3)) * np.array([0.16, 0.12, 0.07])
        arm_x = np.stack(
            [
                np.linspace(0.14, 0.40, 220),
                rng.normal(size=220) * 0.015,
                rng.normal(size=220) * 0.015 + 0.03,
            ],
            axis=1,
        )
        leg_y = np.stack(
            [
                rng.normal(size=160) * 0.015 - 0.04,
                np.linspace(0.10, 0.30, 160),
                rng.normal(size=160) * 0.015,
            ],
            axis=1,
        )
        return np.concatenate([torso, arm_x, leg_y]).astype(np.float64)

    @pytest.mark.parametrize("angle_deg", [172.5, 180.0, 90.0])
    def test_recovers_near_flip(self, angle_deg: float) -> None:
        pytest.importorskip("open3d")
        from demo_v7.service.gaussian_align import register_canonical

        target = self._limbed_body()
        angle = np.radians(angle_deg)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Source = target seen in a rotated canonical frame (plus jitter,
        # as a gaussian cloud never matches a surface sample exactly).
        rng = np.random.default_rng(5)
        source = target @ rotation.T + rng.normal(size=target.shape) * 0.004
        opacities = np.full(len(source), 0.9, dtype=np.float32)

        transform, chamfer = register_canonical(source, opacities, target)
        assert chamfer < 0.03, f"registration chamfer too high: {chamfer}"
        # The limbs are what near-flips get wrong: the transformed arm tip
        # must land on the target arm tip, not across the body.
        arm_tip = np.array([0.40, 0.0, 0.03]) @ rotation.T
        moved_tip = arm_tip @ transform[:3, :3].T + transform[:3, 3]
        assert np.linalg.norm(moved_tip - np.array([0.40, 0.0, 0.03])) < 0.05


class TestRigidWorldCatchup:
    def _cloud(self) -> np.ndarray:
        rng = np.random.default_rng(23)
        return (rng.normal(size=(3000, 3)) * 0.08 + np.array([0.3, 0.1, 0.05])).astype(
            np.float64
        )

    def test_recovers_small_rigid_motion(self) -> None:
        pytest.importorskip("open3d")
        from demo_v7.service.gaussian_align import rigid_world_catchup

        means = self._cloud()
        angle = np.radians(8.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target = means @ rotation.T + np.array([0.06, -0.04, 0.0])
        opacities = np.full(len(means), 0.9, dtype=np.float32)
        transform, info = rigid_world_catchup(means, opacities, target)
        assert transform is not None, f"catch-up rejected: {info}"
        moved = means @ transform[:3, :3].T + transform[:3, 3]
        assert float(np.abs(moved - target).mean()) < 0.01
        assert info["after_cm"] < info["before_cm"]

    def test_rejects_implausible_jump(self) -> None:
        pytest.importorskip("open3d")
        from demo_v7.service.gaussian_align import rigid_world_catchup

        means = self._cloud()
        target = means + np.array([1.5, 0.0, 0.0])  # a table-length away
        opacities = np.full(len(means), 0.9, dtype=np.float32)
        transform, info = rigid_world_catchup(means, opacities, target)
        assert transform is None
        assert "rejected" in info


class TestGaussianLiveRestSeed:
    """Bone rest positions must come from the seeded seq-0 pose so the
    first packet becomes a catch-up deformation (the stuck-in-old-pose
    trap), substepped for large motions."""

    def _bare_renderer(self, count: int = 200):
        from demo_v7.service.gaussian_live import GaussianLiveRenderer

        live = object.__new__(GaussianLiveRenderer)
        live.device = "cpu"
        live.failed = False
        live._torch = torch
        gen = torch.Generator().manual_seed(9)
        means = torch.rand(count, 3, generator=gen)
        quats = torch.zeros(count, 4)
        quats[:, 0] = 1.0
        live._tensors = {
            "means": means.clone(),
            "quats": quats.clone(),
            "scales": torch.full((count, 3), 0.01),
            "opacities": torch.full((count,), 0.9),
            "colors": torch.rand(count, 3, generator=gen),
        }
        live._bone_ids = None
        live._relations = None
        live._ctrl_prev = None
        live._buffer = {}
        live._rest_positions = {}
        live.rest_seeded = False
        live._seed_grace_left = 25
        live.frames_stepped = 0
        live.last_substeps = 0
        live.bones_moved_m = 0.0
        live.splats_moved_m = 0.0
        return live, means

    def _grid(self) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, 3)
        return (
            np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .astype(np.float32)
        )

    def test_seeded_first_packet_catches_up(self) -> None:
        live, means = self._bare_renderer()
        rest = self._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        live.seed_rest_positions({int(i): rest[i] for i in ids})
        shift = np.array([0.09, 0.0, 0.0], dtype=np.float32)
        live.step(rest + shift, ids, np.ones(len(ids), dtype=bool))
        assert live.rest_seeded
        # ceil(0.09 / 0.02) = 5 substeps for the catch-up
        assert live.last_substeps == 5
        assert torch.allclose(
            live._tensors["means"], means + torch.as_tensor(shift), atol=1e-3
        )
        assert live.bones_moved_m > 0.0
        stats = live.follow_stats()
        assert stats is not None and stats["rest_seeded"] is True

    def test_unseeded_keeps_first_packet_rest(self) -> None:
        live, means = self._bare_renderer()
        rest = self._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        live.step(rest, ids, np.ones(len(ids), dtype=bool))
        assert not live.rest_seeded
        assert torch.allclose(live._tensors["means"], means)  # no motion yet
        shift = np.array([0.03, 0.0, 0.0], dtype=np.float32)
        live.step(rest + shift, ids, np.ones(len(ids), dtype=bool))
        assert torch.allclose(
            live._tensors["means"], means + torch.as_tensor(shift), atol=1e-3
        )

    def test_partial_seed_waits_then_falls_back(self) -> None:
        live, means = self._bare_renderer()
        rest = self._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        live.seed_rest_positions({0: rest[0], 1: rest[1]})  # < _MIN_BONES
        # Within the grace window a marginal packet must NOT freeze an
        # unseeded bone set — later packets may complete the intersection.
        live.step(rest, ids, np.ones(len(ids), dtype=bool))
        assert live._bone_ids is None and not live.rest_seeded
        live._seed_grace_left = 0
        live.step(rest, ids, np.ones(len(ids), dtype=bool))
        assert not live.rest_seeded
        assert live._bone_ids is not None and len(live._bone_ids) == len(ids)

    def test_grace_window_lets_late_packets_seed(self) -> None:
        live, means = self._bare_renderer()
        rest = self._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        live.seed_rest_positions({int(i): rest[i] for i in ids})
        # First packet only carries a few object markers (occlusion).
        live.step(rest[:4], ids[:4], np.ones(4, dtype=bool))
        assert live._bone_ids is None
        # A later, fuller packet completes the seedable intersection.
        live.step(rest, ids, np.ones(len(ids), dtype=bool))
        assert live.rest_seeded

    def test_apply_rigid_transform_rotates_quats(self) -> None:
        live, means = self._bare_renderer()
        angle = np.radians(90.0)
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform[:3, 3] = [0.1, -0.2, 0.05]
        live.apply_rigid_transform(transform)
        rotation = torch.as_tensor(transform[:3, :3], dtype=torch.float32)
        expected = means @ rotation.T + torch.as_tensor(
            transform[:3, 3], dtype=torch.float32
        )
        assert torch.allclose(live._tensors["means"], expected, atol=1e-5)
        back = gaussian_dynamics.quat2mat(live._tensors["quats"][:1])
        assert torch.allclose(back[0], rotation, atol=1e-5)


class TestFormalFrame0Loader:
    def test_loads_visible_object_queries_only(self, tmp_path) -> None:
        from demo_v7.service.gaussian_live import (
            load_formal_frame0_rest_positions,
        )

        height, width = 6, 8
        yy, xx = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        pcd = np.stack(
            [xx * 0.1, yy * 0.1, np.full_like(xx, 2.0, dtype=float)], axis=-1
        ).astype(np.float32)[None]
        mask_object = np.zeros((height, width), dtype=bool)
        mask_object[2:5, 2:6] = True
        tracks = np.array(
            [
                [3.0, 4.0],  # visible, on object -> bone
                [3.2, 4.8],  # visible, rounds to (3,5) on object -> bone
                [0.0, 0.0],  # visible but off-object -> dropped
                [3.0, 3.0],  # invisible -> dropped
            ],
            dtype=np.float32,
        )
        visibility = np.array([True, True, True, False])
        npz = tmp_path / "000000.npz"
        np.savez(
            npz,
            seq=np.array([0]),
            tracks_yx=tracks,
            visibility=visibility,
            pcd_points=pcd,
            mask_object=mask_object,
        )
        rest, cloud = load_formal_frame0_rest_positions(npz)
        assert sorted(rest) == [0, 1]
        assert np.allclose(rest[0], [0.4, 0.3, 2.0], atol=1e-6)
        assert np.allclose(rest[1], [0.5, 0.3, 2.0], atol=1e-6)
        assert len(cloud) == int(mask_object.sum())


class TestWhitenBackground:
    def test_amounts(self) -> None:
        from demo_v7.service.gaussian_live import whiten_background

        frame = np.full((2, 2, 3), 100, dtype=np.uint8)
        assert np.allclose(whiten_background(frame, 0.0), 100.0)
        assert np.allclose(whiten_background(frame, 1.0), 255.0)
        assert np.allclose(whiten_background(frame, 0.5), 177.5)
        # Out-of-range amounts clamp instead of exploding.
        assert np.allclose(whiten_background(frame, 2.0), 255.0)
        assert np.allclose(whiten_background(frame, -1.0), 100.0)


class TestQuaternionHemisphereBlend:
    @staticmethod
    def _rot_z(angle_deg: float) -> torch.Tensor:
        angle = np.radians(angle_deg)
        return torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def test_antipodal_neighbor_quats_do_not_cancel(self) -> None:
        """q and -q encode the same rotation; blending must not cancel.

        Two bone clusters rotate +179.9 and -179.9 deg about z — a mere
        0.2 deg apart as ROTATIONS, but mat2quat emits near-antipodal
        quats ([~0,0,0,+1] vs [~0,0,0,-1]). A raw weighted sum collapses
        toward identity (the pre-fix behavior); hemisphere alignment must
        keep the blend a ~180-deg z rotation."""
        offsets = torch.tensor(
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [-0.1, -0.1, 0.0]]
        )
        center_a = torch.tensor([0.0, 0.0, 0.0])
        center_b = torch.tensor([1.0, 0.0, 0.0])
        bones = torch.cat([center_a + offsets, center_b + offsets])
        rot_a, rot_b = self._rot_z(179.9), self._rot_z(-179.9)
        motions = torch.cat(
            [
                (offsets @ rot_a.T + center_a) - bones[:3],
                (offsets @ rot_b.T + center_b) - bones[3:],
            ]
        )
        relations = torch.tensor(
            [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
        )
        particles = torch.tensor([[0.5, 0.0, 0.0]])
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        weights, indices = gaussian_dynamics.knn_weights_sparse(
            bones, particles, K=6
        )
        _new_xyz, new_quat = gaussian_dynamics.interpolate_motions_sparse(
            bones, motions, relations, particles, quats, weights, indices,
            device="cpu",
        )
        blended = new_quat / torch.linalg.norm(new_quat, dim=1, keepdim=True)
        # ~180 deg about z: |z| ~ 1, w ~ 0. The pre-fix raw sum measured
        # w=+0.199 / z=+0.980 here (partial cancellation normalized into a
        # ~23-deg w error); the aligned blend gives w=0.000 / z=1.000.
        assert abs(float(blended[0, 3])) > 0.999
        assert abs(float(blended[0, 0])) < 0.05
