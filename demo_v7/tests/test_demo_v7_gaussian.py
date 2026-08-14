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
    first packet becomes a one-shot catch-up deformation (the
    stuck-in-old-pose trap)."""

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
        live._ctrl_rest = None
        live._ctrl_prev = None
        live._rest_means = None
        live._rest_quats = None
        live._skin_weights = None
        live._skin_indices = None
        live._buffer = {}
        live._rest_positions = {}
        live.rest_seeded = False
        live._seed_grace_left = 25
        live._last_seen_step = None
        live.frames_stepped = 0
        live.bones_moved_m = 0.0
        live.splats_moved_m = 0.0
        live.bone_outliers = 0
        live.bone_stale = 0
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


class TestBoneHygiene:
    """Rogue tracks and occlusion-stale bones must ride their neighbors'
    consensus instead of dragging (or anchoring) their bound splats."""

    def _seeded(self, helper: "TestGaussianLiveRestSeed"):
        live, means = helper._bare_renderer()
        rest = helper._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        live.seed_rest_positions({int(i): rest[i] for i in ids})
        live.step(rest, ids, np.ones(len(ids), dtype=bool))  # init at rest
        return live, means, rest, ids

    def test_rogue_bone_is_overridden_by_neighbors(self) -> None:
        helper = TestGaussianLiveRestSeed()
        live, means, rest, ids = self._seeded(helper)
        shift = np.array([0.03, 0.0, 0.0], dtype=np.float32)
        target = rest + shift
        target[0] = rest[0] + np.array([0.0, 0.0, 0.5], dtype=np.float32)
        live.step(target, ids, np.ones(len(ids), dtype=bool))
        assert live.bone_outliers >= 1
        # The rogue bone's 50cm z-jump must NOT reach the splats: consensus
        # replaces it with the neighborhood's +3cm x translation.
        assert torch.allclose(
            live._tensors["means"], means + torch.as_tensor(shift), atol=2e-3
        )

    def test_stale_bones_ride_the_visible_half(self) -> None:
        helper = TestGaussianLiveRestSeed()
        live, means, rest, ids = self._seeded(helper)
        seen = ids[ids % 2 == 0]
        step = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        # 14 packets (> _BONE_STALE_STEPS) where only even bones update.
        for k in range(1, 15):
            live.step(
                rest[seen] + step * k, seen, np.ones(len(seen), dtype=bool)
            )
        assert live.bone_stale > 0
        # Stale odd bones must not anchor the object at the rest pose: the
        # whole cloud rides the visible bones' translation.
        assert torch.allclose(
            live._tensors["means"],
            means + torch.as_tensor(step * 14),
            atol=2e-3,
        )

    def test_stale_bones_follow_rotation_not_average(self) -> None:
        """The drag fix: an occluded patch must ride the visible bones'
        ROTATION. A translation-average heal puts stale bones at the mean
        neighbor displacement (wrong under rotation); rigid-aware healing
        recovers their rotated positions almost exactly.

        Geometry is at PRODUCTION scale (12cm object, 3cm bone spacing —
        sloth-like): at demo scale the 5cm rigidity threshold keeps fresh
        bones trusted; a metre-scale grid would flag everyone and disable
        healing entirely (fail-soft)."""
        helper = TestGaussianLiveRestSeed()
        live, means = helper._bare_renderer()
        xs = np.linspace(0.0, 0.12, 5, dtype=np.float32)
        rest = (
            np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1)
            .reshape(-1, 3)
        )
        ids = np.arange(len(rest), dtype=np.int64)
        live.seed_rest_positions({int(i): rest[i] for i in ids})
        live.step(rest, ids, np.ones(len(ids), dtype=bool))  # init at rest
        angle = np.radians(25.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = rest.mean(axis=0)
        rotated = (rest - center) @ rotation.T + center
        seen = ids[ids % 2 == 0]
        # 14 packets (> stale threshold) where only even bones update, at
        # the ROTATED pose.
        for _ in range(14):
            live.step(rotated[seen], seen, np.ones(len(seen), dtype=bool))
        applied = live._ctrl_prev.cpu().numpy()
        stale_ids = ids[ids % 2 == 1]
        error = np.linalg.norm(applied[stale_ids] - rotated[stale_ids], axis=1)
        # A global rigid motion is EXACT under the per-bone Kabsch blend;
        # the old mean-consensus heal leaves ~1cm errors at this scale.
        assert float(error.max()) < 0.005, error.max()


class TestFloaterPruning:
    def test_disconnected_island_pruned_even_near_mesh(self) -> None:
        """The measured failure: a small solid island 10cm from every other
        splat but within mesh distance (the mesh arm tip passed nearby) —
        connectivity must catch what mesh distance cannot."""
        from demo_v7.service.gaussian_align import _floater_keep_mask

        rng = np.random.default_rng(4)
        # Uniform cube: ~5mm spacing keeps the body one component at the
        # 8mm link radius (a gaussian ball's sparse tail would fragment).
        blob = rng.uniform(-0.05, 0.05, size=(8000, 3))
        island = rng.uniform(-0.004, 0.004, size=(60, 3)) + np.array(
            [0.30, 0.0, 0.0]
        )
        fuzz_on_blob = rng.uniform(-0.05, 0.05, size=(40, 3))
        fuzz_on_island = rng.uniform(-0.004, 0.004, size=(20, 3)) + np.array(
            [0.30, 0.0, 0.0]
        )
        means = np.concatenate([blob, island, fuzz_on_blob, fuzz_on_island])
        opacities = np.concatenate(
            [
                np.full(8000, 0.9),
                np.full(60, 0.9),
                np.full(40, 0.1),
                np.full(20, 0.1),
            ]
        ).astype(np.float32)
        count = len(means)
        world = GaussianSplats(
            means=means.astype(np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1)),
            scales=np.full((count, 3), 0.005, np.float32),
            opacities=opacities,
            colors=np.full((count, 3), 0.5, np.float32),
        )
        # Mesh passes through the blob AND right next to the island, so the
        # mesh-distance criterion alone keeps everything.
        mesh = np.concatenate(
            [blob, island + np.array([0.01, 0.0, 0.0])]
        )
        keep = _floater_keep_mask(world, mesh)
        assert keep[:8000].all(), "main body must be kept"
        assert not keep[8000:8060].any(), "solid island must be pruned"
        assert keep[8060:8100].all(), "fuzz on the body must be kept"
        assert not keep[8100:].any(), "island fuzz must follow its island"


class TestSelfAlignHelpers:
    def test_pure_articulation_strips_similarity(self, tmp_path, monkeypatch) -> None:
        """A purely-similar ARAP field (rigid+scale, no articulation) must
        strip to ~zero displacement — transplanting it raw onto an
        independently-registered gaussian double-corrects (benchmarked)."""
        from demo_v7.service import gaussian_selfalign as sa

        rng = np.random.default_rng(6)
        canonical = rng.normal(size=(500, 3))
        mesh2world = np.eye(4)
        angle = np.radians(20)
        similarity = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ) * 1.1
        target = canonical @ similarity.T + np.array([0.05, -0.02, 0.01])
        monkeypatch.setattr(
            "demo_v7.service.gaussian_align.arap_residual_field",
            lambda case_dir, m2w: (canonical, target - canonical),
        )
        anchors, articulation, final = sa.pure_articulation_field(
            tmp_path, mesh2world
        )
        assert float(np.abs(articulation).max()) < 1e-6
        assert np.allclose(anchors, final, atol=1e-6)

    def test_combined_score_and_gates(self) -> None:
        from demo_v7.service import gaussian_selfalign as sa

        good = {"iou": 0.90, "c2g_p90_cm": 1.0}
        flat = {"iou": 0.92, "c2g_p90_cm": 2.0}
        # 2 IoU points do not pay for 1cm of coverage tail at the 3pt/cm rate.
        assert sa.combined_score(good) > sa.combined_score(flat)

    def test_subprocess_error_json_is_fail_soft(self, tmp_path) -> None:
        """A child that writes an error payload must yield None, not raise."""
        import json as json_mod

        from demo_v7.service import gaussian_selfalign as sa

        work = tmp_path / "work"
        work.mkdir()
        (work / "self_align_result.json").write_text(
            json_mod.dumps({"error": "boom"})
        )
        # Hand-patch subprocess.run so the child never spawns (the result
        # json is already on disk for the parse path under test).
        import subprocess as sp

        real_run = sp.run
        try:
            sp.run = lambda *a, **k: sp.CompletedProcess(a, 0, "", "")
            assert sa.run_self_align_subprocess(
                tmp_path, tmp_path / "raw.ply", work
            ) is None
        finally:
            sp.run = real_run


class TestAsapIslandCleanup:
    def test_patched_loader_drops_tiny_components(self, tmp_path, monkeypatch) -> None:
        o3d = pytest.importorskip("open3d")
        from demo_v7.service import arap_rescue
        from demo_v7.runtime.streaming import asap

        # Body: a sphere big enough that the one-triangle island stays
        # under the 1% component-fraction gate.
        island_v = np.array([[5.0, 5, 5], [5.1, 5, 5], [5, 5.1, 5]])
        body2 = o3d.geometry.TriangleMesh.create_sphere(resolution=10)
        base = np.asarray(body2.vertices).shape[0]
        verts = np.concatenate([np.asarray(body2.vertices), island_v])
        tris = np.concatenate(
            [np.asarray(body2.triangles), [[base, base + 1, base + 2]]]
        ).astype(np.int32)
        combined = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
        )

        stock = lambda path: o3d.geometry.TriangleMesh(combined)
        monkeypatch.setattr(asap, "_load_clean_mesh", stock, raising=True)
        arap_rescue.patch_asap_island_cleanup()
        try:
            cleaned = asap._load_clean_mesh(tmp_path / "x.glb")
            _labels, counts, _ = cleaned.cluster_connected_triangles()
            assert len(np.asarray(counts)) == 1
            assert np.asarray(cleaned.triangles).shape[0] == np.asarray(
                body2.triangles
            ).shape[0]
        finally:
            # Un-patch so other tests see the real loader.
            monkeypatch.undo()


class TestSelfAlignDefaultPolicy:
    """Owner decision: self-align is demo 7's DEFAULT alignment — B wins
    ties, and the chamfer incumbent survives only a clear loss."""

    def test_b_is_default_among_candidates(self) -> None:
        from demo_v7.service import gaussian_selfalign as sa

        scored = [
            ("self_align", {"iou": 0.90, "c2g_p90_cm": 1.5}),
            ("self_align_art", {"iou": 0.905, "c2g_p90_cm": 1.5}),  # +0.005
        ]
        assert sa.pick_candidate(scored)[0] == "self_align"
        scored[1] = ("self_align_art", {"iou": 0.95, "c2g_p90_cm": 1.5})
        assert sa.pick_candidate(scored)[0] == "self_align_art"

    def test_swap_default_unless_clear_loss(self) -> None:
        from demo_v7.service import gaussian_selfalign as sa

        incumbent = {"iou": 0.70, "c2g_p90_cm": 2.0}
        tie = {"iou": 0.695, "c2g_p90_cm": 2.0}  # -0.005: within tolerance
        assert sa.should_swap(tie, incumbent)
        clear_loss = {"iou": 0.60, "c2g_p90_cm": 5.0}  # drive21-gen1 class
        assert not sa.should_swap(clear_loss, incumbent)
        win = {"iou": 0.92, "c2g_p90_cm": 3.3}
        assert sa.should_swap(win, incumbent)


def _synthetic_glb(tmp_path, name: str = "synth.glb"):
    """A small closed textured-free mesh (icosphere) exported as GLB."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    mesh.visual.vertex_colors = np.tile(
        np.array([180, 120, 60, 255], dtype=np.uint8), (len(mesh.vertices), 1)
    )
    path = tmp_path / name
    mesh.export(str(path))
    return path


class TestMeshSurfaceGaussianizer:
    """mesh_surface backend core: splat centers ON the mesh, deterministic,
    anchors self-verifying."""

    def test_centers_on_surface_and_bary_valid(self, tmp_path) -> None:
        from demo_v7.service.mesh_surface_gaussian import (
            gaussianize_mesh,
            replay_splat_means,
        )

        splats, anchors = gaussianize_mesh(
            _synthetic_glb(tmp_path), target_splats=2000, seed=1
        )
        assert len(splats) >= 2000  # >=1 per face can overshoot slightly
        bary = anchors.barycentric
        assert (bary >= -1e-6).all()
        assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-5)
        replayed = replay_splat_means(
            anchors.rest_vertices.astype(np.float64),
            anchors.faces.astype(np.int64),
            anchors.face_index,
            anchors.barycentric.astype(np.float64),
        )
        err = np.linalg.norm(replayed - splats.means, axis=1)
        assert float(err.max()) < 1e-6  # centers ARE the barycentric replay

    def test_every_face_sampled_and_deterministic(self, tmp_path) -> None:
        from demo_v7.service.mesh_surface_gaussian import gaussianize_mesh

        path = _synthetic_glb(tmp_path)
        splats_a, anchors_a = gaussianize_mesh(path, target_splats=1000, seed=5)
        assert len(np.unique(anchors_a.face_index)) == len(anchors_a.faces)
        splats_b, anchors_b = gaussianize_mesh(path, target_splats=1000, seed=5)
        assert np.array_equal(splats_a.means, splats_b.means)
        assert np.array_equal(anchors_a.barycentric, anchors_b.barycentric)
        splats_c, _ = gaussianize_mesh(path, target_splats=1000, seed=6)
        assert not np.array_equal(splats_a.means, splats_c.means)

    def test_anchors_roundtrip_and_hash_guard(self, tmp_path) -> None:
        from demo_v7.service.mesh_surface_gaussian import (
            gaussianize_mesh,
            load_anchors,
            save_anchors,
        )

        _splats, anchors = gaussianize_mesh(
            _synthetic_glb(tmp_path), target_splats=500, seed=2
        )
        path = tmp_path / "anchors.npz"
        save_anchors(path, anchors)
        loaded = load_anchors(path)
        assert loaded.topology_sha256 == anchors.topology_sha256
        assert np.array_equal(loaded.face_index, anchors.face_index)
        # Tampered rest topology must fail loudly (never silently drift).
        data = dict(np.load(path))
        data["rest_vertices"] = data["rest_vertices"] + 0.01
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError, match="hash mismatch"):
            load_anchors(path)

    def test_zero_area_faces_never_sampled(self) -> None:
        from demo_v7.service.mesh_surface_gaussian import _allocate_samples

        areas = np.array([1.0e-4, 0.0, 2.0e-4, 1.0e-20])
        counts = _allocate_samples(areas, 300)
        assert counts[1] == 0 and counts[3] == 0
        assert counts[0] >= 1 and counts[2] >= 1
        assert counts.sum() >= 300

    def test_face_frames_orthonormal_right_handed(self) -> None:
        from demo_v7.service.mesh_surface_gaussian import face_frames

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],  # degenerate duplicate corner
            ]
        )
        faces = np.array([[0, 1, 2], [0, 3, 1]])  # second face is a sliver
        frames = face_frames(vertices, faces)
        good = frames[0]
        assert np.allclose(good.T @ good, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(good), 1.0)
        assert np.allclose(good[:, 2], [0, 0, 1])  # normal of the xy triangle
        assert np.allclose(frames[1], np.eye(3))  # degenerate -> identity

    def test_rigid_replay_consistency(self, tmp_path) -> None:
        """Rotating the vertices rotates the replayed splats identically."""
        from demo_v7.service.mesh_surface_gaussian import (
            face_frames,
            gaussianize_mesh,
            replay_splat_means,
        )

        splats, anchors = gaussianize_mesh(
            _synthetic_glb(tmp_path), target_splats=800, seed=3
        )
        angle = np.radians(40.0)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        translation = np.array([0.02, -0.05, 0.01])
        verts = anchors.rest_vertices.astype(np.float64)
        faces = anchors.faces.astype(np.int64)
        moved = verts @ rotation.T + translation
        replayed = replay_splat_means(
            moved, faces, anchors.face_index, anchors.barycentric.astype(np.float64)
        )
        expected = (
            replay_splat_means(
                verts, faces, anchors.face_index, anchors.barycentric.astype(np.float64)
            )
            @ rotation.T
            + translation
        )
        assert np.allclose(replayed, expected, atol=1e-9)
        # Frames rotate as R @ frame (orientation rides the mesh).
        frames_rest = face_frames(verts, faces)
        frames_moved = face_frames(moved, faces)
        assert np.allclose(frames_moved, rotation @ frames_rest, atol=1e-9)


class TestMeshAnchoredRenderer:
    """Live mesh-anchored deformation: bones deform the vertices, splats
    stay ON the deformed mesh (CPU, bare construction like the parent's
    tests)."""

    def _bare(self, tmp_path):
        from demo_v7.service.gaussian_live import MeshAnchoredGaussianRenderer
        from demo_v7.service.mesh_surface_gaussian import gaussianize_mesh

        splats, anchors = gaussianize_mesh(
            _synthetic_glb(tmp_path), target_splats=600, seed=4
        )
        live = object.__new__(MeshAnchoredGaussianRenderer)
        live.device = "cpu"
        live.failed = False
        live._torch = torch
        live._tensors = {
            "means": torch.as_tensor(splats.means),
            "quats": torch.as_tensor(splats.quats),
            "scales": torch.as_tensor(splats.scales),
            "opacities": torch.as_tensor(splats.opacities),
            "colors": torch.as_tensor(splats.colors),
        }
        live._bone_ids = None
        live._relations = None
        live._ctrl_rest = None
        live._ctrl_prev = None
        live._rest_means = None
        live._rest_quats = None
        live._skin_weights = None
        live._skin_indices = None
        live._buffer = {}
        live._rest_positions = {}
        live.rest_seeded = False
        live._seed_grace_left = 25
        live._last_seen_step = None
        live.frames_stepped = 0
        live.bones_moved_m = 0.0
        live.splats_moved_m = 0.0
        live.bone_outliers = 0
        live.bone_stale = 0
        live._verts = torch.as_tensor(anchors.rest_vertices)
        live._faces = torch.as_tensor(anchors.faces.astype(np.int64))
        live._anchor_face = torch.as_tensor(anchors.face_index.astype(np.int64))
        live._anchor_bary = torch.as_tensor(anchors.barycentric)
        live._face_quats_prev = None
        means, quats = live._replay(live._verts)
        live._tensors["means"] = means
        live._tensors["quats"] = quats
        return live, anchors

    def test_rigid_transform_moves_verts_and_splats_together(
        self, tmp_path
    ) -> None:
        live, _anchors = self._bare(tmp_path)
        before_means = live._tensors["means"].clone()
        angle = np.radians(35.0)
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform[:3, 3] = [0.05, -0.02, 0.03]
        live.apply_rigid_transform(transform)
        rotation = torch.as_tensor(transform[:3, :3], dtype=torch.float32)
        offset = torch.as_tensor(transform[:3, 3], dtype=torch.float32)
        assert torch.allclose(
            live._tensors["means"], before_means @ rotation.T + offset, atol=1e-5
        )
        # Splats still ON the transformed mesh (binding invariant).
        replayed, _ = live._replay(live._verts)
        assert torch.allclose(live._tensors["means"], replayed, atol=1e-6)

    def test_bones_rigid_motion_carries_mesh_and_splats(self, tmp_path) -> None:
        live, _anchors = self._bare(tmp_path)
        verts = live._verts.numpy()
        rng = np.random.default_rng(11)
        bone_rows = rng.choice(len(verts), size=60, replace=False)
        rest_bones = verts[bone_rows].astype(np.float32)
        ids = np.arange(len(rest_bones), dtype=np.int64)
        live.seed_rest_positions({int(i): rest_bones[i] for i in ids})
        live.step(rest_bones, ids, np.ones(len(ids), dtype=bool))
        assert live.rest_seeded
        angle = np.radians(30.0)
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        translation = np.array([0.03, 0.01, -0.02])
        moved = (rest_bones @ rotation.T + translation).astype(np.float32)
        before = live._tensors["means"].numpy().copy()
        live.step(moved, ids, np.ones(len(ids), dtype=bool))
        got = live._tensors["means"].numpy()
        expected = before @ rotation.T + translation
        err = np.linalg.norm(got - expected, axis=1)
        assert float(np.quantile(err, 0.99)) < 2e-3  # rigid follow, mm-exact
        # The binding invariant survives deformation.
        replayed, _ = live._replay(live._verts)
        assert torch.allclose(
            live._tensors["means"], replayed, atol=1e-6
        )

    def test_degenerate_face_keeps_last_orientation(self, tmp_path) -> None:
        live, _anchors = self._bare(tmp_path)
        quats_before = live._tensors["quats"].clone()
        collapsed = live._verts.clone()
        face0 = live._faces[0]
        collapsed[face0[1]] = collapsed[face0[0]]  # kill face 0's first edge
        collapsed[face0[2]] = collapsed[face0[0]]
        _means, quats = live._replay(collapsed)
        assert torch.isfinite(quats).all()
        affected = (live._anchor_face == 0).numpy()
        assert np.allclose(
            quats.numpy()[affected], quats_before.numpy()[affected], atol=1e-6
        )


class TestMeshSurfaceSelector:
    """mesh_surface vocabulary + the trellis2-only rule at every layer."""

    def test_normalize_accepts_mesh_surface(self) -> None:
        from demo_v7.service import gaussian_options as go

        assert go.normalize_gaussian_backend("mesh_surface") == "mesh_surface"
        assert go.GAUSSIAN_MESH_SURFACE in go.GAUSSIAN_BACKENDS

    def test_allowed_rule(self) -> None:
        from demo_v7.service import gaussian_options as go

        assert go.mesh_surface_allowed("trellis2")
        assert not go.mesh_surface_allowed("sam3d")
        assert not go.mesh_surface_allowed("none")
        assert not go.mesh_surface_allowed(None)

    def test_session_rejects_mesh_surface_without_trellis2(self, tmp_path) -> None:
        from demo_v7.orchestration.session import OrchestratorSession

        with pytest.raises(ValueError, match="mesh_surface"):
            OrchestratorSession(
                source="fake-live",
                fake_live_case="data_collect/fake",
                base_path=tmp_path / "run",
                shape_prior_backend="sam3d",
                gaussian_backend="mesh_surface",
            )

    def test_session_accepts_mesh_surface_with_trellis2(self, tmp_path) -> None:
        from demo_v7.orchestration.session import OrchestratorSession

        session = OrchestratorSession(
            source="fake-live",
            fake_live_case="data_collect/fake",
            base_path=tmp_path / "run",
            shape_prior_backend="trellis2",
            gaussian_backend="mesh_surface",
        )
        assert session.gaussian_backend == "mesh_surface"

    def test_camera_service_parser_accepts_mesh_surface(self) -> None:
        from demo_v7.service.camera_service import _build_v7_parser

        v7_args, _rest = _build_v7_parser().parse_known_args(
            [
                "--socket-dir",
                "/tmp/x",
                "--gaussian-backend",
                "mesh_surface",
                "--input-source",
                "fake-live",
            ]
        )
        assert v7_args.gaussian_backend == "mesh_surface"

    def test_gui_labels_cover_all_backends(self) -> None:
        from demo_v7 import app as app_module
        from demo_v7.service import gaussian_options as go

        label_ids = [backend for backend, _pair in app_module._GAUSSIAN_LABELS]
        assert tuple(label_ids) == go.GAUSSIAN_BACKENDS


class TestMeshSurfaceManagerLifecycle:
    """Fail-soft lifecycle bits that need no GPU and no mesh."""

    def _manager(self, tmp_path):
        from demo_v7.service.mesh_surface_manager import (
            MeshSurfaceGaussianManager,
        )

        events = {"progress": [], "errors": []}
        manager = MeshSurfaceGaussianManager(
            case_dir=tmp_path / "case",
            out_dir=tmp_path / "gaussian",
            emit_progress=lambda stage, detail="", **kw: events["progress"].append(
                (stage, detail)
            ),
            emit_artifacts=lambda kind, paths: events.setdefault(
                "artifacts", []
            ).append((kind, paths)),
            emit_error=lambda stage, message: events["errors"].append(
                (stage, message)
            ),
        )
        return manager, events

    def test_regen_before_ready_refused(self, tmp_path) -> None:
        manager, _events = self._manager(tmp_path)
        manager.start()
        assert manager.regenerate(7) is False  # chain not READY yet

    def test_missing_mesh_is_display_only_error(self, tmp_path) -> None:
        manager, events = self._manager(tmp_path)
        manager.start()
        manager.notify_case_ready()
        manager._first_gen.join(timeout=10.0)
        assert events["errors"] and "final_mesh" in events["errors"][0][1]
        assert not manager.has_world_ply()

    def test_shutdown_blocks_generation(self, tmp_path) -> None:
        manager, events = self._manager(tmp_path)
        manager.start()
        manager.shutdown()
        manager.notify_case_ready()
        if manager._first_gen is not None:
            manager._first_gen.join(timeout=10.0)
        assert not events["errors"]
        assert not manager.has_world_ply()
