"""Unit tests for the gaussian-splats feature (CPU only; no GPU, no models).

Covers the vendored dynamics math (quaternion round trips incl. the
trace<=-1 branch upstream got wrong, hemisphere-aligned blending), the
similarity transform on splats and its ply IO round trip.
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
    @pytest.mark.parametrize("case", ["random", "near_pi"])
    def test_roundtrip_near_pi_rotations(self, case: str) -> None:
        """quat<->mat round trip, including the trace<=-1 branches.

        Guards demo_v7/service/gaussian_dynamics.py:58 (mask_1/2/3):
        reverting mat2quat to upstream's mask_0-only (dead-code) form fails
        the `near_pi` row alone, since the `random` row never leaves mask_0
        and only rides along as the general-position control. The branch is
        reached in production whenever a bone neighbourhood rotates near
        180 deg.
        """
        if case == "random":
            mats = _random_rotations(256)
        else:
            # trace <= -1 exercises the mask_1/2/3 branches (upstream's
            # dead-code bug lived here); rotations by pi about each
            # principal axis.
            mats = torch.as_tensor(
                np.stack(
                    [2.0 * np.outer(a, a) - np.eye(3) for a in np.eye(3)]
                ),
                dtype=torch.float64,
            )
        back = gaussian_dynamics.quat2mat(gaussian_dynamics.mat2quat(mats))
        assert torch.allclose(back, mats, atol=1e-6)


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
        """A similarity must scale the sigmas, and a general linear part
        must be refused.

        Guards demo_v7/service/gaussian_utils.py:209 (`scales=(splats.scales
        * scale)`) with the cbrt(det) extraction at :194 — copying the rest
        sigmas instead of scaling them would ship full-size splats for a
        0.5x registration — and :198-202, the non-uniform rejection.
        """
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
        # A general (non-similarity) linear part has no single splat scale:
        # it must raise rather than silently shear the gaussians.
        with pytest.raises(ValueError, match="similarity"):
            transform_gaussians(self._splats(8), np.diag([1.0, 2.0, 3.0, 1.0]))

    def test_ply_roundtrip(self, tmp_path) -> None:
        """Guards demo_v7/service/gaussian_utils.py:152/154 (opacity logit +
        log(sigma) re-encode) against the decode at :128-129 — the INRIA
        layout is an external contract, so writing activated opacity/scales
        straight into the ply is a plausible 'simplification'."""
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


class TestGaussianBackendSelector:
    """GUI selector vocabulary + session-level forcing rules."""

    def test_normalize_defaults_and_ids(self) -> None:
        """Guards demo_v7/service/gaussian_options.py:46 (`str(value).strip()
        .lower()`, empty -> default) and :50-53 (unknown -> ValueError).

        Dropping strip()/lower() fails this test alone; the value arrives
        from the GUI combo data, session json and the CLI, so whitespace and
        case normalisation is load-bearing.
        """
        from demo_v7.service import gaussian_options as go

        assert go.normalize_gaussian_backend(None) == go.GAUSSIAN_TRIPOSPLAT
        assert go.normalize_gaussian_backend("") == go.GAUSSIAN_TRIPOSPLAT
        assert go.normalize_gaussian_backend(" TripoSplat ") == "triposplat"
        assert go.normalize_gaussian_backend("none") == go.GAUSSIAN_NONE
        assert (
            go.normalize_gaussian_backend("mesh_surface")
            == go.GAUSSIAN_MESH_SURFACE
        )
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

    def test_shape_prior_none_forces_gaussian_none(self, tmp_path) -> None:
        """Guards demo_v7/orchestration/session.py:272-273: restoring the
        normalized backend after that forcing line fails this test alone —
        a shape-prior-less run would start a gaussian chain with no mesh to
        align to."""
        session = self._session(
            tmp_path, shape_prior_backend="none", gaussian_backend="triposplat"
        )
        assert session.gaussian_backend == "none"


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


class TestRigidWorldCatchup:
    def _cloud(self) -> np.ndarray:
        rng = np.random.default_rng(23)
        return (rng.normal(size=(3000, 3)) * 0.08 + np.array([0.3, 0.1, 0.05])).astype(
            np.float64
        )

    @pytest.mark.parametrize("case", ["small_rigid_motion", "implausible_jump"])
    def test_rejects_implausible_jump(self, case: str) -> None:
        """Guards demo_v7/service/gaussian_align.py:506-508, the
        max_translation_m / max_rotation_deg rejection.

        Raising those bounds to infinity fails the `implausible_jump` row
        alone: without the gate a degenerate frame-0 cloud teleports the
        whole world gaussian a table-length and the caller gets no signal.
        The `small_rigid_motion` row is the accept half — the gate must not
        reject the motion it exists to let through.
        """
        pytest.importorskip("open3d")
        from demo_v7.service.gaussian_align import rigid_world_catchup

        means = self._cloud()
        if case == "small_rigid_motion":
            angle = np.radians(8.0)
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            target = means @ rotation.T + np.array([0.06, -0.04, 0.0])
        else:
            target = means + np.array([1.5, 0.0, 0.0])  # a table-length away
        opacities = np.full(len(means), 0.9, dtype=np.float32)
        transform, info = rigid_world_catchup(means, opacities, target)
        if case == "small_rigid_motion":
            assert transform is not None, f"catch-up rejected: {info}"
            moved = means @ transform[:3, :3].T + transform[:3, 3]
            assert float(np.abs(moved - target).mean()) < 0.01
        else:
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

    @pytest.mark.parametrize("seeded", [True, False])
    def test_seeded_first_packet_catches_up(self, seeded: bool) -> None:
        """Guards demo_v7/service/gaussian_live.py:239-247, the seq-0
        rest-pose seeding.

        Making seed_rest_positions a no-op (the pre-2026-08-07 first-packet
        -freeze behaviour) fails the `seeded` row: that is the
        stuck-in-old-pose trap, where all object motion between FORMAL seq 0
        and the worker's first packet is silently discarded. The unseeded
        row is the fallback contract — with no seed the first packet BECOMES
        the rest pose, so motion only starts with the second one.
        """
        live, means = self._bare_renderer()
        rest = self._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        shift = np.array([0.09, 0.0, 0.0], dtype=np.float32)
        if seeded:
            live.seed_rest_positions({int(i): rest[i] for i in ids})
        else:
            live.step(rest, ids, np.ones(len(ids), dtype=bool))
            assert not live.rest_seeded
            assert torch.allclose(live._tensors["means"], means)  # no motion
        live.step(rest + shift, ids, np.ones(len(ids), dtype=bool))
        assert live.rest_seeded is seeded
        assert torch.allclose(
            live._tensors["means"], means + torch.as_tensor(shift), atol=1e-3
        )
        if seeded:
            assert live.bones_moved_m > 0.0
            stats = live.follow_stats()
            assert stats is not None and stats["rest_seeded"] is True

    @pytest.mark.parametrize("late_packet_seeds", [True, False])
    def test_partial_seed_waits_then_falls_back(
        self, late_packet_seeds: bool
    ) -> None:
        """Guards demo_v7/service/gaussian_live.py:248-254 (the
        _seed_grace_left branch) plus the :255-267 fallback.

        Deleting the grace branch fails this test: a marginal first packet
        (occlusion / depth dropouts) would permanently forfeit rest seeding
        even though the buffer is a growing union. `late_packet_seeds=True`
        is the pay-off — a fuller packet inside the window completes the
        intersection; False is the give-up path once the window closes.
        """
        live, _means = self._bare_renderer()
        rest = self._grid()
        ids = np.arange(len(rest), dtype=np.int64)
        if late_packet_seeds:
            live.seed_rest_positions({int(i): rest[i] for i in ids})
            # First packet only carries a few object markers (occlusion).
            live.step(rest[:4], ids[:4], np.ones(4, dtype=bool))
        else:
            live.seed_rest_positions({0: rest[0], 1: rest[1]})  # < _MIN_BONES
            live.step(rest, ids, np.ones(len(ids), dtype=bool))
        # Within the grace window a marginal packet must NOT freeze an
        # unseeded bone set — later packets may complete the intersection.
        assert live._bone_ids is None and not live.rest_seeded
        if late_packet_seeds:
            # A later, fuller packet completes the seedable intersection.
            live.step(rest, ids, np.ones(len(ids), dtype=bool))
            assert live.rest_seeded
        else:
            live._seed_grace_left = 0
            live.step(rest, ids, np.ones(len(ids), dtype=bool))
            assert not live.rest_seeded
            assert live._bone_ids is not None and len(live._bone_ids) == len(ids)

    def test_apply_rigid_transform_rotates_quats(self) -> None:
        """Guards demo_v7/service/gaussian_live.py:199-206: rotate the quats,
        not just the means. A means-only catch-up leaves anisotropic splats
        at their pre-catch-up orientations, rendering as smeared shells."""
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
        """Guards demo_v7/service/gaussian_live.py:119-122: np.rint pixel
        rounding AND the visibility & mask_object filter.

        Rounding is what reproduces the tracker's own lift_tracks_yx_to_world
        to 4.3e-8 m (truncating fails here); dropping the object-mask term
        turns off-object queries into bones on the hand or the table.
        """
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
        """Guards the clamp at demo_v7/service/gaussian_live.py:132
        (`amount = float(min(max(amount, 0.0), 1.0))`).

        Reachable: staged_runtime.py:1108 parses DEMO_V7_GAUSSIAN_BG_WHITEN
        as a raw float with no validation, so an operator typo (65 for 0.65)
        would overflow uint8 inside render_over.
        """
        from demo_v7.service.gaussian_live import whiten_background

        frame = np.full((2, 2, 3), 100, dtype=np.uint8)
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

    @staticmethod
    def _grid_bones() -> torch.Tensor:
        xs = torch.linspace(0, 1, 4)
        return torch.stack(
            torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1
        ).reshape(-1, 3)

    @pytest.mark.parametrize("case", ["antipodal_near_pi", "pure_translation"])
    def test_antipodal_neighbor_quats_do_not_cancel(self, case: str) -> None:
        """q and -q encode the same rotation; blending must not cancel.

        Guards the hemisphere alignment at
        demo_v7/service/gaussian_dynamics.py:238-246, right before the
        weighted quaternion sum. `antipodal_near_pi`: two bone clusters
        rotate +179.9 and -179.9 deg about z — a mere 0.2 deg apart as
        ROTATIONS, but mat2quat emits near-antipodal quats ([~0,0,0,+1] vs
        [~0,0,0,-1]). Restoring upstream's raw weighted sum collapses the
        blend toward identity and fails this row alone. `pure_translation`
        is the easy input on the same function: the splats must follow the
        bones exactly and must NOT pick up a rotation.
        """
        if case == "antipodal_near_pi":
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
            knn_k = 6
            shift = None
        else:
            bones = self._grid_bones()
            shift = torch.tensor([0.05, -0.02, 0.11])
            motions = shift[None].repeat(len(bones), 1)
            relations = gaussian_dynamics.get_topk_indices(bones, K=6)
            particles = torch.rand(500, 3)
            quats = torch.zeros(500, 4)
            quats[:, 0] = 1.0
            knn_k = 8
        weights, indices = gaussian_dynamics.knn_weights_sparse(
            bones, particles, K=knn_k
        )
        new_xyz, new_quat = gaussian_dynamics.interpolate_motions_sparse(
            bones, motions, relations, particles, quats, weights, indices,
            device="cpu",
        )
        if case == "antipodal_near_pi":
            blended = new_quat / torch.linalg.norm(new_quat, dim=1, keepdim=True)
            # ~180 deg about z: |z| ~ 1, w ~ 0. The pre-fix raw sum measured
            # w=+0.199 / z=+0.980 here (partial cancellation normalized into a
            # ~23-deg w error); the aligned blend gives w=0.000 / z=1.000.
            assert abs(float(blended[0, 3])) > 0.999
            assert abs(float(blended[0, 0])) < 0.05
        else:
            assert torch.allclose(new_xyz, particles + shift, atol=1e-4)
            # A pure translation must not rotate the splats.
            assert torch.allclose(new_quat, quats, atol=1e-4)


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
        """Guards demo_v7/service/gaussian_live.py:352-355 (local-rigidity
        outlier detection) + :384-386 (consensus substitution).

        Removing ONLY the outlier term (keeping stale healing) fails this
        test alone. Measured defect: 71/3969 bones slid onto the hand, up to
        17cm off-object, dragging their permanently-bound splats.
        """
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

    @pytest.mark.parametrize("motion", ["translation", "rotation"])
    def test_stale_bones_ride_the_visible_half(self, motion: str) -> None:
        """Guards demo_v7/service/gaussian_live.py:317-320 (stale mask from
        _last_seen_step) + :373-387 (valid-neighbour consensus).

        Zeroing the stale mask while keeping outlier detection fails this
        test. Measured defect: up to 160 occlusion-frozen bones at once,
        anchoring their splats in the old pose.

        The `rotation` row runs the same occlusion at PRODUCTION scale
        (12cm object, 3cm bone spacing — sloth-like) because the heal is
        rigid-aware, not a translation average: at demo scale the 5cm
        rigidity threshold keeps fresh bones trusted, while a metre-scale
        grid would flag everyone and disable healing entirely (fail-soft).
        """
        helper = TestGaussianLiveRestSeed()
        if motion == "translation":
            live, means, rest, ids = self._seeded(helper)
        else:
            live, means = helper._bare_renderer()
            xs = np.linspace(0.0, 0.12, 5, dtype=np.float32)
            rest = (
                np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1)
                .reshape(-1, 3)
            )
            ids = np.arange(len(rest), dtype=np.int64)
            live.seed_rest_positions({int(i): rest[i] for i in ids})
            live.step(rest, ids, np.ones(len(ids), dtype=bool))  # init at rest
        seen = ids[ids % 2 == 0]
        if motion == "translation":
            step = np.array([0.01, 0.0, 0.0], dtype=np.float32)
            # 14 packets (> _BONE_STALE_STEPS) where only even bones update.
            for k in range(1, 15):
                live.step(
                    rest[seen] + step * k, seen, np.ones(len(seen), dtype=bool)
                )
        else:
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
            # 14 packets (> stale threshold) where only even bones update,
            # at the ROTATED pose.
            for _ in range(14):
                live.step(rotated[seen], seen, np.ones(len(seen), dtype=bool))
        assert live.bone_stale > 0
        if motion == "translation":
            # Stale odd bones must not anchor the object at the rest pose:
            # the whole cloud rides the visible bones' translation.
            assert torch.allclose(
                live._tensors["means"],
                means + torch.as_tensor(step * 14),
                atol=2e-3,
            )
        else:
            applied = live._ctrl_prev.cpu().numpy()
            stale_ids = ids[ids % 2 == 1]
            error = np.linalg.norm(
                applied[stale_ids] - rotated[stale_ids], axis=1
            )
            # A global rigid motion is EXACT under the per-bone Kabsch blend;
            # a mean-consensus heal leaves ~1cm errors at this scale.
            assert float(error.max()) < 0.005, error.max()


class TestFloaterPruning:
    def test_disconnected_island_pruned_even_near_mesh(self) -> None:
        """The measured failure: a 212-splat island 10.6cm from every other
        splat but 3.2cm from the mesh (the mesh arm tip passed nearby) —
        connectivity must catch what mesh distance cannot.

        Guards demo_v7/service/gaussian_align.py:390-414 (the connectivity
        criterion plus fuzz inheriting its nearest solid splat's verdict);
        reverting _floater_keep_mask to mesh-distance-only fails it alone.
        """
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
        independently-registered gaussian double-corrects (benchmarked).

        Guards the Umeyama similarity strip at
        demo_v7/service/gaussian_selfalign.py:157-166; returning the raw
        ARAP field fails this test alone.
        """
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

    def test_subprocess_error_json_is_fail_soft(self, tmp_path) -> None:
        """A child that writes an error payload must yield None, not raise.

        Guards demo_v7/service/gaussian_selfalign.py:255-258 (`if "error" in
        result: return None`): parsing `result["gates"]` without the error
        check raises KeyError through here. Phase 2 must degrade to the
        published chamfer alignment, never take down the background thread.
        """
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
        """Guards demo_v7/service/arap_rescue.py:95-103 (drop <1% connected
        components after the stock cleanup); a no-op
        patch_asap_island_cleanup fails this test alone.

        Measured defect: o3d's own remove_non_manifold_edges cuts 1-31
        triangle islands loose, an unconstrained island makes the ARAP
        factorization singular at ANY scale, and whole runs died on
        numerical luck.
        """
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
        """Guards the _ART_PREFERENCE_MARGIN at
        demo_v7/service/gaussian_selfalign.py:69-72: dropping the margin (any
        win picks the articulation variant) fails this test alone — the C2
        variant's benefit was case-dependent in the benchmark, so a 0.005
        noise win must not flip the published alignment."""
        from demo_v7.service import gaussian_selfalign as sa

        scored = [
            ("self_align", {"iou": 0.90, "c2g_p90_cm": 1.5}),
            ("self_align_art", {"iou": 0.905, "c2g_p90_cm": 1.5}),  # +0.005
        ]
        assert sa.pick_candidate(scored)[0] == "self_align"
        scored[1] = ("self_align_art", {"iou": 0.95, "c2g_p90_cm": 1.5})
        assert sa.pick_candidate(scored)[0] == "self_align_art"

    def test_swap_default_unless_clear_loss(self) -> None:
        """Guards _KEEP_INCUMBENT_MARGIN at
        demo_v7/service/gaussian_selfalign.py:78-80 (self-align wins ties —
        the owner decision from 4f4793b; inverting the tolerance so the
        chamfer incumbent wins ties fails this test alone) AND the tail term
        of combined_score at :59-61, via the last row: its candidate differs
        from the incumbent in c2g_p90_cm, so an iou-only score would swap.
        """
        from demo_v7.service import gaussian_selfalign as sa

        incumbent = {"iou": 0.70, "c2g_p90_cm": 2.0}
        rows = [
            # (candidate metrics, should_swap, why)
            ({"iou": 0.695, "c2g_p90_cm": 2.0}, True, "-0.005: within tolerance"),
            ({"iou": 0.60, "c2g_p90_cm": 5.0}, False, "drive21-gen1 class loss"),
            ({"iou": 0.92, "c2g_p90_cm": 3.3}, True, "a clear win"),
            (
                {"iou": 0.72, "c2g_p90_cm": 4.0},
                False,
                "+2 IoU points do not pay for 2cm of coverage tail at 3pt/cm",
            ),
        ]
        for candidate, expected, why in rows:
            assert sa.should_swap(candidate, incumbent) is expected, why


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

    def test_every_face_sampled_and_deterministic(self, tmp_path) -> None:
        """Guards demo_v7/service/mesh_surface_gaussian.py:195-196 (seeded
        generator -> bit-identical splats for the same mesh+seed); letting
        the sampler use a fresh default_rng fails this test alone.
        Determinism is the backend's contract: a REVIEW re-roll is
        seed-addressed and the anchors npz must match the published ply.

        The `bary >= -1e-6` check guards the fold-reflection at :128-129 —
        without it a splat center lands OUTSIDE its own triangle.
        """
        from demo_v7.service.mesh_surface_gaussian import gaussianize_mesh

        path = _synthetic_glb(tmp_path)
        splats_a, anchors_a = gaussianize_mesh(path, target_splats=1000, seed=5)
        assert len(np.unique(anchors_a.face_index)) == len(anchors_a.faces)
        assert (anchors_a.barycentric >= -1e-6).all()
        splats_b, anchors_b = gaussianize_mesh(path, target_splats=1000, seed=5)
        assert np.array_equal(splats_a.means, splats_b.means)
        assert np.array_equal(anchors_a.barycentric, anchors_b.barycentric)
        splats_c, _ = gaussianize_mesh(path, target_splats=1000, seed=6)
        assert not np.array_equal(splats_a.means, splats_c.means)

    def test_anchors_roundtrip_and_hash_guard(self, tmp_path) -> None:
        """Guards demo_v7/service/mesh_surface_gaussian.py:247-253, the
        self-verifying topology hash on load: a load_anchors that trusts the
        file fails this test alone. sample_asap_safe re-cleans and rewrites
        final_mesh.glb concurrently, so anchors mixed across cleanings would
        silently drift instead of failing loudly."""
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
        """Guards the `live = areas > _MIN_FACE_AREA` mask at
        demo_v7/service/mesh_surface_gaussian.py:108 + :118: allocating from
        raw areas (so degenerate faces get their >=1 sample) fails this test
        alone. TRELLIS.2/ARAP meshes carry sliver faces; a sample on one
        yields a degenerate frame and a 1e5 area ratio in the live scale
        correction."""
        from demo_v7.service.mesh_surface_gaussian import _allocate_samples

        areas = np.array([1.0e-4, 0.0, 2.0e-4, 1.0e-20])
        counts = _allocate_samples(areas, 300)
        assert counts[1] == 0 and counts[3] == 0
        assert counts[0] >= 1 and counts[2] >= 1
        assert counts.sum() >= 300

    def test_face_frames_orthonormal_right_handed(self) -> None:
        """Guards `frames[~ok] = np.eye(3)` at
        demo_v7/service/mesh_surface_gaussian.py:94 — the sliver-face
        landmine: dividing through on degenerate faces without the identity
        fallback feeds a non-orthonormal frame into _frames_to_wxyz."""
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


class TestMeshAnchoredRenderer:
    """Live mesh-anchored deformation: bones deform the vertices, splats
    stay ON the deformed mesh (CPU, bare construction like the parent's
    tests)."""

    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

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
        tri_rest = live._verts[live._faces]
        live._rest_double_area = (
            torch.cross(
                tri_rest[:, 1] - tri_rest[:, 0],
                tri_rest[:, 2] - tri_rest[:, 0],
                dim=1,
            )
            .norm(dim=1)
            .clamp(min=1e-12)
        )
        live._rest_scales = live._tensors["scales"].clone()
        edges = torch.cat(
            [live._faces[:, [0, 1]], live._faces[:, [1, 2]], live._faces[:, [2, 0]]],
            dim=0,
        )
        live._edges = torch.unique(torch.sort(edges, dim=1).values, dim=0)
        live._edge_rest_len = (
            live._verts[live._edges[:, 0]] - live._verts[live._edges[:, 1]]
        ).norm(dim=1).clamp(min=1e-6)
        degree = torch.zeros(live._verts.shape[0])
        ones = torch.ones(live._edges.shape[0])
        degree.scatter_add_(0, live._edges[:, 0], ones)
        degree.scatter_add_(0, live._edges[:, 1], ones)
        live._edge_degree = degree.clamp(min=1.0)
        means, quats = live._replay(live._verts)
        live._tensors["means"] = means
        live._tensors["quats"] = quats
        return live, anchors

    def test_rigid_transform_moves_verts_and_splats_together(
        self, tmp_path
    ) -> None:
        """Guards the mesh-anchored override at
        demo_v7/service/gaussian_live.py:664-680 (transform the VERTICES and
        replay, not the splats). Letting MeshAnchoredGaussianRenderer inherit
        the parent's splat-only implementation fails this test alone: means
        would move while _verts stayed behind, breaking the
        face_id+barycentric binding on the very next frame."""
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
        """Guards demo_v7/service/gaussian_live.py:623-625 (_skin_targets
        returns the mesh VERTICES): binding the bones to the splat means
        instead fails this test alone — the mesh would stop following the
        bones while the splats did, i.e. the mesh stops being the geometry
        truth in motion."""
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

    def test_scales_track_triangle_stretch(self, tmp_path) -> None:
        """A stretched triangle grows its splats; a rigid move does not.

        Guards demo_v7/service/gaussian_live.py:658-661 (tangential sigmas
        ride sqrt(face area ratio); column 2 does not) — freezing the
        footprints at the rest sigma (the pre-bf6033a form) fails this test
        alone. Measured on a 606-frame manipulation session: 6.6% of mesh
        edges exceed 1.5x stretch and 1.0% exceed 3x, so a rest-fixed
        footprint leaves holes exactly where the object deforms most.
        """
        live, _anchors = self._bare(tmp_path)
        rest_scales = live._tensors["scales"].clone()
        # Uniform 2x blow-up: every triangle doubles linearly, so every
        # tangential sigma should double; the surfel thickness must not.
        live._replay(live._verts * 2.0)
        grown = live._tensors["scales"]
        assert torch.allclose(grown[:, :2], rest_scales[:, :2] * 2.0, rtol=1e-3)
        assert torch.allclose(grown[:, 2], rest_scales[:, 2])
        # A rigid move changes no area, so the sigmas must come back.
        live._replay(live._verts + 0.07)
        assert torch.allclose(
            live._tensors["scales"], rest_scales, rtol=1e-4, atol=1e-9
        )

    def test_pose_solve_applies_the_projection(self, tmp_path) -> None:
        """Guards the ``_project_edges`` call in
        MeshAnchoredGaussianRenderer._pose_to: testing the projection in
        isolation leaves the call site free to be deleted, and raw LBS
        stretch (measured 4.11% of edges past 1.5x on a manipulation
        session) would silently come back.
        """
        live, _anchors = self._bare(tmp_path)
        verts = live._verts.numpy()
        rng = np.random.default_rng(5)
        rows = rng.choice(len(verts), size=60, replace=False)
        rest_bones = verts[rows].astype(np.float32)
        ids = np.arange(len(rest_bones), dtype=np.int64)
        live.seed_rest_positions({int(i): rest_bones[i] for i in ids})
        live.step(rest_bones, ids, np.ones(len(ids), dtype=bool))
        # Bones pulled 60% apart: raw LBS stretches the edges with them.
        target = torch.as_tensor(rest_bones * 1.6)
        live.step(target.numpy(), ids, np.ones(len(ids), dtype=bool))
        edges = live._edges

        def max_strain(verts):
            return float(
                (
                    (verts[edges[:, 0]] - verts[edges[:, 1]]).norm(dim=1)
                    / live._edge_rest_len
                ).max()
            )

        # Self-calibrating: the same solve WITHOUT the projection, so the
        # assertion cannot drift with the iteration count or stiffness.
        raw, _ = gaussian_dynamics.interpolate_motions_sparse(
            live._ctrl_rest,
            target - live._ctrl_rest,
            live._relations,
            live._verts_rest,
            None,
            live._skin_weights,
            live._skin_indices,
            device="cpu",
        )
        assert max_strain(live._verts) < 0.9 * max_strain(raw), (
            f"projected {max_strain(live._verts):.2f}x vs raw LBS "
            f"{max_strain(raw):.2f}x — the pose solve did not project"
        )

    def test_edge_projection_is_a_noop_under_rigid_motion(self, tmp_path) -> None:
        """A rigid move leaves every edge at its rest length, so the shape
        projection (demo_v7/service/gaussian_live.py:712-734) must not touch
        it — otherwise it would fight the bones. A 1% shrink inside
        _project_edges fails only this test."""
        live, _anchors = self._bare(tmp_path)
        angle = np.radians(30.0)
        rotation = torch.as_tensor(
            self._rot_z(angle), dtype=torch.float32
        )
        moved = live._verts @ rotation.T + torch.tensor([0.05, -0.02, 0.03])
        assert torch.allclose(live._project_edges(moved), moved, atol=1e-6)

    def test_edge_projection_pulls_stretch_back(self, tmp_path) -> None:
        """A stretched mesh is pulled back toward its rest edge lengths.

        Guards the Jacobi edge projection at
        demo_v7/service/gaussian_live.py:702 + :712-734; making
        _project_edges the identity fails this test alone. The shipped
        51fbb32 A/B on a 606-frame manipulation session: LBS left 4.11% of
        edges beyond 1.5x rest; the shipped 20 iterations bring that to
        0.53% while IMPROVING distance to the observed cloud.
        """
        live, _anchors = self._bare(tmp_path)
        stretched = live._verts * 1.6  # every edge at 1.6x rest
        edges = live._edges
        def frac_over(v):
            length = (v[edges[:, 0]] - v[edges[:, 1]]).norm(dim=1)
            return float((length / live._edge_rest_len > 1.5).float().mean())
        assert frac_over(stretched) > 0.99
        assert frac_over(live._project_edges(stretched)) < 0.05

    def test_degenerate_face_keeps_last_orientation(self, tmp_path) -> None:
        """Guards the _face_quats_prev carry-over at
        demo_v7/service/gaussian_live.py:642-648: dropping the torch.where
        fails this test alone — a transiently-collapsed triangle would snap
        its splats to a neutral orientation for that frame instead of
        holding the last good one."""
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

    @pytest.mark.parametrize(
        "shape_prior, accepted", [("trellis2", True), ("sam3d", False)]
    )
    def test_session_rejects_mesh_surface_without_trellis2(
        self, tmp_path, shape_prior: str, accepted: bool
    ) -> None:
        """Guards the cross-option fail-fast at
        demo_v7/orchestration/session.py:274-288: neutralising
        mesh_surface_allowed fails the `sam3d` row. mesh_surface derives its
        splats from the trellis2 chain's final_mesh.glb, so with any other
        mesh backend there is nothing to derive from — and the CLI/config
        path has no combo gating. The trellis2 row is the over-strict-rule
        control.
        """
        from demo_v7.orchestration.session import OrchestratorSession

        def _build():
            return OrchestratorSession(
                source="fake-live",
                fake_live_case="data_collect/fake",
                base_path=tmp_path / "run",
                shape_prior_backend=shape_prior,
                gaussian_backend="mesh_surface",
            )

        if accepted:
            assert _build().gaussian_backend == "mesh_surface"
        else:
            with pytest.raises(ValueError, match="mesh_surface"):
                _build()

    def test_gui_labels_cover_all_backends(self) -> None:
        """Guards demo_v7/app.py:93-103: _GAUSSIAN_LABELS must cover
        gaussian_options.GAUSSIAN_BACKENDS. The realistic future edit — add a
        backend to the vocabulary, forget the GUI label — leaves the option
        unreachable from the source dialog with no other signal."""
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
        """Guards `or not self._case_ready.is_set()` in try_reserve
        (demo_v7/service/mesh_surface_manager.py:143): dropping the readiness
        term fails this test alone — a REVIEW re-roll arriving before the
        shape-prior chain is READY would be acked ok and then die on a
        missing final_mesh.glb, leaving the GUI spinning."""
        manager, _events = self._manager(tmp_path)
        manager.start()
        assert manager.regenerate(7) is False  # chain not READY yet

    def test_missing_mesh_is_display_only_error(self, tmp_path) -> None:
        """Guards demo_v7/service/mesh_surface_manager.py:193-196 raising
        into the :214-218 fail-soft handler (EVT_ERROR, no ply, run
        continues): swallowing the error instead of emitting it fails this
        test alone. This is the display-only contract the whole feature
        rests on."""
        manager, events = self._manager(tmp_path)
        manager.start()
        manager.notify_case_ready()
        manager._first_gen.join(timeout=10.0)
        assert events["errors"] and "final_mesh" in events["errors"][0][1]
        assert not manager.has_world_ply()


class TestNeighbourGraph:
    """Guards gaussian_dynamics.get_topk_indices' documented "no self".

    Bones are tracker queries lifted through a ROUNDED pixel, so two queries
    on one pixel give a bitwise identical point (measured: 15.6-15.8% of
    object bones at frame 0 of two archived sessions had their own index in
    `relations`). Dropping column 0 of a (K+1)-topk assumes the zero-distance
    self always sorts first, which ties do not guarantee.
    """

    def test_duplicate_points_never_neighbour_themselves(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
        )
        indices = gaussian_dynamics.get_topk_indices(points, K=3)
        own = torch.arange(points.shape[0])[:, None]
        assert not bool((indices == own).any()), f"self in its own graph: {indices}"


class TestBoneTransformsAreRotations:
    """compute_bone_transforms must return SO(3) for every neighbourhood.

    Guards gaussian_dynamics.compute_bone_transforms' reflection handling.
    The old repair negated one entry (``R[2, 2] *= -1``), which is only a
    valid SO(3) projection when the reflection axis happens to sit on a
    coordinate axis; off-axis it leaves ``R^T R != I`` — shear applied to
    mesh vertices and splat orientations as if it were rigid. The
    replacement builds the correction from ``det(U V^T)``, correct at every
    rank including the exactly planar patches where ``det(F)`` carries no
    sign. ``test_tilted_planar_patch_stays_in_so3`` fails on the old code.
    """

    def _solve(self, points: np.ndarray, rotation: np.ndarray, translation):
        pts = torch.as_tensor(points, dtype=torch.float32)
        rot = torch.as_tensor(rotation, dtype=torch.float32)
        motions = pts @ rot.T + torch.as_tensor(
            np.asarray(translation), dtype=torch.float32
        ) - pts
        relations = gaussian_dynamics.get_topk_indices(
            pts, K=min(8, len(pts) - 1)
        )
        transforms = gaussian_dynamics.compute_bone_transforms(
            pts, motions, relations, device="cpu"
        )
        return transforms[:, :3, :3]

    def _assert_so3(self, rotations, *, atol=1e-4) -> None:
        eye = torch.eye(3)[None].expand_as(rotations)
        orth = torch.linalg.norm(
            rotations.transpose(1, 2) @ rotations - eye, dim=(1, 2)
        )
        dets = torch.linalg.det(rotations)
        assert float(orth.max()) < atol, f"not orthogonal: {float(orth.max())}"
        assert float((dets - 1.0).abs().max()) < atol, f"det != 1: {dets}"

    def _rotation(self, degrees: float, axis: int = 2) -> np.ndarray:
        angle = np.radians(degrees)
        c, s = np.cos(angle), np.sin(angle)
        base = {
            0: [[1, 0, 0], [0, c, -s], [0, s, c]],
            1: [[c, 0, s], [0, 1, 0], [-s, 0, c]],
            2: [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        }[axis]
        return np.asarray(base, dtype=np.float64)

    def test_tilted_planar_patch_stays_in_so3(self) -> None:
        """Flat bone patch whose plane normal is NOT a coordinate axis.

        Guards demo_v7/service/gaussian_dynamics.py:158-167 (correction from
        ``det(U V^T)`` instead of ``det(F)`` + ``R[2, 2] *= -1``): restoring
        the pre-07ab5e4 implementation fails THIS TEST AND NO OTHER in the
        file. ``tilt=0`` is the axis-aligned planar patch, where the SVD's
        third vectors land back on a coordinate axis so ``R[2, 2] == +/-1``
        and the single-entry flip happens to be a valid projection — it
        passes on the buggy code and is here only as the control. Every
        non-zero tilt puts the normal off-axis, which makes ``R[2, 2]``
        generic and leaves ``||R^T R - I||`` up to 1.4 on the old code.
        """
        rng = np.random.default_rng(1)
        flat = rng.random((12, 2)) * 0.12
        for tilt_axis in (0, 1):
            for tilt in (0.0, 10.0, 25.0, 35.0, 50.0, 65.0, 80.0):
                pts = flat @ self._rotation(tilt, tilt_axis)[:, :2].T
                self._assert_so3(self._solve(pts, self._rotation(40.0), 0.0))


class TestMeshSurfaceShutdownQuiesces:
    """shutdown() must CANCEL AND JOIN an in-flight derivation.

    The staged runtime calls it right before FORMAL so the camera GPU is
    free; the earlier version only set a flag, so a derivation already past
    the entry check kept running — rendering the overlay on CUDA and
    publishing artifacts into a run that had moved on.
    """

    def test_inflight_derivation_is_cancelled_and_joined(
        self, tmp_path, monkeypatch
    ) -> None:
        import threading
        import time as _time

        import demo_v7.service.mesh_surface_gaussian as msg
        from demo_v7.service.mesh_surface_manager import (
            MeshSurfaceGaussianManager,
        )

        mesh = _synthetic_glb(tmp_path)
        case_dir = tmp_path / "case"
        (case_dir / "shape" / "matching").mkdir(parents=True)
        (case_dir / "shape" / "matching" / "final_mesh.glb").write_bytes(
            mesh.read_bytes()
        )
        events: dict[str, list] = {"artifacts": [], "errors": []}
        manager = MeshSurfaceGaussianManager(
            case_dir=case_dir,
            out_dir=tmp_path / "gaussian",
            emit_progress=lambda *a, **k: None,
            emit_artifacts=lambda kind, paths: events["artifacts"].append(paths),
            emit_error=lambda stage, msg_: events["errors"].append(msg_),
        )
        started = threading.Event()
        real = msg.gaussianize_mesh

        def _slow(path, **kwargs):
            started.set()
            _time.sleep(0.3)  # still working when shutdown lands
            return real(path, **kwargs)

        monkeypatch.setattr(msg, "gaussianize_mesh", _slow)
        manager.start()
        manager.notify_case_ready()
        assert started.wait(timeout=10.0)

        manager.shutdown(timeout_s=10.0)
        assert manager._first_gen is not None
        assert not manager._first_gen.is_alive()  # joined, not just flagged
        assert events["artifacts"] == []  # nothing published after cancel
        assert not manager.world_ply_path.is_file()
        # Closed stays closed: _generate's own entry check
        # (mesh_surface_manager.py:182-185) refuses every later re-roll.
        assert manager.regenerate(7) is False
