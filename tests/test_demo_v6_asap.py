"""Unit tests for Demo v6 live ASAP augmentation (design_spec_v6.md)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo_v6 import asap

OBJECT_COUNT = 60
SURFACE_COUNT = 20
INTERIOR_COUNT = 15
SHIFT_M = np.asarray([0.02, 0.0, 0.0], dtype=np.float32)


def _make_case(tmpdir: str) -> tuple[Path, np.ndarray]:
    """Write a trimesh icosphere as final_mesh.glb; return (case_dir, vertices)."""
    import trimesh

    case = Path(tmpdir) / "case"
    mesh_dir = case / "shape" / "matching"
    mesh_dir.mkdir(parents=True)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    sphere.export(mesh_dir / "final_mesh.glb")
    return case, np.asarray(sphere.vertices, dtype=np.float64)


def _fixture(tmpdir: str) -> dict[str, object]:
    case, vertices = _make_case(tmpdir)
    rng = np.random.default_rng(0)
    object0 = vertices[
        rng.choice(vertices.shape[0], OBJECT_COUNT, replace=False)
    ].astype(np.float32)
    surface = vertices[
        rng.choice(vertices.shape[0], SURFACE_COUNT, replace=False)
    ].astype(np.float32)
    interior = (
        vertices[rng.choice(vertices.shape[0], INTERIOR_COUNT, replace=False)] * 0.5
    ).astype(np.float32)
    return {
        "case": case,
        "metadata": {"shape_prior_case_dir": str(case)},
        "object0": object0,
        "surface": surface,
        "interior": interior,
    }


def _window(
    object0: np.ndarray,
    *,
    frame_count: int,
    shift_per_frame: np.ndarray = SHIFT_M,
) -> dict[str, np.ndarray]:
    count = object0.shape[0]
    points = np.stack(
        [object0 + t * shift_per_frame for t in range(frame_count)], axis=0
    ).astype(np.float32)
    return {
        "object_points": points,
        "object_visibilities": np.ones((frame_count, count), dtype=bool),
        "object_motions_valid": np.ones((frame_count, count), dtype=bool),
        "object_colors": np.full((frame_count, count, 3), 0.5, dtype=np.float32),
        "object_sample_query_ids": np.arange(count, dtype=np.int64),
        "object_selected_query_ids": np.arange(count, dtype=np.int64),
        "object_volume_sample_indices": np.arange(count, dtype=np.int64),
        "object_track_status": np.full((count,), "direct", dtype="<U8"),
    }


class MeshResolutionTests(unittest.TestCase):
    def test_missing_case_dir_fails_fast(self) -> None:
        with self.assertRaisesRegex(asap.AsapMeshError, "shape_prior_case_dir"):
            asap.resolve_final_mesh_path({})

    def test_missing_mesh_file_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(asap.AsapMeshError, "not found"):
                asap.resolve_final_mesh_path({"shape_prior_case_dir": tmpdir})

    def test_override_must_exist(self) -> None:
        with self.assertRaisesRegex(asap.AsapMeshError, "override"):
            asap.resolve_final_mesh_path({}, override="/nonexistent/mesh.glb")

    def test_metadata_resolution_finds_final_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case, _vertices = _make_case(tmpdir)
            path = asap.resolve_final_mesh_path({"shape_prior_case_dir": str(case)})
            self.assertTrue(path.is_file())


def _fit_weighted_rigid_original(src, dst, weights):
    """Verbatim downstream original (july2_chunk_vis.py::fit_weighted_rigid)."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    ca = np.sum(src * weights[:, None], axis=0)
    cb = np.sum(dst * weights[:, None], axis=0)
    src0 = src - ca
    dst0 = dst - cb
    covariance = (src0 * weights[:, None]).T @ dst0
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = cb - rotation @ ca
    return rotation, translation


class RigidFitParityTests(unittest.TestCase):
    def test_batch_fit_matches_original_on_rotations_and_reflections(self) -> None:
        rng = np.random.default_rng(7)
        batch, k = 24, 8
        src = rng.normal(size=(batch, k, 3))
        # Random rotations + translations + noise.
        angles = rng.uniform(0.0, np.pi, size=batch)
        dst = np.empty_like(src)
        for m in range(batch):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            cos_a, sin_a = np.cos(angles[m]), np.sin(angles[m])
            cross = np.cross(np.eye(3), axis)
            rotation = cos_a * np.eye(3) + sin_a * cross + (1 - cos_a) * np.outer(
                axis, axis
            )
            dst[m] = src[m] @ rotation.T + rng.normal(size=3)
            dst[m] += rng.normal(scale=1e-3, size=(k, 3))
        # A mirrored near-planar set exercises the det<0 reflection guard.
        planar = rng.normal(size=(k, 3))
        planar[:, 2] *= 1e-6
        src[0] = planar
        dst[0] = planar * np.asarray([1.0, 1.0, -1.0])
        weights = rng.uniform(0.1, 1.0, size=(batch, k))

        rot_batch, trans_batch = asap.fit_weighted_rigid_batch(src, dst, weights)
        for m in range(batch):
            rot_ref, trans_ref = _fit_weighted_rigid_original(
                src[m], dst[m], weights[m]
            )
            np.testing.assert_allclose(rot_batch[m], rot_ref, atol=1e-10)
            np.testing.assert_allclose(trans_batch[m], trans_ref, atol=1e-10)
            self.assertGreater(float(np.linalg.det(rot_batch[m])), 0.0)


class SyntheticIdTests(unittest.TestCase):
    def test_id_ranges_are_disjoint_and_detectable(self) -> None:
        surface_ids = asap.surface_query_ids(3)
        interior_ids = asap.interior_query_ids(3)
        tracker_ids = np.arange(10_000, dtype=np.int64)
        self.assertFalse(np.intersect1d(surface_ids, interior_ids).size)
        self.assertFalse(np.intersect1d(surface_ids, tracker_ids).size)
        self.assertFalse(np.intersect1d(interior_ids, tracker_ids).size)
        self.assertTrue(bool(asap.is_surface_query_id(surface_ids).all()))
        self.assertFalse(bool(asap.is_surface_query_id(interior_ids).any()))
        self.assertTrue(bool(asap.is_interior_query_id(interior_ids).all()))
        self.assertFalse(bool(asap.is_interior_query_id(tracker_ids).any()))


class AugmentWindowTests(unittest.TestCase):
    def _augmented(self, fixture, window):
        runtime = asap.AsapRuntime()
        return (
            runtime,
            *runtime.augment_window(
                window,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            ),
        )

    def test_layout_masks_colors_ids_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            window = _window(fixture["object0"], frame_count=3)
            window["object_visibilities"][2, 5] = False
            window["object_points"][2, 5] = 0.0
            _runtime, out, summary = self._augmented(fixture, window)

            total = OBJECT_COUNT + SURFACE_COUNT + INTERIOR_COUNT
            self.assertEqual(out["object_points"].shape, (3, total, 3))
            self.assertEqual(out["object_colors"].shape, (3, total, 3))
            self.assertEqual(out["object_visibilities"].shape, (3, total))
            self.assertEqual(out["object_motions_valid"].shape, (3, total))
            # Estimated entries keep the original mask values; prior columns
            # are never measurements.
            self.assertFalse(bool(out["object_visibilities"][2, 5]))
            self.assertFalse(bool(out["object_visibilities"][:, OBJECT_COUNT:].any()))
            self.assertFalse(bool(out["object_motions_valid"][:, OBJECT_COUNT:].any()))
            # Valid entries stay bit-exact original measurements.
            np.testing.assert_array_equal(
                out["object_points"][0, :OBJECT_COUNT],
                window["object_points"][0],
            )
            # Default prior colors.
            np.testing.assert_allclose(
                out["object_colors"][0, OBJECT_COUNT],
                np.asarray(asap.SURFACE_DEFAULT_COLOR_RGB, dtype=np.float32),
            )
            np.testing.assert_allclose(
                out["object_colors"][0, OBJECT_COUNT + SURFACE_COUNT],
                np.asarray(asap.INTERIOR_DEFAULT_COLOR_RGB, dtype=np.float32),
            )
            # Synthetic ids, -1 sample-index padding, prior status.
            ids = np.asarray(out["object_sample_query_ids"])
            self.assertEqual(int(ids[OBJECT_COUNT]), asap.SURFACE_QUERY_ID_BASE)
            self.assertEqual(
                int(ids[OBJECT_COUNT + SURFACE_COUNT]), asap.INTERIOR_QUERY_ID_BASE
            )
            self.assertTrue(
                bool(
                    (
                        np.asarray(out["object_volume_sample_indices"])[OBJECT_COUNT:]
                        == -1
                    ).all()
                )
            )
            self.assertEqual(
                str(np.asarray(out["object_track_status"])[OBJECT_COUNT]),
                asap.PRIOR_TRACK_STATUS,
            )
            self.assertEqual(int(summary["asap_estimated_entry_count"]), 1)
            self.assertTrue(summary["asap_augmented"])

    def test_translation_recovers_invalid_entry_and_moves_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            window = _window(fixture["object0"], frame_count=4)
            window["object_visibilities"][2, 5] = False
            window["object_points"][2, 5] = 0.0
            _runtime, out, _summary = self._augmented(fixture, window)

            # Pure translation is ARAP-exact: the invalid entry lands at its
            # translated position and the priors follow the mesh motion.
            expected = fixture["object0"][5] + 2 * SHIFT_M
            np.testing.assert_allclose(
                out["object_points"][2, 5], expected, atol=1e-4
            )
            np.testing.assert_allclose(
                out["object_points"][3, OBJECT_COUNT : OBJECT_COUNT + SURFACE_COUNT],
                fixture["surface"] + 3 * SHIFT_M,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                out["object_points"][3, OBJECT_COUNT + SURFACE_COUNT :],
                fixture["interior"] + 3 * SHIFT_M,
                atol=1e-4,
            )

    def test_thin_constraints_reuse_previous_frame_vertices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            runtime = asap.AsapRuntime()
            window0 = _window(fixture["object0"], frame_count=2)
            runtime.augment_window(
                window0,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            )
            # Second window: frame 1 loses every measurement, so the
            # downstream-provided fallback reuses frame 0's mesh vertices.
            window1 = _window(fixture["object0"], frame_count=2)
            window1["object_points"][0] = fixture["object0"] + 4 * SHIFT_M
            window1["object_points"][1] = 0.0
            window1["object_visibilities"][1] = False
            out, summary = runtime.augment_window(
                window1,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            )
            self.assertEqual(int(summary["asap_fallback_frame_count"]), 1)
            np.testing.assert_allclose(
                out["object_points"][1, :OBJECT_COUNT],
                fixture["object0"] + 4 * SHIFT_M,
                atol=1e-4,
            )

    def test_fill_is_per_frame_not_accumulated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            window = _window(fixture["object0"], frame_count=4)
            window["object_visibilities"][2, 5] = False
            window["object_points"][2, 5] = 0.0
            # Frame 3: the column returns with a measurement deliberately off
            # the mesh trajectory. Per-frame fill (the user contract) must
            # keep it bit-exact; the original's accumulate semantics would
            # replace it with a mesh estimate.
            off_trajectory = (
                fixture["object0"][5]
                + 3 * SHIFT_M
                + np.asarray([0.0, 0.005, 0.0], dtype=np.float32)
            )
            window["object_points"][3, 5] = off_trajectory
            _runtime, out, _summary = self._augmented(fixture, window)
            np.testing.assert_array_equal(
                out["object_points"][3, 5], off_trajectory
            )

    def test_cross_window_fallback_reuses_previous_window_vertices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            runtime = asap.AsapRuntime()
            # Window 0 ends translated by one SHIFT_M.
            window0 = _window(fixture["object0"], frame_count=2)
            runtime.augment_window(
                window0,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            )
            # Window 1 frame 0 has no measurements at all: the fallback must
            # chain across the window boundary and reuse window 0's LAST
            # deformed vertices (translated state), not the reference mesh.
            window1 = _window(fixture["object0"], frame_count=2)
            window1["object_points"][0] = 0.0
            window1["object_visibilities"][0] = False
            window1["object_points"][1] = fixture["object0"] + SHIFT_M
            out, summary = runtime.augment_window(
                window1,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            )
            self.assertGreaterEqual(int(summary["asap_fallback_frame_count"]), 1)
            np.testing.assert_allclose(
                out["object_points"][0, :OBJECT_COUNT],
                fixture["object0"] + SHIFT_M,
                atol=1e-4,
            )

    def test_identity_stable_across_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            runtime = asap.AsapRuntime()
            window0 = _window(fixture["object0"], frame_count=2)
            out0, _ = runtime.augment_window(
                window0,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            )
            window1 = _window(fixture["object0"], frame_count=2)
            out1, _ = runtime.augment_window(
                window1,
                metadata=fixture["metadata"],
                surface_points=fixture["surface"],
                interior_points=fixture["interior"],
            )
            np.testing.assert_array_equal(
                out0["object_sample_query_ids"], out1["object_sample_query_ids"]
            )

    def test_empty_window_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _fixture(tmpdir)
            runtime = asap.AsapRuntime()
            empty = {
                "object_points": np.zeros((0, 0, 3), dtype=np.float32),
                "object_visibilities": np.zeros((0, 0), dtype=bool),
                "object_motions_valid": np.zeros((0, 0), dtype=bool),
                "object_colors": np.zeros((0, 0, 3), dtype=np.float32),
            }
            with self.assertRaises(asap.AsapMeshError):
                runtime.augment_window(
                    empty,
                    metadata=fixture["metadata"],
                    surface_points=fixture["surface"],
                    interior_points=fixture["interior"],
                )


if __name__ == "__main__":
    unittest.main()
