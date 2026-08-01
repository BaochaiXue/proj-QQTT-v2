"""Base-layer invariants: Sim(3) on gaussians, PLY round trip, camera math."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian_align_demo.cameras import (
    intrinsics_for_fov,
    look_at_w2c,
    project_points,
    sample_orbit_w2c,
    unproject_pixels,
)
from gaussian_align_demo.gs_ply import (
    GaussianCloud,
    apply_sim3,
    load_gaussian_ply,
    quat_multiply_wxyz,
    rotation_matrix_to_quat_wxyz,
    save_gaussian_ply,
)


def _random_cloud(rng: np.random.Generator, n: int = 64) -> GaussianCloud:
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        sh_dc=rng.normal(size=(n, 3)).astype(np.float32),
        sh_rest=np.zeros((n, 0), dtype=np.float32),
        opacity_logits=rng.normal(size=(n,)).astype(np.float32),
        log_scales=rng.normal(scale=0.3, size=(n, 3)).astype(np.float32),
        quats_wxyz=quats,
    )


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return _quat_to_matrix(quat)


class QuatTests(unittest.TestCase):
    def test_quat_multiply_matches_matrix_product(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            r1, r2 = _random_rotation(rng), _random_rotation(rng)
            q1 = rotation_matrix_to_quat_wxyz(r1)
            q2 = rotation_matrix_to_quat_wxyz(r2)
            q12 = quat_multiply_wxyz(q1, q2)[0]
            np.testing.assert_allclose(_quat_to_matrix(q12), r1 @ r2, atol=1e-5)

    def test_rotation_quat_round_trip(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            rot = _random_rotation(rng)
            np.testing.assert_allclose(
                _quat_to_matrix(rotation_matrix_to_quat_wxyz(rot)), rot, atol=1e-5
            )


class Sim3Tests(unittest.TestCase):
    def test_covariance_transforms_consistently(self) -> None:
        """Σ' must equal s^2 R Σ R^T — locks quat left-multiply + log-scale shift."""
        rng = np.random.default_rng(2)
        cloud = _random_cloud(rng, n=16)
        rotation = _random_rotation(rng)
        translation = rng.normal(size=3)
        scale = 1.7
        out = apply_sim3(cloud, rotation=rotation, translation=translation, scale=scale)
        np.testing.assert_allclose(
            out.means, (cloud.means @ rotation.T) * scale + translation, atol=1e-4
        )
        for i in range(len(cloud)):
            rot_in = _quat_to_matrix(cloud.quats_wxyz[i])
            scale_in = np.diag(np.exp(cloud.log_scales[i]).astype(np.float64) ** 2)
            cov_in = rot_in @ scale_in @ rot_in.T
            rot_out = _quat_to_matrix(out.quats_wxyz[i])
            scale_out = np.diag(np.exp(out.log_scales[i]).astype(np.float64) ** 2)
            cov_out = rot_out @ scale_out @ rot_out.T
            np.testing.assert_allclose(
                cov_out, scale**2 * rotation @ cov_in @ rotation.T, atol=1e-4
            )

    def test_rejects_improper_rotation_and_bad_scale(self) -> None:
        rng = np.random.default_rng(3)
        cloud = _random_cloud(rng, n=4)
        flip = np.diag([1.0, 1.0, -1.0])
        with self.assertRaises(ValueError):
            apply_sim3(cloud, rotation=flip, translation=np.zeros(3), scale=1.0)
        with self.assertRaises(ValueError):
            apply_sim3(cloud, rotation=np.eye(3), translation=np.zeros(3), scale=0.0)


class PlyRoundTripTests(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        rng = np.random.default_rng(4)
        cloud = _random_cloud(rng, n=32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            save_gaussian_ply(cloud, path)
            loaded = load_gaussian_ply(path)
        np.testing.assert_allclose(loaded.means, cloud.means, atol=1e-6)
        np.testing.assert_allclose(loaded.sh_dc, cloud.sh_dc, atol=1e-6)
        np.testing.assert_allclose(loaded.opacity_logits, cloud.opacity_logits, atol=1e-6)
        np.testing.assert_allclose(loaded.log_scales, cloud.log_scales, atol=1e-6)
        np.testing.assert_allclose(loaded.quats_wxyz, cloud.quats_wxyz, atol=1e-6)


class CameraTests(unittest.TestCase):
    def test_project_unproject_round_trip(self) -> None:
        rng = np.random.default_rng(5)
        K = intrinsics_for_fov(width=848, height=480, fov_x_deg=60.0)
        w2c = look_at_w2c(eye=[1.0, -2.0, 1.5], target=[0.0, 0.0, 0.5])
        points = rng.normal(size=(64, 3)) * 0.2  # around the target, in front
        pixels, depths = project_points(points, K, w2c)
        self.assertTrue(np.all(depths > 0))
        recovered = unproject_pixels(pixels, depths, K, w2c)
        np.testing.assert_allclose(recovered, points, atol=1e-9)

    def test_look_at_axes(self) -> None:
        w2c = look_at_w2c(eye=[0.0, -2.0, 0.0], target=[0.0, 0.0, 0.0], up_hint=[0.0, 0.0, 1.0])
        rotation, translation = w2c[:3, :3], w2c[:3, 3]
        # Camera center maps to origin; target sits on +z at distance 2.
        np.testing.assert_allclose(rotation @ np.array([0.0, -2.0, 0.0]) + translation, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            rotation @ np.array([0.0, 0.0, 0.0]) + translation, [0.0, 0.0, 2.0], atol=1e-12
        )
        # World up (+z) maps to camera -y (image up).
        np.testing.assert_allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-12)

    def test_orbit_geometry(self) -> None:
        center = np.array([0.3, -0.2, 0.8])
        poses = sample_orbit_w2c(
            center=center,
            radius=1.5,
            n_azimuth=8,
            elevations_deg=(-20.0, 15.0, 45.0),
            roll_angles_deg=(0.0, 90.0),
        )
        self.assertEqual(len(poses), 8 * 3 * 2)
        for w2c in poses:
            c2w = np.linalg.inv(w2c)
            eye = c2w[:3, 3]
            self.assertAlmostEqual(np.linalg.norm(eye - center), 1.5, places=9)
            # Center projects onto the optical axis at depth == radius.
            cam = w2c[:3, :3] @ center + w2c[:3, 3]
            np.testing.assert_allclose(cam, [0.0, 0.0, 1.5], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
