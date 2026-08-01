"""Invariants for the skinning math and the Sim(3) estimators."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from gaussian_align_demo.alignment import ransac_umeyama, umeyama_sim3
from gaussian_align_demo.dynamic_utils import (
    apply_bone_transforms,
    bind_gaussians,
    build_bone_relations,
    compute_bone_transforms,
    matrix_to_quat_wxyz,
    skin_weights,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class UmeyamaTests(unittest.TestCase):
    def test_recovers_known_sim3(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.normal(size=(50, 3))
        rotation = _random_rotation(rng)
        scale, translation = 0.47, np.array([0.2, -0.4, 0.9])
        dst = scale * src @ rotation.T + translation
        model = umeyama_sim3(src, dst)
        np.testing.assert_allclose(model.rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(model.scale, scale, atol=1e-9)
        np.testing.assert_allclose(model.translation, translation, atol=1e-9)

    def test_ransac_survives_outliers(self) -> None:
        rng = np.random.default_rng(1)
        src = rng.normal(size=(80, 3))
        rotation = _random_rotation(rng)
        dst = 0.5 * src @ rotation.T + np.array([1.0, 2.0, 3.0])
        dst[:30] += rng.normal(scale=0.5, size=(30, 3))  # 37% gross outliers
        result = ransac_umeyama(src, dst, threshold_m=0.01, iterations=500, seed=3)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.num_inliers, 45)
        np.testing.assert_allclose(result.sim3.rotation, rotation, atol=1e-6)
        np.testing.assert_allclose(result.sim3.scale, 0.5, atol=1e-6)


class SkinningTests(unittest.TestCase):
    def _grid_bones(self) -> torch.Tensor:
        axis = torch.linspace(-0.5, 0.5, 4)
        return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)

    def test_rigid_motion_transports_gaussians_exactly(self) -> None:
        """If every bone moves rigidly, every gaussian must follow that rigid map."""
        torch.manual_seed(0)
        bones = self._grid_bones()
        relations = build_bone_relations(bones, 8)
        rng = np.random.default_rng(2)
        rotation = torch.from_numpy(_random_rotation(rng)).float()
        translation = torch.tensor([0.05, -0.02, 0.08])
        curr = bones @ rotation.T + translation

        rotations, translations = compute_bone_transforms(bones, curr, relations)
        for i in range(bones.shape[0]):
            torch.testing.assert_close(rotations[i], rotation, atol=1e-4, rtol=0)

        means = torch.randn(500, 3) * 0.4
        quats = torch.nn.functional.normalize(torch.randn(500, 4), dim=1)
        indices = bind_gaussians(bones, means, 8)
        new_means, new_quats = apply_bone_transforms(
            means, quats, bones, rotations, translations, indices
        )
        torch.testing.assert_close(new_means, means @ rotation.T + translation,
                                   atol=2e-4, rtol=0)
        norms = new_quats.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_zero_motion_is_identity(self) -> None:
        torch.manual_seed(1)
        bones = self._grid_bones()
        relations = build_bone_relations(bones, 8)
        rotations, translations = compute_bone_transforms(bones, bones.clone(), relations)
        eye = torch.eye(3).expand_as(rotations)
        torch.testing.assert_close(rotations, eye, atol=1e-5, rtol=0)
        self.assertEqual(translations.abs().max().item(), 0.0)

        means = torch.randn(200, 3) * 0.4
        quats = torch.nn.functional.normalize(torch.randn(200, 4), dim=1)
        indices = bind_gaussians(bones, means, 8)
        new_means, new_quats = apply_bone_transforms(
            means, quats, bones, rotations, translations, indices
        )
        torch.testing.assert_close(new_means, means, atol=1e-6, rtol=0)
        dots = (new_quats * quats).sum(dim=1).abs()
        torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=0)

    def test_weights_normalized_and_local(self) -> None:
        torch.manual_seed(2)
        bones = self._grid_bones()
        points = torch.randn(100, 3) * 0.3
        indices = bind_gaussians(bones, points, 5)
        weights = skin_weights(bones, points, indices)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(100), atol=1e-5, rtol=0)
        # Nearest bone gets the largest weight.
        self.assertTrue((weights.argmax(dim=1) == 0).all())

    def test_matrix_to_quat_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        mats = np.stack([_random_rotation(rng) for _ in range(64)])
        quats = matrix_to_quat_wxyz(torch.from_numpy(mats).float())
        w, x, y, z = quats.unbind(1)
        rebuilt = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
        ], dim=1)
        torch.testing.assert_close(rebuilt, torch.from_numpy(mats).float(), atol=1e-5, rtol=0)
        self.assertTrue((w >= 0).all())


if __name__ == "__main__":
    unittest.main()
