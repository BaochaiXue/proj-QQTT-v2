from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from qqtt.demo.render_fastpath import Open3DSceneTensorLayer


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array


class _FakeCore:
    class Tensor:
        @staticmethod
        def from_numpy(array: np.ndarray) -> _FakeTensor:
            return _FakeTensor(array)


class _FakePointCloud:
    def __init__(self, _device: object) -> None:
        self.point = SimpleNamespace()


class _FakeO3D:
    class t:
        class geometry:
            PointCloud = _FakePointCloud


class _FakeRendering:
    class Scene:
        UPDATE_POINTS_FLAG = 1
        UPDATE_COLORS_FLAG = 2


class _FakeInnerScene:
    def __init__(self) -> None:
        self.update_calls = 0

    def update_geometry(self, _name: str, _pcd: object, _flags: int) -> None:
        self.update_calls += 1


class _FakeScene:
    def __init__(self) -> None:
        self.scene = _FakeInnerScene()
        self.add_calls = 0
        self.remove_calls = 0

    def add_geometry(self, _name: str, _pcd: object, _material: object) -> None:
        self.add_calls += 1

    def remove_geometry(self, _name: str) -> None:
        self.remove_calls += 1


class RenderFastpathTest(unittest.TestCase):
    def _layer(self, *, min_capacity: int = 0) -> tuple[Open3DSceneTensorLayer, _FakeScene]:
        scene = _FakeScene()
        layer = Open3DSceneTensorLayer(
            name="pcd",
            o3d_module=_FakeO3D,
            o3c_module=_FakeCore,
            rendering_module=_FakeRendering,
            scene=scene,
            material=object(),
            device=object(),
            min_capacity=min_capacity,
        )
        return layer, scene

    def test_inplace_layer_reuses_capacity_when_point_count_shrinks(self) -> None:
        layer, scene = self._layer()
        points = np.arange(30, dtype=np.float32).reshape(10, 3)
        colors = np.full((10, 3), 255, dtype=np.uint8)
        first = layer.update(points, colors)

        smaller_points = np.arange(18, dtype=np.float32).reshape(6, 3)
        smaller_colors = np.full((6, 3), 64, dtype=np.uint8)
        second = layer.update(smaller_points, smaller_colors)

        self.assertTrue(first.geometry_recreated)
        self.assertTrue(first.tensor_rebound)
        self.assertFalse(second.geometry_recreated)
        self.assertFalse(second.tensor_rebound)
        self.assertEqual(scene.add_calls, 1)
        self.assertEqual(scene.remove_calls, 0)
        self.assertEqual(scene.scene.update_calls, 1)
        self.assertEqual(layer.point_count, 6)
        self.assertEqual(layer.capacity, 10)
        np.testing.assert_allclose(layer.pcd.point.positions.array[:6], smaller_points)
        np.testing.assert_allclose(layer.pcd.point.positions.array[6:, 2], -1.0)
        np.testing.assert_allclose(layer.pcd.point.colors.array[:6], np.float32(64.0 / 255.0))
        np.testing.assert_allclose(layer.pcd.point.colors.array[6:], 0.0)

    def test_inplace_layer_grows_capacity_with_one_recreate(self) -> None:
        layer, scene = self._layer()
        layer.update(np.zeros((4, 3), dtype=np.float32), np.zeros((4, 3), dtype=np.uint8))
        grown = layer.update(np.zeros((7, 3), dtype=np.float32), np.zeros((7, 3), dtype=np.uint8))

        self.assertTrue(grown.geometry_recreated)
        self.assertTrue(grown.tensor_rebound)
        self.assertEqual(scene.add_calls, 2)
        self.assertEqual(scene.remove_calls, 1)
        self.assertEqual(layer.point_count, 7)
        self.assertEqual(layer.capacity, 7)

    def test_inplace_layer_honors_min_capacity(self) -> None:
        layer, scene = self._layer(min_capacity=8)
        update = layer.update(np.ones((3, 3), dtype=np.float32), np.ones((3, 3), dtype=np.uint8))

        self.assertTrue(update.geometry_recreated)
        self.assertEqual(scene.add_calls, 1)
        self.assertEqual(layer.point_count, 3)
        self.assertEqual(layer.capacity, 8)
        self.assertEqual(layer.pcd.point.positions.array.shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
