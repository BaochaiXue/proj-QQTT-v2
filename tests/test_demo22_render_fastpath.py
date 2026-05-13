from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from demo_v2_2 import render_fastpath as fastpath
from demo_v2_2 import runtime as demo


class _FakeTensor:
    from_numpy_calls = 0

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> np.ndarray:
        cls.from_numpy_calls += 1
        return array


class _FakePointCloud:
    def __init__(self, _device: object) -> None:
        self.point = SimpleNamespace()


class _FakeSceneInner:
    UPDATE_POINTS_FLAG = 1
    UPDATE_COLORS_FLAG = 2

    def __init__(self) -> None:
        self.update_calls = 0

    def update_geometry(self, _name: str, _pcd: object, _flags: int) -> None:
        self.update_calls += 1


class _FakeScene:
    def __init__(self) -> None:
        self.scene = _FakeSceneInner()
        self.add_calls = 0
        self.remove_calls = 0

    def add_geometry(self, _name: str, _pcd: object, _material: object) -> None:
        self.add_calls += 1

    def remove_geometry(self, _name: str) -> None:
        self.remove_calls += 1


class Demo22RenderFastpathTest(unittest.TestCase):
    def test_latest_only_render_buffer_replaces_pending_without_backpressure(self) -> None:
        buffer: fastpath.LatestOnlyRenderBuffer[int] = fastpath.LatestOnlyRenderBuffer()

        buffer.publish(1)
        buffer.publish(2)

        self.assertEqual(buffer.take_latest(), 2)
        self.assertIsNone(buffer.take_latest())
        self.assertEqual(
            buffer.snapshot(),
            {
                "published": 2,
                "taken": 1,
                "displayed": 1,
                "dropped": 1,
                "pending": 0,
                "backpressure_count": 0,
            },
        )

    def test_coalesced_render_post_gate_allows_only_one_pending_callback(self) -> None:
        gate = fastpath.CoalescedRenderPostGate()

        self.assertTrue(gate.try_mark_pending())
        self.assertFalse(gate.try_mark_pending())
        self.assertEqual(gate.snapshot()["posted"], 1)
        self.assertEqual(gate.snapshot()["coalesced"], 1)

        gate.mark_done()
        self.assertTrue(gate.try_mark_pending())
        self.assertEqual(gate.snapshot()["posted"], 2)

    def test_render_micro_profile_summary_reports_split_timings(self) -> None:
        records = [
            {
                "render_packet_id": 1,
                "points_count": 10,
                "gpu_to_cpu_copy_ms": 0.0,
                "cpu_format_ms": 1.0,
                "open3d_update_geometry_ms": 2.0,
                "open3d_poll_events_ms": 0.5,
                "open3d_update_renderer_ms": 0.5,
                "render_total_ms": 4.0,
                "backpressure": False,
            },
            {
                "render_packet_id": 2,
                "points_count": 20,
                "gpu_to_cpu_copy_ms": 0.0,
                "cpu_format_ms": 3.0,
                "open3d_update_geometry_ms": 6.0,
                "open3d_poll_events_ms": 1.0,
                "open3d_update_renderer_ms": 1.0,
                "render_total_ms": 12.0,
                "backpressure": False,
            },
        ]

        summary = fastpath.summarize_render_records(records)

        self.assertEqual(summary["render_packets_displayed"], 2)
        self.assertEqual(summary["render_backpressure_count"], 0)
        self.assertEqual(summary["metrics"]["render_total_ms"]["p50"], 8.0)
        self.assertEqual(summary["metrics"]["render_open3d_update_ms"]["p90"], 5.6)
        self.assertEqual(summary["metrics"]["render_points_count"]["p50"], 15.0)

    def test_legacy_inplace_layer_reuses_tensors_when_point_count_is_stable(self) -> None:
        _FakeTensor.from_numpy_calls = 0
        scene = _FakeScene()
        layer = fastpath.Open3DSceneTensorLayer(
            name="pcd",
            o3d_module=SimpleNamespace(t=SimpleNamespace(geometry=SimpleNamespace(PointCloud=_FakePointCloud))),
            o3c_module=SimpleNamespace(Tensor=_FakeTensor),
            rendering_module=SimpleNamespace(Scene=_FakeSceneInner),
            scene=scene,
            material=object(),
            device=object(),
            backend=fastpath.RENDER_BACKEND_LEGACY_INPLACE,
        )
        points = np.ones((2, 3), dtype=np.float32)
        colors = np.full((2, 3), 255, dtype=np.uint8)

        first = layer.update(points, colors)
        second = layer.update(points + 1.0, colors)

        self.assertTrue(first.tensor_rebound)
        self.assertFalse(second.tensor_rebound)
        self.assertEqual(_FakeTensor.from_numpy_calls, 2)
        self.assertEqual(scene.add_calls, 1)
        self.assertEqual(scene.scene.update_calls, 1)

    def test_legacy_inplace_layer_readds_geometry_when_point_count_changes(self) -> None:
        _FakeTensor.from_numpy_calls = 0
        scene = _FakeScene()
        layer = fastpath.Open3DSceneTensorLayer(
            name="pcd",
            o3d_module=SimpleNamespace(t=SimpleNamespace(geometry=SimpleNamespace(PointCloud=_FakePointCloud))),
            o3c_module=SimpleNamespace(Tensor=_FakeTensor),
            rendering_module=SimpleNamespace(Scene=_FakeSceneInner),
            scene=scene,
            material=object(),
            device=object(),
            backend=fastpath.RENDER_BACKEND_LEGACY_INPLACE,
        )

        first = layer.update(
            np.ones((2, 3), dtype=np.float32),
            np.full((2, 3), 255, dtype=np.uint8),
        )
        second = layer.update(
            np.ones((3, 3), dtype=np.float32),
            np.full((3, 3), 255, dtype=np.uint8),
        )

        self.assertTrue(first.geometry_recreated)
        self.assertTrue(second.geometry_recreated)
        self.assertEqual(scene.add_calls, 2)
        self.assertEqual(scene.remove_calls, 1)
        self.assertEqual(scene.scene.update_calls, 0)

    def test_render_layer_combiner_preserves_all_points_and_colors(self) -> None:
        combiner = fastpath.RenderLayerCombiner()
        object_points = np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]], dtype=np.float32)
        object_colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        controller_points = np.array([[0.0, 0.1, 0.5]], dtype=np.float32)
        controller_colors = np.array([[0, 0, 255]], dtype=np.uint8)

        points, colors, combine_ms = combiner.combine(
            ((object_points, object_colors), (controller_points, controller_colors))
        )

        np.testing.assert_array_equal(points, np.concatenate([object_points, controller_points], axis=0))
        np.testing.assert_array_equal(colors, np.concatenate([object_colors, controller_colors], axis=0))
        self.assertGreaterEqual(combine_ms, 0.0)

    def test_demo22_contract_records_no_quality_loss_render_fastpath(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO22_ASYNC_FILTER_5FPS,
                "--render-backend",
                fastpath.RENDER_BACKEND_LEGACY_INPLACE,
                "--render-copy-mode",
                fastpath.RENDER_COPY_MODE_ASYNC_PINNED,
                "--render-micro-profile",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--render-backend", "--render-copy-mode", "--render-micro-profile"},
        )
        contract = demo.build_contract(args)

        self.assertEqual(contract["renderer"]["backend"], fastpath.RENDER_BACKEND_LEGACY_INPLACE)
        self.assertEqual(contract["renderer"]["layer_mode"], fastpath.RENDER_LAYER_MODE_COMBINED)
        self.assertTrue(contract["renderer"]["async_latest_only"])
        self.assertEqual(contract["renderer"]["copy_mode"], fastpath.RENDER_COPY_MODE_ASYNC_PINNED)
        self.assertTrue(contract["renderer"]["micro_profile"])
        self.assertFalse(contract["renderer"]["quality_loss_default"])


if __name__ == "__main__":
    unittest.main()
