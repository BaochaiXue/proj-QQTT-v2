from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.services.render_pcd_service import RenderPcdService, RenderServiceConfig
from qqtt.demo.services.service_types import RenderLayer, RenderPacket


class FakeRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render_packet(self, packet: RenderPacket) -> dict[str, float]:
        self.calls += 1
        return {
            "render_total_ms": 1.25,
            "open3d_update_geometry_ms": 0.5,
        }


def _packet(group_id: int) -> RenderPacket:
    points = np.zeros((group_id, 3), dtype=np.float32)
    colors = np.zeros((group_id, 3), dtype=np.uint8)
    return RenderPacket(
        group_id=group_id,
        timestamp_s=float(group_id),
        layers=(RenderLayer(name="object", points_xyz=points, colors_rgb=colors),),
    )


class RenderPcdServiceTests(unittest.TestCase):
    def test_submit_latest_drops_old_packets(self) -> None:
        service = RenderPcdService(RenderServiceConfig(window_title="unit"))
        service.submit_latest(_packet(1))
        service.submit_latest(_packet(2))

        self.assertEqual(service.buffer.snapshot()["dropped"], 1)
        rendered = service.render_once()

        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.group_id, 2)

    def test_render_once_records_profile(self) -> None:
        renderer = FakeRenderer()
        service = RenderPcdService(RenderServiceConfig(window_title="unit"), renderer=renderer)

        service.submit_latest(_packet(3))
        service.render_once()

        snapshot = service.snapshot()
        self.assertEqual(renderer.calls, 1)
        self.assertEqual(snapshot["render_micro_profile"]["render_packets_displayed"], 1)
        self.assertEqual(snapshot["render_buffer"]["displayed"], 1)


if __name__ == "__main__":
    unittest.main()
