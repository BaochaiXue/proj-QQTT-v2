from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.services.semantic_fusion_service import (
    CameraSemanticFrame,
    SemanticFusionConfig,
    SemanticFusionService,
    compare_fused_semantic_pcds,
)
from qqtt.demo.services.service_types import CameraIntrinsics


class SemanticFusionServiceTests(unittest.TestCase):
    def _frame(
        self,
        *,
        camera_idx: int = 0,
        group_id: int = 10,
        c2w: np.ndarray | None = None,
    ) -> CameraSemanticFrame:
        rgb = np.array(
            [
                [[10, 0, 0], [20, 0, 0]],
                [[30, 0, 0], [40, 0, 0]],
            ],
            dtype=np.uint8,
        )
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        object_mask = np.array([[False, False], [True, False]], dtype=bool)
        controller_mask = np.array([[False, True], [False, False]], dtype=bool)
        return CameraSemanticFrame(
            camera_idx=int(camera_idx),
            group_id=int(group_id),
            timestamp_s=1.25,
            rgb=rgb,
            depth_m=depth,
            object_mask=object_mask,
            controller_mask=controller_mask,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            c2w=np.eye(4, dtype=np.float32) if c2w is None else c2w,
        )

    def test_backprojects_object_and_controller_points(self) -> None:
        service = SemanticFusionService(SemanticFusionConfig(depth_source="realsense"))

        fused = service.fuse({0: self._frame()})

        np.testing.assert_allclose(fused.object_xyz, np.array([[0.0, 3.0, 3.0]], dtype=np.float32))
        np.testing.assert_allclose(fused.controller_xyz, np.array([[2.0, 0.0, 2.0]], dtype=np.float32))
        np.testing.assert_array_equal(fused.object_rgb, np.array([[30, 0, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(fused.controller_rgb, np.array([[20, 0, 0]], dtype=np.uint8))
        self.assertEqual(fused.stats["fusion_impl"], "service-fast")
        self.assertEqual(fused.stats["per_camera_point_counts"][0], {"object": 1, "controller": 1})
        self.assertIn("fusion_backproject_ms", fused.stats)
        self.assertEqual(service.snapshot()["intrinsics_grid_cache_entries"], 1)

    def test_applies_camera_to_world_transform(self) -> None:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([10.0, -1.0, 0.5], dtype=np.float32)
        service = SemanticFusionService(SemanticFusionConfig(depth_source="ffs"))

        fused = service.fuse({0: self._frame(c2w=c2w)})

        np.testing.assert_allclose(fused.object_xyz, np.array([[10.0, 2.0, 3.5]], dtype=np.float32))
        np.testing.assert_allclose(fused.controller_xyz, np.array([[12.0, -1.0, 2.5]], dtype=np.float32))
        self.assertEqual(fused.stats["depth_source"], "ffs")

    def test_rejects_mismatched_group_ids(self) -> None:
        service = SemanticFusionService(SemanticFusionConfig(depth_source="realsense"))

        with self.assertRaises(ValueError):
            service.fuse({0: self._frame(camera_idx=0, group_id=1), 1: self._frame(camera_idx=1, group_id=2)})

    def test_debug_per_camera_xyz_is_opt_in(self) -> None:
        service = SemanticFusionService(
            SemanticFusionConfig(depth_source="realsense", debug_per_camera_colors=True)
        )

        fused = service.fuse({0: self._frame()})

        self.assertIn(0, fused.camera_debug_xyz)
        self.assertEqual(fused.camera_debug_xyz[0].shape, (2, 3))
        self.assertGreaterEqual(fused.stats["fusion_debug_ms"], 0.0)

    def test_quality_comparison_reports_exact_match(self) -> None:
        service = SemanticFusionService(SemanticFusionConfig(depth_source="realsense"))
        fused = service.fuse({0: self._frame()})

        summary = compare_fused_semantic_pcds(fused, fused)

        self.assertTrue(summary["fusion_quality_guard_enabled"])
        self.assertEqual(summary["fusion_object_voxel_iou_5mm"], 1.0)
        self.assertEqual(summary["fusion_controller_voxel_iou_5mm"], 1.0)
        self.assertEqual(summary["fusion_object_bbox_delta_mm"], 0.0)
        self.assertEqual(summary["fusion_controller_centroid_delta_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
