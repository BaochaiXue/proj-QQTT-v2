from __future__ import annotations

import unittest

import numpy as np

from demo_v2_1 import realtime_three_view_masked_fused_pcd as demo


class DemoV21ThreeViewFusedPcdSmoke(unittest.TestCase):
    def test_default_semantic_postprocess_policy(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_CONTROLLER_OBJECT,
            object_label="stuffed animal",
            controller_label="controller",
        )
        by_label = {layer.label: layer for layer in layers}
        self.assertEqual(by_label["stuffed animal"].default_postprocess, demo.POSTPROCESS_ENHANCED_PT)
        self.assertEqual(by_label["controller"].default_postprocess, demo.POSTPROCESS_PT_FILTER)

    def test_object_only_has_no_controller_layer(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_OBJECT_ONLY,
            object_label="stuffed animal",
            controller_label="controller",
        )
        self.assertEqual([layer.label for layer in layers], ["stuffed animal"])

    def test_fusion_keeps_object_and_controller_separate(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_CONTROLLER_OBJECT,
            object_label="stuffed animal",
            controller_label="controller",
        )
        clouds = [
            demo.CameraLayerCloud(
                camera_idx=0,
                label="stuffed animal",
                points_m=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                colors_rgb=np.array([[255, 0, 0]], dtype=np.uint8),
            ),
            demo.CameraLayerCloud(
                camera_idx=1,
                label="stuffed animal",
                points_m=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
                colors_rgb=np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            ),
            demo.CameraLayerCloud(
                camera_idx=2,
                label="controller",
                points_m=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                colors_rgb=np.array([[255, 255, 0]], dtype=np.uint8),
            ),
        ]
        fused = demo.fuse_semantic_camera_clouds(clouds, layers)

        self.assertEqual(fused["stuffed animal"].point_count, 3)
        self.assertEqual(fused["controller"].point_count, 1)
        self.assertEqual(fused["stuffed animal"].postprocess_mode, demo.POSTPROCESS_ENHANCED_PT)
        self.assertEqual(fused["controller"].postprocess_mode, demo.POSTPROCESS_PT_FILTER)

    def test_dry_run_contract_records_official_quality_policy(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--track-mode", "controller-object"])
        contract = demo.build_contract(args)
        self.assertTrue(contract["frame_by_frame_streaming"])
        self.assertFalse(contract["offline_video_input_used"])
        self.assertTrue(contract["official_quality_depth"])
        self.assertEqual(contract["compile_mode"], "vision-reduce-overhead")
        self.assertTrue(contract["fusion"]["labels_are_filtered_separately"])


if __name__ == "__main__":
    unittest.main()
