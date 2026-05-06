from __future__ import annotations

import unittest

import numpy as np

from demo_v2_1 import realtime_three_view_masked_fused_pcd as demo


class DemoV21ThreeViewFusedPcdSmoke(unittest.TestCase):
    def test_default_semantic_postprocess_policy(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_CONTROLLER_OBJECT,
            object_label="stuffed animal",
            controller_label="hand",
        )
        by_label = {layer.label: layer for layer in layers}
        self.assertEqual(by_label["stuffed animal"].default_postprocess, demo.POSTPROCESS_ENHANCED_PT)
        self.assertEqual(by_label["hand"].default_postprocess, demo.POSTPROCESS_PT_FILTER)

    def test_controller_slot_uses_pt_filter_for_hand_prompt(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_CONTROLLER_OBJECT,
            object_label="stuffed animal",
            controller_label="hand",
        )
        by_id = {layer.obj_id: layer for layer in layers}
        self.assertEqual(by_id[demo.CONTROLLER_ID].label, "hand")
        self.assertEqual(by_id[demo.CONTROLLER_ID].default_postprocess, demo.POSTPROCESS_PT_FILTER)
        self.assertEqual(by_id[demo.OBJECT_ID].default_postprocess, demo.POSTPROCESS_ENHANCED_PT)

    def test_object_only_has_no_controller_layer(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_OBJECT_ONLY,
            object_label="stuffed animal",
            controller_label="hand",
        )
        self.assertEqual([layer.label for layer in layers], ["stuffed animal"])

    def test_fusion_keeps_object_and_controller_separate(self) -> None:
        layers = demo.semantic_layers_for_track_mode(
            demo.TRACK_MODE_CONTROLLER_OBJECT,
            object_label="stuffed animal",
            controller_label="hand",
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
                label="hand",
                points_m=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                colors_rgb=np.array([[255, 255, 0]], dtype=np.uint8),
            ),
        ]
        fused = demo.fuse_semantic_camera_clouds(clouds, layers)

        self.assertEqual(fused["stuffed animal"].point_count, 3)
        self.assertEqual(fused["hand"].point_count, 1)
        self.assertEqual(fused["stuffed animal"].postprocess_mode, demo.POSTPROCESS_ENHANCED_PT)
        self.assertEqual(fused["hand"].postprocess_mode, demo.POSTPROCESS_PT_FILTER)

    def test_semantic_postprocess_caps_before_cleanup(self) -> None:
        layer = demo.FusedLayerCloud(
            label="stuffed animal",
            postprocess_mode=demo.POSTPROCESS_NONE,
            points_m=np.array(
                [
                    [0.00, 0.00, 0.50],
                    [0.01, 0.00, 0.50],
                    [0.20, 0.00, 0.50],
                ],
                dtype=np.float32,
            ),
            colors_rgb=np.arange(9, dtype=np.uint8).reshape(3, 3),
            per_camera=(),
        )

        points, colors, stats = demo.apply_semantic_postprocess(
            layer,
            filter_cap=2,
            filter_voxel_size_m=0.10,
            phystwin_radius_m=0.01,
            phystwin_nb_points=1,
            enhanced_component_voxel_size_m=0.01,
            enhanced_keep_near_main_gap_m=0.0,
        )

        self.assertLessEqual(points.shape[0], 2)
        self.assertEqual(colors.shape, (points.shape[0], 3))
        self.assertEqual(stats["input_point_count"], 3)
        self.assertLessEqual(stats["capped_point_count"], 2)

    def test_dry_run_contract_records_official_quality_policy(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--track-mode", "controller-object", "--enable-pcd-filter"])
        args = demo.apply_preset_defaults(args, explicit_options=set())
        contract = demo.build_contract(args)
        self.assertTrue(contract["frame_by_frame_streaming"])
        self.assertFalse(contract["offline_video_input_used"])
        self.assertTrue(contract["official_quality_depth"])
        self.assertEqual(contract["compile_mode"], "vision-reduce-overhead")
        self.assertTrue(contract["fusion"]["labels_are_filtered_separately"])
        self.assertTrue(contract["filter_scheduler"]["enabled"])
        self.assertEqual(contract["filter_scheduler"]["mode"], "async")
        self.assertEqual(contract["filter_scheduler"]["object"]["postprocess"], demo.POSTPROCESS_ENHANCED_PT)
        self.assertEqual(contract["filter_scheduler"]["controller"]["postprocess"], demo.POSTPROCESS_PT_FILTER)
        self.assertFalse(contract["fusion"]["object_controller_union_before_filter"])
        self.assertEqual(contract["ffs_contract"]["worker_mode"], "shared")
        self.assertEqual(contract["edgetam"]["worker_mode"], "per-camera")
        self.assertEqual(contract["gpu_gate"]["mode"], "serialized")
        self.assertEqual(contract["gpu_gate"]["max_concurrent"], 1)

    def test_professor_safe_preset_sets_low_fps_object_only_demo(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "professor-safe"])
        args = demo.apply_preset_defaults(args, explicit_options={"--preset", "--dry-run"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["preset"], "professor-safe")
        self.assertEqual(contract["profile"], "848x480")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["track_mode"], "object-only")
        self.assertEqual(contract["render_mode"], "pointcloud")
        self.assertEqual(contract["fusion_target_fps"], 2.0)
        self.assertEqual(contract["fusion_timeout_ms"], 250.0)
        self.assertEqual(contract["gpu_gate"], {"mode": "serialized", "max_concurrent": 1})
        self.assertEqual([layer["label"] for layer in contract["semantic_layers"]], ["stuffed animal"])

    def test_visual_5fps_preset_keeps_quality_path_with_gate2(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps"])
        args = demo.apply_preset_defaults(args, explicit_options={"--preset", "--dry-run"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["preset"], "visual-5fps")
        self.assertEqual(contract["profile"], "848x480")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["track_mode"], "object-only")
        self.assertEqual(contract["render_mode"], "pointcloud")
        self.assertEqual(contract["fusion_target_fps"], 5.0)
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["gpu_gate"], {"mode": "limited", "max_concurrent": 2})
        self.assertEqual(contract["semantic_layers"][0]["postprocess"], "enhanced-pt")

    def test_preset_keeps_explicit_track_and_render_overrides(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            ["--dry-run", "--preset", "professor-safe", "--track-mode", "object-only", "--render-mode", "none"]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--track-mode", "--render-mode"},
        )
        contract = demo.build_contract(args)

        self.assertEqual(contract["track_mode"], "object-only")
        self.assertEqual(contract["render_mode"], "none")
        self.assertEqual([layer["label"] for layer in contract["semantic_layers"]], ["stuffed animal"])

    def test_capture_only_isolation_contract_is_not_official_depth(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--depth-source", "none", "--track-mode", "none"])
        args = demo.apply_preset_defaults(args, explicit_options=set())
        contract = demo.build_contract(args)
        self.assertFalse(contract["official_quality_depth"])
        self.assertEqual(contract["semantic_layers"], [])

    def test_capture_group_timeout_is_skipped(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--depth-source",
                "none",
                "--track-mode",
                "none",
                "--fusion-target-fps",
                "1000",
            ]
        )
        runtime = demo.Demo21Runtime(args)

        class FakeCameraSystem:
            def __init__(self) -> None:
                self.calls = 0

            def get_observation(self) -> dict[int, object]:
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("simulated")
                runtime.stop_event.set()
                raise RuntimeError("stop")

        fake_camera = FakeCameraSystem()
        runtime.camera_system = fake_camera
        runtime._capture_group_worker()

        self.assertEqual(runtime._summary["capture_timeout_count"], 1)
        self.assertEqual(fake_camera.calls, 2)
        self.assertTrue(runtime.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
