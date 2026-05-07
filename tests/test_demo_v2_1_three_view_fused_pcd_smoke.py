from __future__ import annotations

from collections import deque
import unittest

import numpy as np

from demo_v2_1 import realtime_three_view_masked_fused_pcd as demo


def _dummy_frame(camera_idx: int, *, seq: int, timestamp_ms: float) -> demo.CameraFramePacket:
    return demo.CameraFramePacket(
        group_id=-1,
        camera_idx=int(camera_idx),
        frame_seq=int(seq),
        timestamp_ns=int(float(timestamp_ms) * 1_000_000),
        realsense_timestamp_ms=None,
        realsense_frame_number=None,
        timestamp_domain=None,
        capture_arrival_perf_ns=int(float(timestamp_ms) * 1_000_000),
        color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        ir_left_u8=np.zeros((2, 2), dtype=np.uint8),
        ir_right_u8=np.zeros((2, 2), dtype=np.uint8),
        k_color=np.eye(3, dtype=np.float32),
        k_ir_left=np.eye(3, dtype=np.float32),
        t_ir_left_to_color=np.eye(4, dtype=np.float32),
        baseline_m=0.055,
        intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        c2w=np.eye(4, dtype=np.float32),
    )


class DemoV21ThreeViewFusedPcdSmoke(unittest.TestCase):
    def test_timestamp_nearest_grouping_selects_min_skew_triplet(self) -> None:
        buffers = {
            0: deque([_dummy_frame(0, seq=0, timestamp_ms=0), _dummy_frame(0, seq=1, timestamp_ms=33), _dummy_frame(0, seq=2, timestamp_ms=66)]),
            1: deque([_dummy_frame(1, seq=0, timestamp_ms=5), _dummy_frame(1, seq=1, timestamp_ms=38), _dummy_frame(1, seq=2, timestamp_ms=71)]),
            2: deque([_dummy_frame(2, seq=0, timestamp_ms=7), _dummy_frame(2, seq=1, timestamp_ms=40), _dummy_frame(2, seq=2, timestamp_ms=73)]),
        }

        selection = demo.select_temporal_capture_triplet(
            buffers,
            camera_ids=(0, 1, 2),
            policy=demo.CAPTURE_GROUP_POLICY_TIMESTAMP_NEAREST,
            max_frame_age_ms=150.0,
            now_perf_ns=int(80 * 1_000_000),
        )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.per_camera_frame_seq, {0: 2, 1: 2, 2: 2})
        self.assertAlmostEqual(selection.max_temporal_skew_ms, 7.0)

    def test_timestamp_grouping_drops_when_skew_exceeds_threshold(self) -> None:
        frames = {
            0: _dummy_frame(0, seq=0, timestamp_ms=0),
            1: _dummy_frame(1, seq=0, timestamp_ms=40),
            2: _dummy_frame(2, seq=0, timestamp_ms=80),
        }
        selection = demo._temporal_selection_from_frames(frames, now_perf_ns=int(90 * 1_000_000))
        group = demo.build_temporal_capture_group(group_id=10, created_perf_s=1.0, selection=selection)

        self.assertAlmostEqual(group.max_temporal_skew_ms, 80.0)
        self.assertFalse(demo.temporal_group_is_coherent(group, max_capture_skew_ms=33.4))

    def test_capture_group_records_per_camera_offsets(self) -> None:
        frames = {
            0: _dummy_frame(0, seq=4, timestamp_ms=100),
            1: _dummy_frame(1, seq=5, timestamp_ms=110),
            2: _dummy_frame(2, seq=6, timestamp_ms=120),
        }
        group = demo.build_temporal_capture_group(
            group_id=3,
            created_perf_s=2.0,
            selection=demo._temporal_selection_from_frames(frames, now_perf_ns=int(130 * 1_000_000)),
        )

        self.assertEqual(group.per_camera_frame_seq, {0: 4, 1: 5, 2: 6})
        self.assertEqual(group.timestamp_source, "host_receive_timestamp")
        self.assertAlmostEqual(group.per_camera_time_offset_ms[0], -10.0)
        self.assertAlmostEqual(group.per_camera_time_offset_ms[1], 0.0)
        self.assertAlmostEqual(group.per_camera_time_offset_ms[2], 10.0)

    def test_ffs_and_fusion_skew_guards_use_same_temporal_contract(self) -> None:
        frames = {
            0: _dummy_frame(0, seq=0, timestamp_ms=0),
            1: _dummy_frame(1, seq=0, timestamp_ms=20),
            2: _dummy_frame(2, seq=0, timestamp_ms=70),
        }
        group = demo.build_temporal_capture_group(
            group_id=2,
            created_perf_s=1.0,
            selection=demo._temporal_selection_from_frames(frames, now_perf_ns=int(80 * 1_000_000)),
        )
        depth_group = demo.DepthGroup(
            group_id=group.group_id,
            depths={},
            total_ms=0.0,
            per_camera_ms={},
            gpu_gate_wait_ms=0.0,
            max_temporal_skew_ms=group.max_temporal_skew_ms,
            per_camera_time_offset_ms=group.per_camera_time_offset_ms,
            per_camera_frame_seq=group.per_camera_frame_seq,
            timestamp_source=group.timestamp_source,
        )

        self.assertFalse(demo.temporal_group_is_coherent(group, max_capture_skew_ms=33.4))
        self.assertFalse(demo.temporal_group_is_coherent(depth_group, max_capture_skew_ms=33.4))

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

    def test_controller_prompt_can_be_experimental_without_changing_default(self) -> None:
        parser = demo.build_arg_parser()
        default_args = parser.parse_args(["--dry-run"])
        self.assertEqual(default_args.controller_prompt, "hand")

        cloth_args = parser.parse_args(["--dry-run", "--track-mode", "controller-object", "--controller-prompt", "cloth"])
        cloth_args = demo.apply_preset_defaults(
            cloth_args,
            explicit_options={"--dry-run", "--track-mode", "--controller-prompt"},
        )
        contract = demo.build_contract(cloth_args)

        self.assertEqual(contract["semantic_layers"][0]["label"], "cloth")
        self.assertEqual(contract["semantic_layers"][0]["postprocess"], demo.POSTPROCESS_PT_FILTER)
        self.assertEqual(contract["semantic_layers"][1]["label"], "stuffed animal")
        self.assertEqual(contract["semantic_layers"][1]["postprocess"], demo.POSTPROCESS_ENHANCED_PT)

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
        self.assertEqual(contract["temporal_grouping"]["policy"], demo.CAPTURE_GROUP_POLICY_TIMESTAMP_NEAREST)
        self.assertEqual(contract["temporal_grouping"]["max_capture_skew_ms"], demo.DEFAULT_MAX_CAPTURE_SKEW_MS)
        self.assertTrue(contract["temporal_grouping"]["drop_skewed_groups"])
        self.assertTrue(contract["temporal_grouping"]["no_temporal_coherent_group_no_ffs"])

    def test_professor_safe_preset_defaults_to_controller_object_demo(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "professor-safe"])
        args = demo.apply_preset_defaults(args, explicit_options={"--preset", "--dry-run"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["preset"], "professor-safe")
        self.assertEqual(contract["profile"], "848x480")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["track_mode"], "controller-object")
        self.assertEqual(contract["render_mode"], "pointcloud")
        self.assertEqual(contract["fusion_target_fps"], 2.0)
        self.assertEqual(contract["fusion_timeout_ms"], 250.0)
        self.assertEqual(contract["gpu_gate"], {"mode": "serialized", "max_concurrent": 1})
        self.assertEqual(contract["temporal_grouping"]["policy"], "timestamp-nearest")
        self.assertEqual(contract["temporal_grouping"]["max_capture_skew_ms"], 33.4)
        self.assertEqual(contract["temporal_grouping"]["capture_buffer_size"], 4)
        self.assertEqual([layer["label"] for layer in contract["semantic_layers"]], ["hand", "stuffed animal"])

    def test_visual_5fps_preset_keeps_quality_path_with_gate2(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps", "--track-mode", "object-only"])
        args = demo.apply_preset_defaults(args, explicit_options={"--preset", "--dry-run", "--track-mode"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["preset"], "visual-5fps")
        self.assertEqual(contract["profile"], "848x480")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["track_mode"], "object-only")
        self.assertEqual(contract["render_mode"], "pointcloud")
        self.assertEqual(contract["fusion_target_fps"], 5.0)
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["gpu_gate"], {"mode": "limited", "max_concurrent": 2})
        self.assertEqual(contract["temporal_grouping"]["policy"], "timestamp-nearest")
        self.assertEqual(contract["temporal_grouping"]["max_capture_skew_ms"], 33.4)
        self.assertTrue(contract["temporal_grouping"]["drop_skewed_groups"])
        self.assertEqual(contract["semantic_layers"][0]["postprocess"], "enhanced-pt")

    def test_visual_5fps_no_gate_contract_keeps_quality_path(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps-no-gate", "--track-mode", "object-only"])
        args = demo.apply_preset_defaults(args, explicit_options={"--preset", "--dry-run", "--track-mode"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["preset"], "visual-5fps-no-gate")
        self.assertEqual(contract["profile"], "848x480")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["track_mode"], "object-only")
        self.assertEqual(contract["render_mode"], "pointcloud")
        self.assertEqual(contract["fusion_target_fps"], 5.0)
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["gpu_gate"], {"mode": "off", "max_concurrent": 0})
        self.assertEqual(contract["temporal_grouping"]["policy"], "timestamp-nearest")
        self.assertEqual(contract["temporal_grouping"]["max_capture_skew_ms"], 33.4)
        self.assertTrue(contract["temporal_grouping"]["drop_skewed_groups"])
        self.assertEqual(contract["ffs_contract"]["worker_mode"], "shared")
        self.assertEqual(contract["ffs_contract"]["schedule"], "strict3-latest")
        self.assertEqual(contract["ffs_contract"]["input_staging"], "pinned")
        self.assertEqual(contract["semantic_layers"][0]["postprocess"], "enhanced-pt")
        self.assertFalse(contract["fusion"]["object_controller_union_before_filter"])

    def test_visual_5fps_single_owner_contract_keeps_quality_path(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps-single-owner", "--track-mode", "object-only"])
        args = demo.apply_preset_defaults(args, explicit_options={"--preset", "--dry-run", "--track-mode"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["preset"], "visual-5fps-single-owner")
        self.assertEqual(contract["profile"], "848x480")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["track_mode"], "object-only")
        self.assertEqual(contract["render_mode"], "pointcloud")
        self.assertEqual(contract["fusion_target_fps"], 5.0)
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["gpu_pipeline"]["mode"], "single-owner")
        self.assertEqual(contract["gpu_pipeline"]["internal_order"], "ffs-then-edgetam")
        self.assertTrue(contract["gpu_pipeline"]["depth_and_masks_published_together"])
        self.assertFalse(contract["gpu_pipeline"]["separate_ffs_and_edgetam_workers"])
        self.assertEqual(contract["ffs_contract"]["worker_mode"], "shared")
        self.assertEqual(contract["ffs_contract"]["input_staging"], "pinned")
        self.assertEqual(contract["semantic_layers"][0]["postprocess"], "enhanced-pt")
        self.assertFalse(contract["fusion"]["object_controller_union_before_filter"])

    def test_single_owner_thread_specs_disable_separate_gpu_workers(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                "visual-5fps-single-owner",
                "--track-mode",
                "object-only",
                "--depth-source",
                "ffs",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--track-mode", "--depth-source"},
        )
        runtime = demo.Demo21Runtime(args)
        thread_names = [name for name, _target in runtime._thread_specs()]

        self.assertIn("capture-group", thread_names)
        self.assertIn("gpu-owner", thread_names)
        self.assertIn("fusion", thread_names)
        self.assertNotIn("shared-ffs", thread_names)
        self.assertFalse(any(name.startswith("edgetam-cam") for name in thread_names))

    def test_static_device_buffers_do_not_change_quality_contract(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                "visual-5fps-single-owner",
                "--static-device-buffers",
                "--preallocate-pcd-buffers",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--static-device-buffers", "--preallocate-pcd-buffers"},
        )
        contract = demo.build_contract(args)

        self.assertTrue(contract["memory_for_speed"]["static_device_buffers"])
        self.assertTrue(contract["memory_for_speed"]["preallocate_pcd_buffers"])
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["compile_mode"], demo.DEFAULT_COMPILE_MODE)
        self.assertEqual(contract["filter_scheduler"]["object"]["postprocess"], "enhanced-pt")

    def test_pin_memory_shorthand_resolves_to_all(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps", "--pin-memory"])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset", "--pin-memory"})
        contract = demo.build_contract(args)

        self.assertTrue(contract["h2d_transfer"]["pin_memory"])
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], "all")
        self.assertTrue(contract["h2d_transfer"]["edge_pin_enabled"])
        self.assertEqual(contract["h2d_transfer"]["ffs_input_staging"], "pinned")

    def test_true_no_pin_baseline_records_pageable_ffs_staging(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                "visual-5fps",
                "--profile-h2d",
                "--ffs-input-staging",
                "pageable",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--profile-h2d", "--ffs-input-staging"},
        )
        contract = demo.build_contract(args)

        self.assertFalse(contract["h2d_transfer"]["pin_memory"])
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], "off")
        self.assertEqual(contract["h2d_transfer"]["ffs_input_staging"], "pageable")
        self.assertEqual(contract["ffs_contract"]["input_staging"], "pageable")

    def test_pin_memory_mode_edge_preserves_explicit_pageable_ffs(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                "visual-5fps",
                "--pin-memory-mode",
                "edge",
                "--ffs-input-staging",
                "pageable",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--pin-memory-mode", "--ffs-input-staging"},
        )
        contract = demo.build_contract(args)

        self.assertTrue(contract["h2d_transfer"]["pin_memory"])
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], "edge")
        self.assertTrue(contract["h2d_transfer"]["edge_pin_enabled"])
        self.assertEqual(contract["h2d_transfer"]["ffs_input_staging"], "pageable")

    def test_pin_memory_mode_ffs_does_not_enable_edge_pinning(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps", "--pin-memory-mode", "ffs"])
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--pin-memory-mode"},
        )
        contract = demo.build_contract(args)

        self.assertTrue(contract["h2d_transfer"]["pin_memory"])
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], "ffs")
        self.assertFalse(contract["h2d_transfer"]["edge_pin_enabled"])
        self.assertEqual(contract["h2d_transfer"]["ffs_input_staging"], "pinned")

    def test_saved_mask_roots_are_recorded_in_contract(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                "visual-5fps",
                "--init-mode",
                "saved-masks",
                "--object-init-mask-root",
                "result/demo2_1/object_masks",
                "--controller-init-mask-root",
                "result/demo2_1/controller_masks",
            ]
        )
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["init"]["mode"], "saved-masks")
        self.assertEqual(contract["init"]["object_init_mask_root"], "result/demo2_1/object_masks")
        self.assertEqual(contract["init"]["controller_init_mask_root"], "result/demo2_1/controller_masks")
        self.assertTrue(contract["init"]["formal_demo_requires_live_sam31"])
        self.assertFalse(contract["init"]["fallback_allowed"])

    def test_live_sam31_defaults_are_fail_fast_for_formal_demo(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps", "--init-mode", "sam31-first-frame"])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset", "--init-mode"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["init"]["mode"], "sam31-first-frame")
        self.assertEqual(contract["init"]["sam31_retry_interval_s"], 0.5)
        self.assertEqual(contract["init"]["sam31_max_attempts"], 1)
        self.assertTrue(contract["init"]["formal_demo_requires_live_sam31"])
        self.assertFalse(contract["init"]["fallback_allowed"])

    def test_saved_masks_are_rejected_for_formal_demo2_1(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--depth-source",
                "none",
                "--init-mode",
                "saved-masks",
                "--track-mode",
                "object-only",
            ]
        )
        runtime = demo.Demo21Runtime(args)

        with self.assertRaisesRegex(RuntimeError, "live SAM3.1"):
            runtime._validate_live_contract()

    def test_worker_fatal_error_is_recorded_for_nonzero_runtime_exit(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--depth-source", "none", "--track-mode", "none"])
        runtime = demo.Demo21Runtime(args)

        runtime._mark_fatal_error("edgetam-cam0", RuntimeError("sam31 failed"))

        self.assertIn("edgetam-cam0", runtime._fatal_error or "")
        self.assertEqual(runtime._summary["fatal_error"], runtime._fatal_error)

    def test_visual_profile_flags_are_explicit_and_default_off(self) -> None:
        parser = demo.build_arg_parser()
        defaults = parser.parse_args(["--dry-run", "--preset", "visual-5fps"])
        self.assertFalse(defaults.profile_pipeline)
        self.assertFalse(defaults.profile_filter)
        self.assertFalse(defaults.profile_visualization)
        self.assertFalse(defaults.profile_gpu_gate)
        self.assertFalse(defaults.profile_h2d)
        self.assertIsNone(defaults.profile_json_output)

        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                "visual-5fps",
                "--profile-pipeline",
                "--profile-filter",
                "--profile-visualization",
                "--profile-gpu-gate",
                "--profile-warmup-exclude-s",
                "20",
                "--profile-json-output",
                "docs/generated/demo2_1_visual5fps_profile_object_only.json",
            ]
        )
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})

        self.assertTrue(args.profile_pipeline)
        self.assertTrue(args.profile_filter)
        self.assertTrue(args.profile_visualization)
        self.assertTrue(args.profile_gpu_gate)
        self.assertEqual(args.profile_warmup_exclude_s, 20)
        self.assertEqual(args.profile_json_output, "docs/generated/demo2_1_visual5fps_profile_object_only.json")

    def test_profile_summary_distinguishes_upstream_from_visualization(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", "visual-5fps", "--profile-pipeline"])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset", "--profile-pipeline"})
        runtime = demo.Demo21Runtime(args)

        upstream = [
            {"group_id": 0, "t_group_created": 0.0, "complete": True, "fusion": {"publish_s": 0.0}},
            {"group_id": 1, "t_group_created": 0.5, "complete": True, "fusion": {"publish_s": 0.5}},
        ]
        self.assertEqual(runtime._profile_summary_for_records(upstream)["bottleneck_class"], "upstream_supply")

        visual = [
            {
                "group_id": 0,
                "t_group_created": 0.0,
                "complete": True,
                "fusion": {"publish_s": 0.0},
                "render": {"render_s": 0.0},
            },
            {
                "group_id": 1,
                "t_group_created": 0.1,
                "complete": True,
                "fusion": {"publish_s": 0.1},
                "render": {"render_s": 0.5},
            },
        ]
        self.assertEqual(runtime._profile_summary_for_records(visual)["bottleneck_class"], "visualization")

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

    def test_pinned_pixel_stager_preserves_values(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency guard
            self.skipTest(f"torch unavailable: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        stager = demo.PinnedPixelValueStager(
            torch_module=torch,
            device="cuda",
            ring_size=1,
            h2d_stream_mode=demo.H2D_STREAM_MODE_DEDICATED,
            verify_copies=True,
        )
        source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        staged, profile = stager.stage(source, dtype=torch.float32)
        torch.cuda.current_stream().synchronize()

        self.assertTrue(torch.equal(staged.cpu(), source))
        self.assertTrue(profile["pin_memory"])
        self.assertEqual(profile["h2d_stream_mode"], demo.H2D_STREAM_MODE_DEDICATED)


if __name__ == "__main__":
    unittest.main()
