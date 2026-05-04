# Generated Docs Index

This folder keeps generated validation notes, inventories, and small reusable JSON assets.

Current grouping:

## Hardware / Capture Validation

- `d455_single_probe_validation.md`
- `d455_stream_probe_results.json`
- `d455_stream_probe_results.md`
- `both_eval_30_frame_validation.md`
- `wsl_realsense_rsusb_validation.md`

These record what the current machine and D455 topology actually supported during probe or capture checks.

## FFS / SAM-family Integration Validation

- `ffs_demo_validation.md`
- `wsl_env_bootstrap_validation.md`
- `ffs_tensorrt_windows_validation.md`
- `ffs_tensorrt_wsl_validation.md`
- `ffs_official_twostage_triton_env_validation.md`
- `ffs_single_engine_tensorrt_wsl_validation.md`
- `ffs_batch3_viewer_validation.md`
- `ffs_benchmark_tradeoff_validation.md`
- `ffs_static_replay_matrix_validation.md`
- `ffs_static_replay_matrix_concurrent3view_validation.md`
- `ffs_live_3cam_benchmark_validation.md`
- `ffs_live_vs_proxy_boundary.md`
- `ffs_live_trt_viewer_validation.md`
- `ffs_depth_backend_integration_validation.md`
- `ffs_comparison_workflow_validation.md`
- `ffs_left_right_codepath_audit.md`
- `ffs_max_sam31_realsense_env_validation.md`
- `demo_2_max_env_validation.md`
- `sam21_max_env_validation.md`
- `sam21_max_round2_benchmark.md`
- `sam21_dynamics_checkpoint_ladder_benchmark.md`
- `edgetam_max_env_validation.md`
- `edgetam_dynamics_round1_3x6_panel_benchmark.md`
- `edgetam_vs_sam21_speed_ablation.md`
- `sam21_edgetam_mask_overlay_3x3_benchmark.md`
- `sloth_base_motion_ffs_mask_overlay_3x3_benchmark.md`
- `sloth_base_motion_ffs_fused_pcd_overlay_2x3_benchmark.md`
- `edgetam_onnx_trt_probe.md`
- `edgetam_video_trt_compile_probe.md`
- `hf_edgetam_streaming_validation.md`
- `hf_edgetam_streaming_processor_session_validation.md`
- `hf_edgetam_streaming_realcase_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.md`
- `sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_enhanced_pt_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_enhanced_pt_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd_enhanced_pt_benchmark.md`
- `sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_benchmark.md`
- `sloth_set_2_motion_ffs_sam31_object_controller_pcd_benchmark.md`

These document external FFS and SAM-family proof-of-life, repo integration behavior, and targeted audits around the FFS code path.

## Visualization Validation

- `depth_visualization_validation.md`
- `sam31_env_validation.md`
- `visual_stack_cleanup_inventory.md`
- `visual_stack_cleanup_validation.md`

These are the main source of truth for current visualization workflows, ownership, and validation status.

`rerun_compare_validation.md` was removed after its content was absorbed by the broader visualization validation surface and current harness checks.

## Repo / Contract Hardening

- `repo_hardening_inventory.md`
- `repo_hardening_validation.md`
- `contract_hardening_validation.md`
- `results_retention_cleanup_20260501.md`
- `result_root_unification_20260501.md`
- `local_ephemeral_artifact_cleanup_20260503.md`

These record repo-structure, scope, and metadata-contract hardening passes.

## Reusable Assets

- `box_face_patches_static_frame_0000.json`
- `object_only_manual_image_roi_native_30_static_frame_0000.json`
- `object_only_manual_image_roi_fullhead_native_30_static_frame_0000.json`
- `sam21_max_still_object_video_benchmark.py`
- `sam21_max_still_object_video_benchmark_results.json`
- `sam21_max_round2_benchmark_results.json`
- `sam21_max_round2_mask_quality.json`
- `sam21_dynamics_checkpoint_ladder_results.json`
- `sam21_dynamics_checkpoint_ladder_mask_quality.json`
- `edgetam_dynamics_round1_3x6_panel_results.json`
- `edgetam_vs_sam21_speed_ablation.json`
- `sam21_edgetam_mask_overlay_3x3_results.json`
- `sloth_base_motion_ffs_mask_overlay_3x3_results.json`
- `sloth_base_motion_ffs_fused_pcd_overlay_2x3_results.json`
- `edgetam_onnx_trt_probe.json`
- `edgetam_video_trt_compile_probe.json`
- `hf_edgetam_streaming_results.json`
- `hf_edgetam_streaming_processor_session_results.json`
- `hf_edgetam_streaming_realcase_results.json`
- `hf_edgetam_streaming_quality.json`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_quality.json`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.json`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_vision_reduce_overhead_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_vision_reduce_overhead_vs_same_run_eager_quality.json`
- `sloth_set_2_motion_ffs_hf_edgetam_hand_object_streaming_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_enhanced_pt_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_streaming_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_enhanced_pt_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_streaming_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_controller_streaming_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd_enhanced_pt_results.json`
- `sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_results.json`
- `sloth_set_2_motion_ffs_sam31_object_controller_pcd_results.json`

These are not general datasets. They are small helper assets used by specific validation and visualization workflows.

## Naming Guidance

- `*_validation.md`: one concrete validation run or validation family
- `*_inventory.md`: current ownership / structure snapshot
- `*_results.*`: machine-readable and human-readable probe outputs
- `*.json` helper assets: only keep if still referenced by docs or scripts

When adding a new generated doc, prefer extending the existing validation file for that theme instead of creating another near-duplicate top-level note.
