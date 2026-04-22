from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]

QUICK_HELP_SCRIPTS: tuple[str, ...] = (
    "cameras_viewer.py",
    "cameras_viewer_FFS.py",
    "cameras_calibrate.py",
    "record_data.py",
    "data_process/record_data_align.py",
    "scripts/harness/benchmark_ffs_configs.py",
    "scripts/harness/run_ffs_static_replay_matrix.py",
    "scripts/harness/visualize_ffs_static_confidence_panels.py",
    "scripts/harness/visual_compare_depth_panels.py",
    "scripts/harness/visual_compare_reprojection.py",
    "scripts/harness/visual_compare_turntable.py",
    "scripts/harness/cleanup_different_types_cases.py",
)

FULL_HELP_SCRIPTS: tuple[str, ...] = (
    "cameras_viewer.py",
    "cameras_viewer_FFS.py",
    "cameras_calibrate.py",
    "record_data.py",
    "data_process/record_data_align.py",
    "scripts/harness/verify_ffs_demo.py",
    "scripts/harness/verify_ffs_tensorrt_wsl.py",
    "scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py",
    "scripts/harness/probe_d455_ir_pair.py",
    "scripts/harness/probe_d455_stream_capability.py",
    "scripts/harness/render_d455_stream_probe_report.py",
    "scripts/harness/generate_sam31_masks.py",
    "scripts/harness/visual_compare_depth_panels.py",
    "scripts/harness/diagnose_floating_point_sources.py",
    "scripts/harness/benchmark_ffs_configs.py",
    "scripts/harness/run_ffs_static_replay_matrix.py",
    "scripts/harness/visualize_ffs_static_confidence_panels.py",
    "scripts/harness/visual_compare_masked_pointcloud.py",
    "scripts/harness/visual_compare_masked_camera_views.py",
    "scripts/harness/visual_compare_reprojection.py",
    "scripts/harness/visual_compare_depth_video.py",
    "scripts/harness/visual_compare_depth_triplet_ply.py",
    "scripts/harness/visual_compare_depth_triplet_video.py",
    "scripts/harness/visual_compare_rerun.py",
    "scripts/harness/visual_compare_turntable.py",
    "scripts/harness/visual_compare_stereo_order_pcd.py",
    "scripts/harness/visual_make_professor_triptych.py",
    "scripts/harness/visual_make_match_board.py",
    "scripts/harness/audit_ffs_left_right.py",
    "scripts/harness/compare_face_smoothness.py",
    "scripts/harness/run_ffs_on_saved_pair.py",
    "scripts/harness/reproject_ffs_to_color.py",
    "scripts/harness/cleanup_different_types_cases.py",
)

QUICK_UNITTEST_BATCHES: tuple[tuple[str, ...], ...] = (
    (
        "tests.test_agents_scope_contract_smoke",
        "tests.test_recording_metadata_schema_v2",
        "tests.test_record_preflight_policy_smoke",
        "tests.test_record_data_preflight_message_smoke",
        "tests.test_calibrate_loader_smoke",
    ),
    (
        "tests.test_record_data_align_smoke",
        "tests.test_cameras_viewer_ffs_smoke",
        "tests.test_depth_backend_contract_smoke",
        "tests.test_ffs_intrinsic_file_format",
        "tests.test_ffs_reprojection_smoke",
        "tests.test_ffs_remove_invisible_mask_smoke",
        "tests.test_ffs_tensorrt_single_engine_smoke",
        "tests.test_ffs_confidence_panels_smoke",
        "tests.test_ffs_static_replay_matrix_smoke",
    ),
    (
        "tests.test_visual_compare_depth_panels_smoke",
        "tests.test_visual_compare_reprojection_smoke",
        "tests.test_visual_compare_turntable_smoke",
    ),
)

FULL_UNITTEST_MODULES: tuple[str, ...] = (
    "tests.test_agents_scope_contract_smoke",
    "tests.test_cameras_viewer_fps_smoke",
    "tests.test_cameras_viewer_ffs_smoke",
    "tests.test_record_data_align_smoke",
    "tests.test_ffs_intrinsic_file_format",
    "tests.test_ffs_reprojection_smoke",
    "tests.test_depth_quantization_smoke",
    "tests.test_depth_colormap_consistency_smoke",
    "tests.test_depth_panel_layout_smoke",
    "tests.test_depth_panel_roi_overlay_smoke",
    "tests.test_depth_panel_preset_smoke",
    "tests.test_floating_point_diagnostics_smoke",
    "tests.test_ffs_benchmarking_smoke",
    "tests.test_masked_pointcloud_compare_smoke",
    "tests.test_original_camera_view_config_smoke",
    "tests.test_masked_camera_view_compare_smoke",
    "tests.test_sam31_mask_helper_smoke",
    "tests.test_recording_metadata_schema_v2",
    "tests.test_camera_system_partial_stall_smoke",
    "tests.test_single_realsense_recovery_smoke",
    "tests.test_record_preflight_policy_smoke",
    "tests.test_record_data_preflight_message_smoke",
    "tests.test_depth_backend_contract_smoke",
    "tests.test_ffs_remove_invisible_mask_smoke",
    "tests.test_ffs_native_like_depth_postprocess_smoke",
    "tests.test_ffs_radius_outlier_filter_smoke",
    "tests.test_ffs_tensorrt_single_engine_smoke",
    "tests.test_ffs_confidence_panels_smoke",
    "tests.test_ffs_static_replay_matrix_smoke",
    "tests.test_record_data_align_ffs_smoke",
    "tests.test_record_data_align_both_smoke",
    "tests.test_calibrate_loader_smoke",
    "tests.test_calibration_contract_hardening",
    "tests.test_camera_pose_view_config_smoke",
    "tests.test_camera_frusta_smoke",
    "tests.test_object_centered_orbit_smoke",
    "tests.test_observed_hemisphere_orbit_smoke",
    "tests.test_auto_object_bbox_smoke",
    "tests.test_dual_render_output_smoke",
    "tests.test_dual_triple_output_planning_smoke",
    "tests.test_camera_overview_inset_smoke",
    "tests.test_support_render_smoke",
    "tests.test_full_360_unsupported_annotation_smoke",
    "tests.test_table_crop_roi_smoke",
    "tests.test_table_focus_center_smoke",
    "tests.test_manual_image_roi_filter_smoke",
    "tests.test_auto_object_refinement_from_projected_bbox_smoke",
    "tests.test_graph_union_preserves_protrusion_smoke",
    "tests.test_pixel_mask_expands_world_roi_smoke",
    "tests.test_compare_debug_metrics_refinement_smoke",
    "tests.test_object_compare_contract_smoke",
    "tests.test_source_id_propagation_smoke",
    "tests.test_source_attribution_overlay_smoke",
    "tests.test_source_split_render_smoke",
    "tests.test_mismatch_residual_smoke",
    "tests.test_source_legend_smoke",
    "tests.test_visual_import_graph_smoke",
    "tests.test_visual_types_contract_smoke",
    "tests.test_selection_contracts_smoke",
    "tests.test_artifact_writer_smoke",
    "tests.test_layout_builder_smoke",
    "tests.test_merge_diagnostics_workflow_smoke",
    "tests.test_turntable_workflow_smoke",
    "tests.test_turntable_frame_contract_smoke",
    "tests.test_object_union_bbox_smoke",
    "tests.test_object_first_sampling_smoke",
    "tests.test_object_debug_artifacts_smoke",
    "tests.test_pointcloud_fusion_smoke",
    "tests.test_aligned_metadata_loader_smoke",
    "tests.test_io_case_ffs_native_like_loader_smoke",
    "tests.test_io_case_ffs_raw_loader_smoke",
    "tests.test_grid_2x3_label_layout_smoke",
    "tests.test_projection_mode_smoke",
    "tests.test_fallback_splat_render_smoke",
    "tests.test_grouped_aligned_case_resolution_smoke",
    "tests.test_single_frame_case_selection_smoke",
    "tests.test_turntable_view_generation_smoke",
    "tests.test_turntable_board_layout_smoke",
    "tests.test_keyframe_sheet_generation_smoke",
    "tests.test_professor_triptych_angle_selection_smoke",
    "tests.test_professor_triptych_output_contract_smoke",
    "tests.test_match_board_angle_selection_smoke",
    "tests.test_match_board_output_contract_smoke",
    "tests.test_match_board_layout_smoke",
    "tests.test_match_board_legend_smoke",
    "tests.test_left_right_audit_stats_generation",
    "tests.test_raw_clipped_disparity_audit_path",
    "tests.test_face_patch_json_parsing",
    "tests.test_plane_fit_metric_computation",
    "tests.test_face_quality_board_layout",
    "tests.test_semantic_world_inference_smoke",
    "tests.test_stereo_order_registration_layout_smoke",
    "tests.test_stereo_order_registration_source_color_smoke",
    "tests.test_stereo_order_registration_workflow_smoke",
    "tests.test_visual_compare_depth_panels_smoke",
    "tests.test_diagnose_floating_point_sources_smoke",
    "tests.test_visual_compare_reprojection_smoke",
    "tests.test_fused_cloud_render_config_smoke",
    "tests.test_visual_compare_depth_video_smoke",
    "tests.test_visual_compare_depth_video_grid_smoke",
    "tests.test_triplet_ply_compare_workflow_smoke",
    "tests.test_triplet_video_compare_workflow_smoke",
    "tests.test_rerun_compare_workflow_smoke",
    "tests.test_visual_compare_turntable_smoke",
    "tests.test_visual_make_professor_triptych_smoke",
    "tests.test_visual_make_match_board_smoke",
    "tests.test_cleanup_different_types_cases_smoke",
    "tests.test_check_all_smoke",
)

PYTEST_BATCHES: tuple[tuple[str, ...], ...] = (
    (
        "tests/test_d455_probe_matrix_builder.py",
        "tests/test_d455_probe_result_schema.py",
    ),
)

CHECK_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("scripts/harness/check_visual_architecture.py",),
    ("-m", "scripts.harness.check_scope"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic repo validation. Default is the fast quick profile; "
            "pass --full for the broader legacy validation set."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="Validation profile to run. Defaults to quick.",
    )
    parser.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="profile",
        help="Shortcut for --profile full.",
    )
    return parser.parse_args(argv)


def run(cmd: list[str]) -> None:
    print(f"[check] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _help_commands(*, python: str, scripts: tuple[str, ...]) -> list[list[str]]:
    return [[python, script, "--help"] for script in scripts]


def _check_commands(*, python: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for command in CHECK_COMMANDS:
        if command[0] == "-m":
            commands.append([python, *command])
        else:
            commands.append([python, command[0]])
    return commands


def _unittest_commands(*, python: str, module_batches: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    return [[python, "-m", "unittest", "-v", *modules] for modules in module_batches]


def _pytest_commands(*, python: str) -> list[list[str]]:
    return [[python, "-m", "pytest", *batch] for batch in PYTEST_BATCHES]


def build_commands(*, python: str, profile: str) -> list[list[str]]:
    if profile == "quick":
        return [
            *_help_commands(python=python, scripts=QUICK_HELP_SCRIPTS),
            *_check_commands(python=python),
            *_unittest_commands(python=python, module_batches=QUICK_UNITTEST_BATCHES),
        ]
    if profile == "full":
        full_unittest_batches = tuple((module,) for module in FULL_UNITTEST_MODULES)
        return [
            *_help_commands(python=python, scripts=FULL_HELP_SCRIPTS),
            *_check_commands(python=python),
            *_unittest_commands(python=python, module_batches=full_unittest_batches),
            *_pytest_commands(python=python),
        ]
    raise ValueError(f"Unsupported profile: {profile}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    python = sys.executable
    print(f"[check] profile={args.profile}")
    for cmd in build_commands(python=python, profile=args.profile):
        run(cmd)
    print(f"[check] {args.profile} deterministic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
