from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print(f"[check] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    python = sys.executable
    run([python, "cameras_viewer.py", "--help"])
    run([python, "cameras_calibrate.py", "--help"])
    run([python, "record_data.py", "--help"])
    run([python, "data_process/record_data_align.py", "--help"])
    run([python, "scripts/harness/verify_ffs_demo.py", "--help"])
    run([python, "scripts/harness/probe_d455_ir_pair.py", "--help"])
    run([python, "scripts/harness/probe_d455_stream_capability.py", "--help"])
    run([python, "scripts/harness/render_d455_stream_probe_report.py", "--help"])
    run([python, "scripts/harness/visual_compare_depth_panels.py", "--help"])
    run([python, "scripts/harness/visual_compare_reprojection.py", "--help"])
    run([python, "scripts/harness/visual_compare_depth_video.py", "--help"])
    run([python, "scripts/harness/visual_compare_turntable.py", "--help"])
    run([python, "scripts/harness/run_ffs_on_saved_pair.py", "--help"])
    run([python, "scripts/harness/reproject_ffs_to_color.py", "--help"])
    run([python, "-m", "scripts.harness.check_scope"])
    run([python, "-m", "unittest", "-v", "tests.test_record_data_align_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_ffs_intrinsic_file_format"])
    run([python, "-m", "unittest", "-v", "tests.test_ffs_reprojection_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_depth_quantization_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_depth_colormap_consistency_smoke"])
    run([python, "-m", "pytest", "tests/test_d455_probe_matrix_builder.py", "tests/test_d455_probe_result_schema.py"])
    run([python, "-m", "unittest", "-v", "tests.test_recording_metadata_schema_v2"])
    run([python, "-m", "unittest", "-v", "tests.test_depth_backend_contract_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_record_data_align_ffs_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_record_data_align_both_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_calibrate_loader_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_camera_pose_view_config_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_camera_frusta_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_object_centered_orbit_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_observed_hemisphere_orbit_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_auto_object_bbox_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_dual_render_output_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_dual_triple_output_planning_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_camera_overview_inset_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_support_render_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_full_360_unsupported_annotation_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_table_crop_roi_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_table_focus_center_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_manual_image_roi_filter_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_pointcloud_fusion_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_grid_2x3_label_layout_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_projection_mode_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_fallback_splat_render_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_single_frame_case_selection_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_turntable_view_generation_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_turntable_board_layout_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_keyframe_sheet_generation_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_visual_compare_depth_panels_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_visual_compare_reprojection_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_fused_cloud_render_config_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_visual_compare_depth_video_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_visual_compare_depth_video_grid_smoke"])
    run([python, "-m", "unittest", "-v", "tests.test_visual_compare_turntable_smoke"])
    print("[check] all deterministic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
