from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HelpProfile = Literal["quick", "full"]


@dataclass(frozen=True)
class HarnessEntry:
    path: str
    category: str
    summary: str
    help_profile: HelpProfile | None = None


CATALOG: tuple[HarnessEntry, ...] = (
    HarnessEntry(
        "scripts/harness/check_all.py",
        "checks",
        "Deterministic quick/full validation runner.",
    ),
    HarnessEntry(
        "scripts/harness/check_experiment_boundaries.py",
        "checks",
        "Guard formal runtime code from experiment-only imports.",
    ),
    HarnessEntry(
        "scripts/harness/check_harness_catalog.py",
        "checks",
        "Guard that every harness Python file is categorized here.",
    ),
    HarnessEntry(
        "scripts/harness/check_scope.py",
        "checks",
        "Repo scope guard for removed or forbidden legacy surfaces.",
    ),
    HarnessEntry(
        "scripts/harness/check_visual_architecture.py",
        "checks",
        "Visualization layering and file-size guard.",
    ),
    HarnessEntry(
        "scripts/harness/benchmark_ffs_configs.py",
        "hardware_external",
        "Saved-pair PyTorch FFS config screening; not live 3-camera realtime.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/benchmark_sam31_still_object_views.py",
        "hardware_external",
        "SAM 3.1 30-frame still-object per-camera segmentation benchmark.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/probe_d455_ir_pair.py",
        "hardware_external",
        "Manual D455 IR-pair probe.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/probe_d455_stream_capability.py",
        "hardware_external",
        "Manual D455 stream/profile capability probe.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/realtime_single_camera_pointcloud.py",
        "hardware_external",
        "Compatibility path for the active demo_v2 single-D455 realtime point-cloud demo.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/render_d455_stream_probe_report.py",
        "hardware_external",
        "Render D455 probe JSON as a readable report.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/run_ffs_on_saved_pair.py",
        "hardware_external",
        "Run FFS on one saved stereo pair.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/run_ffs_static_replay_matrix.py",
        "hardware_external",
        "Offline static replay / TensorRT proxy matrix; not live PyTorch realtime.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/run_wslg_open3d.sh",
        "hardware_external",
        "WSLg Open3D GUI wrapper.",
    ),
    HarnessEntry(
        "scripts/harness/verify_ffs_demo.py",
        "hardware_external",
        "External FFS demo proof-of-life utility.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py",
        "hardware_external",
        "WSL single-engine TensorRT proof-of-life utility.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/verify_ffs_tensorrt_windows.py",
        "hardware_external",
        "Windows TensorRT proof-of-life utility.",
    ),
    HarnessEntry(
        "scripts/harness/verify_ffs_tensorrt_wsl.py",
        "hardware_external",
        "WSL TensorRT proof-of-life utility.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/generate_sam31_masks.py",
        "mask_support",
        "Operator-side SAM 3.1 mask generation CLI.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/object_case_registry.py",
        "mask_support",
        "Shared raw object capture registry for harness scripts and tests.",
    ),
    HarnessEntry(
        "scripts/harness/reproject_ffs_to_color.py",
        "mask_support",
        "Reproject single-pair FFS depth into color-frame geometry.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/sam31_mask_helper.py",
        "mask_support",
        "Shared SAM 3.1 helper used by operator-side harness CLIs.",
    ),
    HarnessEntry(
        "scripts/harness/cleanup_different_types_cases.py",
        "formal_cleanup",
        "Dry-run or execute data/different_types cleanup.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_depth_panels.py",
        "current_compare",
        "Per-camera RealSense-vs-FFS depth panels.",
        "quick",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_depth_triplet_ply.py",
        "current_compare",
        "Single-frame native/FFS raw/FFS postprocess fused PLY compare.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_depth_triplet_video.py",
        "current_compare",
        "Multi-frame native/FFS raw/FFS postprocess point-cloud video compare.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_depth_video.py",
        "current_compare",
        "Older temporal fused native-vs-FFS depth compare.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_masked_camera_views.py",
        "current_compare",
        "SAM-masked native-vs-FFS camera-view board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_masked_pointcloud.py",
        "current_compare",
        "SAM-masked native-vs-FFS point-cloud board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_reprojection.py",
        "current_compare",
        "Aligned native-vs-FFS reprojection diagnostics.",
        "quick",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_rerun.py",
        "current_compare",
        "Rerun export plus fused PLYs for removed-invisible inspection.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_stereo_order_pcd.py",
        "current_compare",
        "Current-vs-swapped stereo-order registration board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_compare_turntable.py",
        "current_compare",
        "Current single-frame professor-facing compare.",
        "quick",
    ),
    HarnessEntry(
        "scripts/harness/visual_make_match_board.py",
        "current_compare",
        "Professor-facing 3-view point-cloud match board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/visual_make_professor_triptych.py",
        "current_compare",
        "Professor-facing three-figure summary pack.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_edgetam_vs_sam21_compile_ablation.py",
        "experiments",
        "Official-style EdgeTAM compile-mode vs SAM2.1 Small/Tiny speed ablation.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_edgetam_video_masks.py",
        "experiments",
        "EdgeTAM video mask worker used by the dynamics 3x6 panel experiment.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/benchmark_edgetam_trt_components.py",
        "experiments",
        "Benchmark EdgeTAM ONNX/TensorRT component engines on recorded frames.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/inspect_edgetam_onnx.py",
        "experiments",
        "Inspect EdgeTAM ONNX component graph shapes and op coverage.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_ffs_confidence_filter_sweep.py",
        "experiments",
        "FFS confidence filtering sweep runner.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_sam21_checkpoint_ladder_3x5_gifs.py",
        "experiments",
        "SAM3.1 vs SAM2.1 checkpoint ladder 3x5 time GIF benchmark.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_sloth_base_motion_mask_overlay_3x3_gif.py",
        "experiments",
        "Regenerate sloth_base_motion masks and render Small/Tiny/compiled EdgeTAM XOR overlay GIF.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_sloth_base_motion_fused_pcd_overlay_2x3_gif.py",
        "experiments",
        "Render sloth_base_motion Small/compiled EdgeTAM fused-PCD overlay GIF against SAM3.1.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/run_still_object_round1_projection_panel.py",
        "experiments",
        "Still-object round1 native/FFS projected-PCD removal board.",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py",
        "experiments",
        "No cleanup vs PhysTwin-like radius cleanup vs enhanced component cleanup.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_enhanced_phystwin_removed_overlay.py",
        "experiments",
        "Removed-point overlay and optional IR-pair board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_ffs_confidence_filter_pcd.py",
        "experiments",
        "Confidence-filtered FFS point-cloud board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_ffs_confidence_threshold_sweep_pcd.py",
        "experiments",
        "Confidence threshold sweep point-cloud board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py",
        "experiments",
        "Multipage object-mask erosion sweep.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_ffs_mask_erode_sweep_pcd.py",
        "experiments",
        "Compact object-mask erosion sweep.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visual_compare_native_ffs_fused_pcd.py",
        "experiments",
        "Native, original FFS, and fused native/FFS point-cloud board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualize_sam21_edgetam_mask_overlay_3x3_gif.py",
        "experiments",
        "SAM2.1 Small/Tiny and compiled EdgeTAM mask overlay GIF against SAM3.1.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualize_ffs_static_confidence_panels.py",
        "experiments",
        "Static masked RGB/depth/confidence board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualize_ffs_static_confidence_pcd_panels.py",
        "experiments",
        "Static masked RGB/PCD/confidence board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualize_still_object_orbit_gif.py",
        "experiments",
        "Headless Native Depth vs FFS masked-object orbit GIF.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py",
        "experiments",
        "Still-object/rope orbit GIF erosion sweep.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py",
        "experiments",
        "Still-object/rope orbit GIF board.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/audit_ffs_left_right.py",
        "focused_diagnostics",
        "Focused FFS left/right ordering audit.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/compare_face_smoothness.py",
        "focused_diagnostics",
        "Fixed face-patch smoothness/noise comparison.",
        "full",
    ),
    HarnessEntry(
        "scripts/harness/diagnose_floating_point_sources.py",
        "focused_diagnostics",
        "Floating-point source diagnostics for aligned cases.",
        "full",
    ),
)


def entries_by_category() -> dict[str, tuple[HarnessEntry, ...]]:
    grouped: dict[str, list[HarnessEntry]] = {}
    for entry in CATALOG:
        grouped.setdefault(entry.category, []).append(entry)
    return {category: tuple(entries) for category, entries in grouped.items()}


def help_scripts(profile: HelpProfile) -> tuple[str, ...]:
    if profile == "quick":
        return tuple(entry.path for entry in CATALOG if entry.help_profile == "quick")
    if profile == "full":
        return tuple(entry.path for entry in CATALOG if entry.help_profile in {"quick", "full"})
    raise ValueError(f"Unsupported profile: {profile}")
