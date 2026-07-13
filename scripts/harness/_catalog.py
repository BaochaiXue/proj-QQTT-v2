from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValidationProfile = Literal["smoke", "deterministic", "hardware", "exhaustive"]
ProfileAlias = Literal["quick", "full"]
Lifecycle = Literal["guards", "validation", "diagnostics", "benchmarks", "experiments", "support"]

VALIDATION_PROFILES: tuple[str, ...] = ("smoke", "deterministic", "hardware", "exhaustive")
LIFECYCLES: tuple[str, ...] = ("guards", "validation", "diagnostics", "benchmarks", "experiments", "support")


@dataclass(frozen=True)
class HarnessEntry:
    path: str
    lifecycle: Lifecycle
    domain: str
    summary: str
    validation_profile: ValidationProfile | None = None
    help: bool = False
    automatic: bool = True
    requires: tuple[str, ...] = ()

    @property
    def category(self) -> str:
        if self.lifecycle in {"guards", "validation"}:
            return "checks"
        if self.lifecycle == "experiments":
            return "experiments"
        if self.lifecycle == "benchmarks":
            return "hardware_external"
        if self.lifecycle == "support":
            if self.domain == "data":
                return "formal_cleanup"
            return "mask_support"
        if self.lifecycle == "diagnostics":
            if self.validation_profile == "hardware":
                return "hardware_external"
            basename = self.path.rsplit("/", 1)[-1]
            if self.domain == "depth" and basename in {
                "audit_ffs_left_right.py",
                "compare_face_smoothness.py",
                "diagnose_floating_point_sources.py",
            }:
                return "focused_diagnostics"
            return "current_compare"
        raise ValueError(f"Unsupported lifecycle: {self.lifecycle}")

    @property
    def help_profile(self) -> str | None:
        if not self.help:
            return None
        if self.validation_profile == "smoke":
            return "quick"
        if self.validation_profile in {"deterministic", "exhaustive", "hardware"}:
            return "full"
        return None


CATALOG: tuple[HarnessEntry, ...] = (
    HarnessEntry(
        "scripts/harness/validation/run.py",
        "validation",
        "runner",
        "Catalog-driven validation profile runner.",
        "smoke",
    ),
    HarnessEntry(
        "scripts/harness/guards/check_experiment_boundaries.py",
        "guards",
        "repo",
        "Guard formal runtime code from experiment-only imports.",
        "smoke",
    ),
    HarnessEntry(
        "scripts/harness/guards/check_harness_catalog.py",
        "guards",
        "repo",
        "Guard that every harness Python file is categorized here.",
        "smoke",
    ),
    HarnessEntry(
        "scripts/harness/guards/check_demo_v5_no_compat_wrappers.py",
        "guards",
        "repo",
        "Guard Demo v5 and Demo v5.1 from legacy import wrapper modules.",
        "smoke",
    ),
    HarnessEntry(
        "scripts/harness/guards/check_scope.py",
        "guards",
        "repo",
        "Repo scope guard for removed or forbidden legacy surfaces.",
        "smoke",
    ),
    HarnessEntry(
        "scripts/harness/guards/check_visual_architecture.py",
        "guards",
        "repo",
        "Visualization layering and file-size guard.",
        "smoke",
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/benchmark_ffs_configs.py",
        "benchmarks",
        "ffs",
        "Saved-pair PyTorch FFS config screening for single-camera FFS depth work.",
        "hardware",
        help=True,
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/sam/benchmark_sam31_still_object_views.py",
        "benchmarks",
        "sam",
        "SAM 3.1 30-frame still-object per-camera segmentation benchmark.",
        "exhaustive",
        help=True,
        requires=("gpu", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/hardware/probe_d455_ir_pair.py",
        "diagnostics",
        "hardware",
        "Manual D455 IR-pair probe.",
        "hardware",
        help=True,
        automatic=False,
        requires=("camera",),
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/hardware/probe_d455_stream_capability.py",
        "diagnostics",
        "hardware",
        "Manual D455 stream/profile capability probe.",
        "hardware",
        help=True,
        automatic=False,
        requires=("camera",),
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/hardware/render_d455_stream_probe_report.py",
        "diagnostics",
        "hardware",
        "Render D455 probe JSON as a readable report.",
        "hardware",
        help=True,
        automatic=False,
        requires=("camera",),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/run_ffs_on_saved_pair.py",
        "benchmarks",
        "ffs",
        "Run FFS on one saved stereo pair.",
        "hardware",
        help=True,
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/run_ffs_static_replay_matrix.py",
        "benchmarks",
        "ffs",
        "Offline static replay / TensorRT proxy matrix; not live PyTorch realtime.",
        "hardware",
        help=True,
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/hardware/run_wslg_open3d.sh",
        "diagnostics",
        "hardware",
        "WSLg Open3D GUI wrapper.",
        "hardware",
        automatic=False,
        requires=("camera", "gpu", "gui"),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/verify_ffs_demo.py",
        "benchmarks",
        "ffs",
        "External FFS demo proof-of-life utility.",
        "hardware",
        help=True,
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/verify_ffs_single_engine_tensorrt_wsl.py",
        "benchmarks",
        "ffs",
        "WSL single-engine TensorRT proof-of-life utility.",
        "hardware",
        help=True,
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/verify_ffs_tensorrt_windows.py",
        "benchmarks",
        "ffs",
        "Windows TensorRT proof-of-life utility.",
        "hardware",
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/benchmarks/ffs/verify_ffs_tensorrt_wsl.py",
        "benchmarks",
        "ffs",
        "WSL TensorRT proof-of-life utility.",
        "hardware",
        help=True,
        automatic=False,
        requires=("gpu", "tensorrt", "external_repo"),
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/generate_sam31_masks.py",
        "diagnostics",
        "mask",
        "Operator-side SAM 3.1 mask generation CLI.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/support/object_case_registry.py",
        "support",
        "mask",
        "Shared raw object capture registry for harness scripts and tests.",
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/reproject_ffs_to_color.py",
        "diagnostics",
        "reprojection",
        "Reproject single-pair FFS depth into color-frame geometry.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/demo/render_demo32_headless_capture.py",
        "diagnostics",
        "demo",
        "Render Demo 3.2 headless enhanced-pt PCD capture artifacts to MP4.",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/support/sam31_mask_helper.py",
        "support",
        "mask",
        "Shared SAM 3.1 helper used by operator-side harness CLIs.",
    ),
    HarnessEntry(
        "scripts/harness/support/cleanup_different_types_cases.py",
        "support",
        "data",
        "Dry-run or execute data/different_types cleanup.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/visual_compare_depth_panels.py",
        "diagnostics",
        "depth",
        "Per-camera RealSense-vs-FFS depth panels.",
        "smoke",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/visual_compare_depth_triplet_ply.py",
        "diagnostics",
        "depth",
        "Single-frame native/FFS raw/FFS postprocess fused PLY compare.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/visual_compare_depth_triplet_video.py",
        "diagnostics",
        "depth",
        "Multi-frame native/FFS raw/FFS postprocess point-cloud video compare.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/visual_compare_depth_video.py",
        "diagnostics",
        "depth",
        "Older temporal fused native-vs-FFS depth compare.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_compare_masked_camera_views.py",
        "diagnostics",
        "mask",
        "SAM-masked native-vs-FFS camera-view board.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_compare_masked_pointcloud.py",
        "diagnostics",
        "mask",
        "SAM-masked native-vs-FFS point-cloud board.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/visual_compare_reprojection.py",
        "diagnostics",
        "reprojection",
        "Aligned native-vs-FFS reprojection diagnostics.",
        "smoke",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_compare_rerun.py",
        "diagnostics",
        "rerun",
        "Rerun export plus fused PLYs for removed-invisible inspection.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_compare_stereo_order_pcd.py",
        "diagnostics",
        "depth",
        "Current-vs-swapped stereo-order registration board.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_compare_turntable.py",
        "diagnostics",
        "depth",
        "Current single-frame professor-facing compare.",
        "smoke",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_make_match_board.py",
        "diagnostics",
        "depth",
        "Professor-facing 3-view point-cloud match board.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/visualization/visual_make_professor_triptych.py",
        "diagnostics",
        "depth",
        "Professor-facing three-figure summary pack.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_edgetam_vs_sam21_compile_ablation.py",
        "experiments",
        "edgetam",
        "Official-style EdgeTAM compile-mode vs SAM2.1 Small/Tiny speed ablation.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_edgetam_video_masks.py",
        "experiments",
        "edgetam",
        "EdgeTAM video mask worker used by the dynamics 3x6 panel experiment.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/benchmark_edgetam_trt_components.py",
        "experiments",
        "edgetam",
        "Benchmark EdgeTAM ONNX/TensorRT component engines on recorded frames.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/inspect_edgetam_onnx.py",
        "experiments",
        "edgetam",
        "Inspect EdgeTAM ONNX component graph shapes and op coverage.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/probe_edgetam_video_trt_compile.py",
        "experiments",
        "edgetam",
        "Probe official EdgeTAM video components for ONNX/TensorRT compile feasibility.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_hf_edgetam_streaming_realcase.py",
        "experiments",
        "edgetam",
        "Hugging Face EdgeTAMVideo streaming benchmark on real aligned QQTT cases.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_sloth_set2_hf_edgetam_streaming_pcd_xor_gif.py",
        "experiments",
        "edgetam",
        "Render Sloth Set 2 HF EdgeTAM streaming fused-PCD XOR GIF against SAM3.1.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py",
        "experiments",
        "edgetam",
        "Render Sloth Set 2 HF EdgeTAM streaming hand/object fused-PCD GIF.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/run_ffs_confidence_filter_sweep.py",
        "experiments",
        "ffs",
        "FFS confidence filtering sweep runner.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/sam/run_sam21_checkpoint_ladder_3x5_gifs.py",
        "experiments",
        "sam",
        "SAM3.1 vs SAM2.1 checkpoint ladder 3x5 time GIF benchmark.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_sloth_base_motion_mask_overlay_3x3_gif.py",
        "experiments",
        "edgetam",
        "Regenerate sloth_base_motion masks and render Small/Tiny/compiled EdgeTAM XOR overlay GIF.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/edgetam/run_sloth_base_motion_fused_pcd_overlay_2x3_gif.py",
        "experiments",
        "edgetam",
        "Render sloth_base_motion Small/compiled EdgeTAM fused-PCD overlay GIF against SAM3.1.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualization/run_still_object_round1_projection_panel.py",
        "experiments",
        "still_object",
        "Still-object round1 native/FFS projected-PCD removal board.",
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visual_compare_ffs_confidence_filter_pcd.py",
        "experiments",
        "ffs",
        "Confidence-filtered FFS point-cloud board.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visual_compare_ffs_confidence_threshold_sweep_pcd.py",
        "experiments",
        "ffs",
        "Confidence threshold sweep point-cloud board.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py",
        "experiments",
        "ffs",
        "Multipage object-mask erosion sweep.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visual_compare_ffs_mask_erode_sweep_pcd.py",
        "experiments",
        "ffs",
        "Compact object-mask erosion sweep.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visual_compare_native_ffs_fused_pcd.py",
        "experiments",
        "ffs",
        "Native, original FFS, and fused native/FFS point-cloud board.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/sam/visualize_sam21_edgetam_mask_overlay_3x3_gif.py",
        "experiments",
        "edgetam",
        "SAM2.1 Small/Tiny and compiled EdgeTAM mask overlay GIF against SAM3.1.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visualize_ffs_static_confidence_panels.py",
        "experiments",
        "ffs",
        "Static masked RGB/depth/confidence board.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/ffs/visualize_ffs_static_confidence_pcd_panels.py",
        "experiments",
        "ffs",
        "Static masked RGB/PCD/confidence board.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualization/visualize_still_object_orbit_gif.py",
        "experiments",
        "still_object",
        "Headless Native Depth vs FFS masked-object orbit GIF.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualization/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py",
        "experiments",
        "still_object",
        "Still-object/rope orbit GIF erosion sweep.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/experiments/visualization/visualize_still_object_rope_6x2_orbit_gif.py",
        "experiments",
        "still_object",
        "Still-object/rope orbit GIF board.",
        "exhaustive",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/audit_ffs_left_right.py",
        "diagnostics",
        "depth",
        "Focused FFS left/right ordering audit.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/compare_face_smoothness.py",
        "diagnostics",
        "depth",
        "Fixed face-patch smoothness/noise comparison.",
        "deterministic",
        help=True,
    ),
    HarnessEntry(
        "scripts/harness/diagnostics/depth/diagnose_floating_point_sources.py",
        "diagnostics",
        "depth",
        "Floating-point source diagnostics for aligned cases.",
        "deterministic",
        help=True,
    ),
)


def entries_by_lifecycle() -> dict[str, tuple[HarnessEntry, ...]]:
    grouped: dict[str, list[HarnessEntry]] = {}
    for entry in CATALOG:
        grouped.setdefault(entry.lifecycle, []).append(entry)
    return {lifecycle: tuple(entries) for lifecycle, entries in grouped.items()}


def _normalize_profile(profile: ValidationProfile | ProfileAlias) -> ValidationProfile:
    if profile == "quick":
        return "smoke"
    if profile == "full":
        return "deterministic"
    return profile


def entries_for_profile(
    profile: ValidationProfile | ProfileAlias,
    *,
    include_manual: bool = False,
) -> tuple[HarnessEntry, ...]:
    profile = _normalize_profile(profile)
    if profile == "smoke":
        allowed = {"smoke"}
    elif profile == "deterministic":
        allowed = {"smoke", "deterministic"}
    elif profile == "exhaustive":
        allowed = {"smoke", "deterministic", "exhaustive"}
    elif profile == "hardware":
        allowed = {"hardware"}
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    entries = []
    for entry in CATALOG:
        if entry.validation_profile not in allowed:
            continue
        if not include_manual and not entry.automatic:
            continue
        entries.append(entry)
    return tuple(entries)


def help_scripts(
    profile: ValidationProfile | ProfileAlias,
    *,
    include_manual: bool = False,
) -> tuple[str, ...]:
    return tuple(
        entry.path
        for entry in entries_for_profile(profile, include_manual=include_manual)
        if entry.help
    )
