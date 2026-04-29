# Harness Script Map

`scripts/harness/` is for thin operator-facing CLIs, probes, guards, and bounded diagnostics. Shared math, rendering, calibration, depth, and point-cloud logic belongs under `data_process/`.

## Operating Rules

- Keep orchestration and reusable logic in `data_process/visualization/` or `data_process/depth_backends/`; harness files should mostly parse args and call those modules.
- Keep one-off experiment CLIs under `scripts/harness/experiments/`; do not add root-level compatibility wrappers for experiment scripts.
- Keep RealSense, external dependency, and environment proof-of-life utilities here because they are operator tools, not production library code.
- Do not grow this folder with reusable point-cloud IO, crop logic, layout builders, calibration loaders, or view-planning helpers.

## FFS Defaults

- New realtime and visualization FFS work defaults to env `FFS-SAM-RS`, checkpoint `20-30-48`, `valid_iters=4`, `max_disp=192`, and two-stage ONNX/TensorRT.
- Current default TensorRT artifact: `data/experiments/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864/`.
- QQTT performance claims must use real RealSense `848x480` inputs from local recorded/static `ir_left` and `ir_right` unless explicitly labeled as synthetic/control.
- Models needing multiples of `32` should pad `848x480` to `864x480` and unpad outputs afterward; do not report resized `640x480` as QQTT runtime.
- `640x480` runs are allowed only as official-table reproduction/control benchmarks and must be labeled as such.
- PyTorch-only confidence-logit experiments still use the same checkpoint/iteration defaults because the current TRT artifact does not export confidence logits.
- `experiments/run_still_object_round1_projection_panel.py` aligns still-object round1 with this default level-5 TRT setting, writes masks inside the experiment folder, and renders the 13-row native/FFS projected-PCD removal board.

## Object Cases

- Use `scripts.harness.object_case_registry` for raw object-case lookup by `(object_set, round_id)`.
- Keep previous `static_object` captures separate from 2026-04-28 `still_object` captures; do not silently reuse `static_round1` names.
- Current still-object raw captures:
  - `data_collect/both_30_still_object_round1_20260428`
  - `data_collect/both_30_still_object_round2_20260428`
  - `data_collect/both_30_still_object_round3_20260428`
  - `data_collect/both_30_still_object_round4_20260428`
  - `data_collect/both_30_still_object_round7_20260428` (intentional round5-6 skip)
  - `data_collect/both_30_still_object_round8_20260428` (intentional non-contiguous sequence)
- When aligned, still-object outputs should use a distinct namespace such as `data/still_object/`.

| Helper | Purpose |
| --- | --- |
| `object_case_registry.py` | Shared raw object capture registry for harness scripts and tests. |

## Checks

| Script | Purpose |
| --- | --- |
| `check_all.py` | Deterministic quick/full validation runner. Default quick profile should stay under roughly one minute. |
| `check_experiment_boundaries.py` | Prevent experiment-only modules from leaking into formal runtime code. |
| `check_scope.py` | Enforce repo scope boundaries. |
| `check_visual_architecture.py` | Guard visualization layering and file-size constraints. |

Run:

```bash
python scripts/harness/check_all.py
python scripts/harness/check_all.py --full
```

## Hardware And External Proofs

| Script | Purpose |
| --- | --- |
| `benchmark_ffs_configs.py` | FFS config benchmark wrapper. |
| `probe_d455_ir_pair.py` | Manual D455 IR-pair capability probe. |
| `probe_d455_stream_capability.py` | Manual D455 stream/profile probe. |
| `realtime_single_camera_pointcloud.py` | Manual single-D455 realtime `camera_color_frame` RGB-D viewer with fast camera-view image backend, point-cloud backend, and Open3D FPS/latency HUD. |
| `render_d455_stream_probe_report.py` | Convert probe JSON to a readable report. |
| `run_ffs_static_replay_matrix.py` | Static replay benchmark matrix over recorded RealSense pairs. |
| `verify_ffs_demo.py` | External FFS demo proof-of-life. |
| `verify_ffs_single_engine_tensorrt_wsl.py` | Single-engine TensorRT proof on WSL. |
| `verify_ffs_tensorrt_windows.py` | TensorRT proof on Windows side. |
| `verify_ffs_tensorrt_wsl.py` | TensorRT proof on WSL. |
| `run_ffs_on_saved_pair.py` | Run external FFS on one saved stereo pair. |
| `reproject_ffs_to_color.py` | Reproject FFS output into the color camera frame. |
| `generate_sam31_masks.py` | Operator-side SAM 3.1 sidecar mask generation. |
| `sam31_mask_helper.py` | Shared implementation for the SAM 3.1 sidecar CLI and tests. |

External repos, checkpoints, and SAM assets stay outside this repo and are passed by path or resolved from environment/cache.

## Formal Cleanup

| Script | Purpose |
| --- | --- |
| `cleanup_different_types_cases.py` | Dry-run or execute downstream-facing cleanup under `data/different_types/`; preserves canonical `color/`, `depth/`, and optional color MP4 sidecars. |

## Current Compare CLIs

These should stay thin wrappers around workflow modules under `data_process/visualization/workflows/`.

| Script | Purpose |
| --- | --- |
| `visual_compare_depth_panels.py` | Per-camera RealSense-vs-FFS depth diagnostic panels. |
| `visual_compare_reprojection.py` | Cross-view reprojection diagnostics. |
| `visual_compare_depth_video.py` | Older temporal fused depth compare. |
| `visual_compare_depth_triplet_ply.py` | Single-frame native / FFS raw / FFS postprocess fused PLY compare. |
| `visual_compare_depth_triplet_video.py` | Multi-frame native / FFS raw / FFS postprocess point-cloud video compare. |
| `visual_compare_masked_pointcloud.py` | Single-frame SAM-masked native-vs-FFS point-cloud board. |
| `visual_compare_masked_camera_views.py` | Masked native-vs-FFS views from original calibrated cameras. |
| `visual_compare_turntable.py` | Current single-frame professor-facing compare. |
| `visual_compare_rerun.py` | Rerun export plus fused PLYs for native-vs-FFS removed-invisible inspection. |
| `visual_make_match_board.py` | Professor-facing 3-view point-cloud match board. |
| `visual_make_professor_triptych.py` | Three-figure professor-facing summary pack. |

## Experiment CLIs

| Script | Purpose |
| --- | --- |
| `experiments/run_ffs_confidence_filter_sweep.py` | FFS confidence filtering sweep runner. |
| `experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py` | Native/FFS PCD compare under no postprocess, PT-like, and enhanced PT-like cleanup. |
| `experiments/visual_compare_enhanced_phystwin_removed_overlay.py` | `5x3` RealSense-depth/FFS removed-point overlay; `--native_row_mode ir_pair` produces `6x3` IR-left/right boards. |
| `experiments/visual_compare_ffs_confidence_filter_pcd.py` | `6x3` native/raw/confidence-filtered FFS PCD board. |
| `experiments/visual_compare_ffs_confidence_threshold_sweep_pcd.py` | Threshold sweep over confidence-filtered FFS PCD boards. |
| `experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py` | Multipage mask-erosion sweep. |
| `experiments/visual_compare_ffs_mask_erode_sweep_pcd.py` | Compact mask-erosion sweep. |
| `experiments/visual_compare_native_ffs_fused_pcd.py` | Native, original FFS, and fused native/FFS PCD board. |
| `experiments/visualize_ffs_static_confidence_panels.py` | Static masked RGB/depth/confidence panel boards. |
| `experiments/visualize_ffs_static_confidence_pcd_panels.py` | Static masked RGB/PCD/confidence panel boards. |

## Focused Diagnostics

| Script | Purpose |
| --- | --- |
| `audit_ffs_left_right.py` | Focused FFS left/right ordering audit. |
| `compare_face_smoothness.py` | Fixed face-patch smoothness/noise comparison. |
| `diagnose_floating_point_sources.py` | Floating-point source diagnostics for aligned cases. |
| `visual_compare_stereo_order_pcd.py` | Current-vs-swapped stereo-order registration board. |

## Retention Policy

- Keep current user-facing CLIs, deterministic checks, hardware probes, and bounded diagnostics.
- Remove local cache directories such as `__pycache__/`.
- Move reusable implementation out of harness before it becomes shared behavior.
- Record external dependency proof-of-life outcomes under `docs/generated/`, not in script comments.
