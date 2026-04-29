# Harness Script Map

`scripts/harness/` is for operator-facing CLIs, probes, guards, and bounded diagnostics. Reusable calibration, depth, point-cloud, layout, and rendering logic belongs under `data_process/`.

## Operating Rules

- Harness files should mostly parse CLI args and call implementation modules.
- Put one-off visualization experiments under `scripts/harness/experiments/`; avoid root-level compatibility wrappers for experiments.
- Keep RealSense, TensorRT, SAM, WSLg, and external-dependency proof utilities here because they are operator tools.
- Do not add reusable point-cloud IO, crop logic, layout builders, calibration loaders, or view-planning helpers to this folder.
- External repos, checkpoints, TensorRT engines, and SAM assets stay outside this repo and are passed by path, environment, or cache.

## FFS Defaults

- Default FFS environment: `FFS-SAM-RS`.
- Default checkpoint/config: `20-30-48`, `valid_iters=4`, `max_disp=192`, two-stage ONNX/TensorRT.
- Default TensorRT artifact: `data/experiments/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864/`.
- QQTT performance claims must use real local RealSense `848x480` `ir_left` / `ir_right` inputs unless explicitly labeled synthetic or control.
- Models that require multiples of `32` should pad `848x480` to `864x480` and unpad outputs afterward.
- `640x480` runs are only official-table reproduction/control benchmarks and must be labeled that way.
- PyTorch confidence-logit experiments use the same checkpoint and iteration defaults because the current TensorRT artifact does not export confidence logits.
- Single-camera realtime FFS uses latest-wins capture, inference, render-prep, and UI stages; the known-good Open3D path is packet-arrival coalesced rendering, not a fixed-60Hz GUI timer.
- Single-camera orbit mode defaults to `--max-points 200000` and `--point-size 1.0`; explicit CLI values override both.

## Object Cases

- Use `scripts.harness.object_case_registry` for raw object-case lookup by `(object_set, round_id)`.
- Keep old `static_object` captures separate from 2026-04-28 `still_object` captures.
- Current still-object raw captures are `both_30_still_object_round1_20260428`, `round2`, `round3`, `round4`, `round7`, and `round8`; the round5-6 skip is intentional.
- Aligned still-object outputs should use a distinct namespace such as `data/still_object/`.

## Checks

| File | Purpose |
| --- | --- |
| `check_all.py` | Deterministic quick/full validation runner. |
| `check_experiment_boundaries.py` | Prevent experiment modules from leaking into formal runtime code. |
| `check_scope.py` | Enforce repo scope boundaries. |
| `check_visual_architecture.py` | Guard visualization layering and file-size constraints. |

Default quick checks cover core entrypoint help, formal record/alignment help, repo guards, FFS geometry contracts, and a small current visualization smoke set. Use `--full` for broad experiment CLI coverage, realtime demo smoke tests, TensorRT/benchmark helper coverage, pytest probe schema checks, or shared visualization/depth changes.

```bash
python scripts/harness/check_all.py
python scripts/harness/check_all.py --full
```

## Hardware And External Proofs

| File | Purpose |
| --- | --- |
| `benchmark_ffs_configs.py` | FFS config benchmark wrapper. |
| `probe_d455_ir_pair.py`, `probe_d455_stream_capability.py` | Manual D455 IR/profile probes. |
| `realtime_single_camera_pointcloud.py` | Manual single-D455 realtime RGB-D viewer with RealSense or FFS depth and Open3D HUD. |
| `run_wslg_open3d.sh` | WSLg Open3D GUI wrapper. |
| `render_d455_stream_probe_report.py` | Convert stream probe JSON to a readable report. |
| `run_ffs_static_replay_matrix.py` | Static replay benchmark matrix over recorded RealSense pairs. |
| `verify_ffs_demo.py`, `verify_ffs_single_engine_tensorrt_wsl.py`, `verify_ffs_tensorrt_windows.py`, `verify_ffs_tensorrt_wsl.py` | External FFS and TensorRT proof-of-life utilities. |
| `run_ffs_on_saved_pair.py`, `reproject_ffs_to_color.py` | Single-pair FFS run and color-frame reprojection helpers. |
| `generate_sam31_masks.py`, `sam31_mask_helper.py` | Operator-side SAM 3.1 mask generation and shared helper. |

Record durable external dependency proof results under `docs/generated/`, not in script comments.

## Formal Cleanup

| File | Purpose |
| --- | --- |
| `cleanup_different_types_cases.py` | Dry-run or execute downstream-facing cleanup under `data/different_types/`; preserves canonical `color/`, `depth/`, and optional color MP4 sidecars. |

## Current Compare CLIs

These stay thin wrappers around workflow modules under `data_process/visualization/workflows/`.

| File | Purpose |
| --- | --- |
| `visual_compare_depth_panels.py` | Per-camera RealSense-vs-FFS depth panels. |
| `visual_compare_reprojection.py` | Cross-view reprojection diagnostics. |
| `visual_compare_depth_video.py`, `visual_compare_depth_triplet_video.py`, `visual_compare_depth_triplet_ply.py` | Older temporal/fused native-vs-FFS depth comparisons. |
| `visual_compare_masked_pointcloud.py`, `visual_compare_masked_camera_views.py` | SAM-masked native-vs-FFS point-cloud and camera-view boards. |
| `visual_compare_turntable.py`, `visual_make_match_board.py`, `visual_make_professor_triptych.py` | Current professor-facing comparison outputs. |
| `visual_compare_rerun.py` | Rerun export plus fused PLYs for removed-invisible inspection. |

## Experiment CLIs

These belong under `scripts/harness/experiments/` and call implementation modules under `data_process/visualization/experiments/`.

| File | Purpose |
| --- | --- |
| `run_ffs_confidence_filter_sweep.py` | FFS confidence filtering sweep runner. |
| `run_still_object_round1_projection_panel.py` | Still-object round1 native/FFS projected-PCD removal board. |
| `visualize_still_object_orbit_gif.py` | Headless Native Depth vs FFS masked-object orbit GIF. |
| `visualize_still_object_rope_6x2_orbit_gif.py` | Headless raw-RGB Native Depth vs FFS orbit GIF for still-object rounds 1-4 and still-rope rounds 1-2; supports `--layout 6x2/3x4`, mask erosion, deletion-mode enhanced PT-like postprocess, and no-delete removed-point marking by source camera color. |
| `visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py` | Four orbit GIFs with 1/3/5/10px object-mask erosion; defaults to 3x4 layout with two rounds per row and no-delete enhanced PT-like removed-point marking by source camera color. |
| `visual_compare_enhanced_phystwin_postprocess_pcd.py` | No cleanup vs PhysTwin-like radius-neighbor cleanup vs enhanced component cleanup. |
| `visual_compare_enhanced_phystwin_removed_overlay.py` | `5x3` removed-point overlay; `--native_row_mode ir_pair` produces `6x3` infrared-left/right boards. |
| `visual_compare_ffs_confidence_filter_pcd.py`, `visual_compare_ffs_confidence_threshold_sweep_pcd.py` | Confidence-filtered Fast-FoundationStereo point-cloud boards and threshold sweeps. |
| `visual_compare_ffs_mask_erode_sweep_pcd.py`, `visual_compare_ffs_mask_erode_multipage_sweep_pcd.py` | Compact and multipage mask-erosion sweeps. |
| `visual_compare_native_ffs_fused_pcd.py` | Native, original FFS, and fused native/FFS point-cloud board. |
| `visualize_ffs_static_confidence_panels.py`, `visualize_ffs_static_confidence_pcd_panels.py` | Static masked RGB/depth/confidence and point-cloud confidence boards. |

## Focused Diagnostics

| File | Purpose |
| --- | --- |
| `audit_ffs_left_right.py` | Focused FFS left/right ordering audit. |
| `compare_face_smoothness.py` | Fixed face-patch smoothness/noise comparison. |
| `diagnose_floating_point_sources.py` | Floating-point source diagnostics for aligned cases. |
| `object_case_registry.py` | Shared raw object capture registry for harness scripts and tests. |
| `visual_compare_stereo_order_pcd.py` | Current-vs-swapped stereo-order registration board. |

## Retention Policy

- Keep current user-facing CLIs, deterministic checks, hardware probes, and bounded diagnostics.
- Remove ignored local cache directories such as `__pycache__/`.
- Move reusable implementation out of harness before it becomes shared behavior.
