# Harness Script Map

This folder is intentionally limited to thin command-line wrappers, probe/check utilities, and a small number of bounded one-off diagnostics.

Rule of thumb:

- keep orchestration, rendering, and analysis logic in `data_process/visualization/` or `data_process/depth_backends/`
- keep `scripts/harness/` as the user-facing CLI shell around that logic
- keep experiment-only CLIs under `scripts/harness/experiments/`
- keep RealSense / environment checks here because they are operational harness utilities, not production library code

## Keep Here

### Checks / Guards

- `check_all.py`
- `check_experiment_boundaries.py`
- `check_scope.py`
- `check_visual_architecture.py`

`check_all.py` now has two deterministic profiles:

- default `python scripts/harness/check_all.py`
  - fast quick profile
  - intended to finish in under one minute on a healthy repo environment
- explicit `python scripts/harness/check_all.py --full`
  - broader legacy validation surface
  - use for larger refactors or when you need the old wide regression net

### Hardware / External Proof-of-Life

- `verify_ffs_demo.py`
- `verify_ffs_tensorrt_windows.py`
- `verify_ffs_tensorrt_wsl.py`
- `probe_d455_ir_pair.py`
- `probe_d455_stream_capability.py`
- `render_d455_stream_probe_report.py`
- `run_ffs_on_saved_pair.py`
- `reproject_ffs_to_color.py`
- `generate_sam31_masks.py`

### Data Cleanup

- `cleanup_different_types_cases.py`

This script is for downstream-facing final-case cleanup under `data/different_types/` and is intentionally separate from the repo-internal aligned-case visualization workflows.
It preserves the canonical `color/` and `depth/` frame trees plus optional `color/0.mp4`, `1.mp4`, and `2.mp4` RGB sidecars when present. Execute mode also backfills those color mp4 sidecars from `color/<camera>/*.png` if they are missing.

### Current User-Facing Compare CLIs

- `visual_compare_depth_panels.py`
- `visual_compare_reprojection.py`
- `visual_compare_depth_video.py`
- `visual_compare_depth_triplet_ply.py`
- `visual_compare_depth_triplet_video.py`
- `visual_compare_masked_pointcloud.py`
- `visual_compare_masked_camera_views.py`
- `visual_compare_turntable.py`
- `visual_compare_rerun.py`
- `visual_make_match_board.py`
- `visual_make_professor_triptych.py`

These should stay thin wrappers around workflow modules under `data_process/visualization/workflows/`.

### Experiment-Only CLIs

- `experiments/run_ffs_confidence_filter_sweep.py`
- `experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py`
- `experiments/visual_compare_enhanced_phystwin_removed_overlay.py`
- `experiments/visual_compare_ffs_confidence_filter_pcd.py`
- `experiments/visual_compare_ffs_confidence_threshold_sweep_pcd.py`
- `experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py`
- `experiments/visual_compare_ffs_mask_erode_sweep_pcd.py`
- `experiments/visual_compare_native_ffs_fused_pcd.py`
- `experiments/visualize_ffs_static_confidence_panels.py`
- `experiments/visualize_ffs_static_confidence_pcd_panels.py`

These are canonical paths for one-off FFS experiments. The old root-level
paths remain as thin compatibility wrappers, but new experiments should not add
more root-level harness scripts.

`experiments/visual_compare_ffs_confidence_filter_pcd.py` renders the static round 1-3 frame-0 object-mask `6x3` Open3D boards for native, raw FFS, and four confidence-filtered FFS variants. Use `--phystwin_like_postprocess --phystwin_radius_m 0.01 --phystwin_nb_points 40` when the displayed clouds should match the PhysTwin-like radius-neighbor cleanup.

`experiments/visual_compare_ffs_confidence_threshold_sweep_pcd.py` renders the same `6x3` board shape as one experiment over thresholds `0.01,0.05,0.10,0.15,0.20,0.25,0.50`. The default experiment uses the object mask, erodes the mask inward by `1px`, and applies the PhysTwin-like radius-neighbor cleanup before rendering each row.

`experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py` renders static round 1-3 frame-0 object-only `6x3` boards comparing native and original FFS point clouds under no postprocess, the existing PhysTwin-like radius-neighbor postprocess, and the enhanced radius-plus-3D-component postprocess. The enhanced mode keeps the main component and records removed component bbox/gap stats for tuning.

`experiments/visual_compare_enhanced_phystwin_removed_overlay.py` renders static round 1-3 source-camera `5x3` diagnostic boards with RGB object masks, object-masked native depth, fused PCD removed-point highlights, FFS depth removed overlays, and RGB removed overlays. Removed-point colors encode source camera (`Cam0` magenta, `Cam1` cyan, `Cam2` amber). The default highlight scope includes both radius-neighbor outliers and component-filter removals.

`experiments/visual_compare_ffs_mask_erode_sweep_pcd.py` renders static round 1-3 frame-0 object-only `10x3` Open3D boards for native depth, original FFS depth, and original FFS depth with mask erosion from `1px` through `8px`. The default experiment keeps all outputs under one result folder, applies display-only PhysTwin-like radius-neighbor cleanup before rendering each row, and uses a wider left label band; adjust `--row_label_width` if labels still need more room.

`experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py` renders the extended static round 1-3 frame-0 object-only mask-erode experiment as three `10x3` pages per round. Page 1 contains native depth, original FFS, and erode `1..8px`; pages 2 and 3 continue with erode `9..18px` and `19..28px` only so each page remains `10x3`. The experiment uses one result folder and applies display-only PhysTwin-like radius-neighbor cleanup before rendering.

`experiments/visual_compare_native_ffs_fused_pcd.py` renders the static round 1-3 frame-0 object-only `3x3` PCD boards for native, original FFS, and fused native/FFS depth. The fused row keeps every valid native depth pixel and uses FFS only where native depth is missing; it reuses the existing static SAM mask and applies display-only PhysTwin-like radius-neighbor cleanup before rendering.

### Focused Diagnostics

- `audit_ffs_left_right.py`
- `compare_face_smoothness.py`
- `visual_compare_stereo_order_pcd.py`

These remain in harness because they are bounded, investigation-oriented tools rather than core product workflows.

### External Sidecars

- `generate_sam31_masks.py`

This is a workspace-local helper for running external `SAM 3.1` segmentation against QQTT case assets.
It is intentionally kept out of `data_process/segment*.py` because the repo scope guard bans reintroducing that PhysTwin-era file surface.
Use it only as an operator-side sidecar:

- it reads `color/<camera>.mp4` when present, otherwise `color/<camera>/*.png`
- it writes mask artifacts under a caller-selected output directory, defaulting to `<case_root>/sam31_masks/`
- it expects `sam3` to be installed in the active environment and Hugging Face auth/checkpoint access to be handled outside the repo
- checkpoints remain external and should be passed by path or resolved from Hugging Face cache via login / `QQTT_SAM31_CHECKPOINT`
- for current PyPI `sam3`, keep `bpe_simple_vocab_16e6.txt.gz` external as well, preferably next to the checkpoint, or set `QQTT_SAM31_BPE_PATH`

## Do Not Grow Here

- large shared rendering/math helpers
- reusable point-cloud IO or crop logic
- reusable layout builders
- reusable calibration / view-planning logic

Those belong under `data_process/visualization/` and should be imported by these wrappers.

## Current Cleanup Result

- thin wrappers remain in `scripts/harness/`
- old cache/junk directories have been removed
- legacy duplicate generated-doc references have been collapsed onto the newer cleanup docs under `docs/generated/`
