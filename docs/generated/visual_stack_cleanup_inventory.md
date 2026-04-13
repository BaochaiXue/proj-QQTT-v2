# Visual Stack Cleanup Inventory

This file supersedes the older `visual_stack_inventory.md` snapshot and is the current source of truth for visualization ownership and artifact layout.

Date: 2026-04-10

## Current CLI Entrypoints

Primary compare entrypoints:

- `scripts/harness/visual_compare_depth_panels.py`
- `scripts/harness/visual_compare_reprojection.py`
- `scripts/harness/visual_compare_depth_video.py`
- `scripts/harness/visual_compare_depth_triplet_ply.py`
- `scripts/harness/visual_compare_turntable.py`
- `scripts/harness/visual_make_match_board.py`
- `scripts/harness/visual_make_professor_triptych.py`
- `scripts/harness/visual_compare_stereo_order_pcd.py`
- `scripts/harness/audit_ffs_left_right.py`
- `scripts/harness/compare_face_smoothness.py`

All of these are intended to stay thin wrappers around visualization workflow modules.

## Current Compare Workflows

### Per-camera depth review

- owner:
  - `data_process/visualization/panel_compare.py`
  - `data_process/visualization/workflows/depth_panels.py`
- product:
  - per-camera panel PNGs / optional MP4
- main question:
  - local per-camera depth quality

### Reprojection compare

- owner:
  - `data_process/visualization/reprojection_compare.py`
  - `data_process/visualization/workflows/reprojection_compare.py`
- product:
  - per-pair reprojection boards / summary metrics
- main question:
  - multi-view RGB-depth consistency

### Temporal fused depth video

- owner:
  - `data_process/visualization/pointcloud_compare.py`
- product:
  - older temporal fused-cloud videos
- main question:
  - secondary motion/consistency inspection

### Single-frame triplet fused PLY compare

- owner:
  - `data_process/visualization/triplet_ply_compare.py`
  - `data_process/visualization/workflows/triplet_ply_compare.py`
- product:
  - 3 fused full-scene PLYs
  - compact summary
- main question:
  - how `Native`, `FFS raw`, and `FFS postprocess` differ in fused 3D for one aligned frame

### Turntable compare

- owner:
  - `data_process/visualization/turntable_compare.py`
  - `data_process/visualization/workflows/turntable_compare.py`
- product:
  - hero compare
  - geom / rgb / support / source / mismatch outputs
  - turntable metadata
- main question:
  - single-frame professor-facing fused-cloud diagnosis

### Match board

- owner:
  - `data_process/visualization/match_board.py`
  - `data_process/visualization/workflows/match_board.py`
- product:
  - one 2x3 match board
  - compact summary
- main question:
  - whether Native vs FFS match well in 3-view fused geometry

### Professor triptych

- owner:
  - `data_process/visualization/professor_triptych.py`
  - `data_process/visualization/workflows/professor_triptych.py`
- product:
  - hero compare
  - merge evidence
  - truth board
  - compact summary
- main question:
  - one slide-friendly conclusion pack

### Stereo-order registration

- owner:
  - `data_process/visualization/stereo_audit.py`
  - `scripts/harness/visual_compare_stereo_order_pcd.py`
- product:
  - one 3x4 current-vs-swapped point-cloud registration board
  - compact summary
- main question:
  - whether current or swapped FFS stereo ordering registers better in 3D

## Current Display-Frame Logic

Raw calibration semantics:

- `data_process/visualization/calibration_io.py`
- `data_process/visualization/calibration_frame.py`

Visualization-only semantic display transform:

- `data_process/visualization/semantic_world.py`

Current contract:

- `calibrate.pkl` remains raw `camera_to_world (c2w)` in ChArUco calibration world
- professor-facing turntable and stereo-order workflows default to `display_frame = semantic_world`
- raw debug fallback remains available through `display_frame = calibration_world`

Current typed contract owner:

- `DisplayFrameContract` in `data_process/visualization/types.py`

## Current Angle-Selection Logic

Shared contract owner:

- `data_process/visualization/selection_contracts.py`

Current typed summaries:

- `AngleSelectionSummary`
- `TruthPairSelectionSummary`

Current users:

- `match_board.py`
  - object-aware match-angle selection
- `professor_triptych.py`
  - object-aware hero-angle selection
  - object-aware truth-pair selection

## Current ROI Contract

Main owners:

- `data_process/visualization/object_roi.py`
- `data_process/visualization/object_compare.py`
- `data_process/visualization/turntable_compare.py`

Current first-class semantics:

- pass1 world ROI
- pass2 refined world ROI
- per-camera projected bbox
- per-camera refined masks
- object-only vs context layers

Current typed contract owner:

- `RoiPassSummary` in `data_process/visualization/types.py`

Current artifact family:

- `object_roi_pass1_world.json`
- `object_roi_pass2_world.json`
- `per_camera_auto_bbox/cam*.json`
- `debug/compare_debug_metrics.json`

## Current Summary / Artifact JSON Schemas

Product-oriented summaries:

- `match_board_summary.json`
  - match angle selection
  - support/mismatch summaries
  - product/debug artifact sets
- `summary.json` from professor triptych
  - hero angle selection
  - truth pair selection
  - product/debug artifact sets
- `match_board_summary.json` from stereo-order registration
  - display-frame contract
  - render settings
  - row point counts
  - board view reuse
  - product/debug artifact sets

Metric/debug families:

- `source_metrics.json`
- `support_metrics.json`
- `mismatch_metrics.json`
- `compare_debug_metrics.json`

Shared typed contracts now available in `types.py`:

- `SourceSummary`
- `SupportSummary`
- `MismatchSummary`
- `ProductArtifactSet`
- `DebugArtifactSet`

## Current Module Ownership

### Workflow orchestration

- `panel_compare.py`
- `reprojection_compare.py`
- `turntable_compare.py`
- `match_board.py`
- `professor_triptych.py`
- `stereo_audit.py`
- `workflows/*.py`

### Shared scene/view helpers

- `compare_scene.py`
  - single-frame turntable scene assembly
  - orbit state building
  - object-view metrics

### Geometry / display-frame transforms

- `semantic_world.py`
- `views.py`
- `camera_frusta.py`
- `roi.py`

### ROI / masking / object extraction

- `object_roi.py`
- `object_compare.py`

### Rendering

- `renderers/*`
- `source_compare.py`
- `support_compare.py`

### Artifact writing

- `io_artifacts.py`
- `layouts.py`

### Typed contracts

- `types.py`
- `selection_contracts.py`
- `calibration_frame.py`

## Current Product Outputs vs Debug Outputs

Match board:

- top-level:
  - one board
  - one summary
- debug:
  - candidate JSON only when requested

Professor triptych:

- top-level:
  - three figures
  - one summary
- debug:
  - selection JSONs
  - optional full turntable/reprojection bundles

Stereo-order registration:

- top-level:
  - one board
  - one summary
  - optional closeup only when requested
- debug:
  - selection/overview bundle only when requested

Turntable compare:

- top-level:
  - current workflow still intentionally writes a richer artifact set
- debug:
  - per-camera/object/ROI diagnostics remain under `debug/`

## Current Pain Points Still Visible Before Cleanup

- `turntable_compare.py` is still the largest mixed-responsibility workflow file
- `professor_triptych.py` still owns a substantial amount of reprojection/truth-board composition logic
- turntable remains the richest artifact family and therefore the easiest place for output/schema drift
- some older metric JSON families remain workflow-specific even after typed summary contracts were added
