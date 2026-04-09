# Repo Hardening Inventory

Date: 2026-04-09

## Current User-Facing CLI Entrypoints

Primary product entrypoints:

- `cameras_viewer.py`
- `cameras_calibrate.py`
- `record_data.py`
- `data_process/record_data_align.py`

Comparison harness entrypoints:

- `scripts/harness/visual_compare_depth_panels.py`
- `scripts/harness/visual_compare_reprojection.py`
- `scripts/harness/visual_compare_depth_video.py`
- `scripts/harness/visual_compare_turntable.py`

Repo checks:

- `scripts/harness/check_scope.py`
- `scripts/harness/check_visual_architecture.py`
- `scripts/harness/check_all.py`

## Current Record / Alignment / Compare Workflows

### Record

- `record_data.py`
- current capture modes:
  - `rgbd`
  - `stereo_ir`
  - `both_eval`

### Align

- `data_process/record_data_align.py`
- current depth backends:
  - `realsense`
  - `ffs`
  - `both`

### Compare

- per-camera local depth quality:
  - `visual_compare_depth_panels.py`
- cross-view reprojection consistency:
  - `visual_compare_reprojection.py`
- older temporal fused-cloud compare:
  - `visual_compare_depth_video.py`
- current professor-facing single-frame compare:
  - `visual_compare_turntable.py`

## Current Visualization Module Map

Shared low-level modules:

- `data_process/visualization/calibration_frame.py`
- `data_process/visualization/calibration_io.py`
- `data_process/visualization/io_case.py`
- `data_process/visualization/io_artifacts.py`
- `data_process/visualization/roi.py`
- `data_process/visualization/views.py`
- `data_process/visualization/layouts.py`
- `data_process/visualization/types.py`
- `data_process/visualization/renderers/fallback.py`

Specialized diagnostics:

- `data_process/visualization/object_roi.py`
- `data_process/visualization/object_compare.py`
- `data_process/visualization/support_compare.py`
- `data_process/visualization/source_compare.py`
- `data_process/visualization/depth_diagnostics.py`
- `data_process/visualization/camera_frusta.py`

Workflow-facing modules:

- `data_process/visualization/panel_compare.py`
- `data_process/visualization/reprojection_compare.py`
- `data_process/visualization/pointcloud_compare.py`
- `data_process/visualization/turntable_compare.py`
- `data_process/visualization/workflows/merge_diagnostics.py`
- `data_process/visualization/workflows/depth_panels.py`
- `data_process/visualization/workflows/reprojection_compare.py`
- `data_process/visualization/workflows/turntable_compare.py`

## Current Calibration Contract

`calibrate.pkl` is currently treated as:

- list / tuple / ndarray of shape `(N, 4, 4)`
- each transform is `camera -> world` (`c2w`)
- the world frame is the raw ChArUco/board calibration frame
- subset captures rely on `metadata["calibration_reference_serials"]` for serial-order mapping

The compare stack now makes this explicit through:

- `data_process/visualization/calibration_io.py`
- `data_process/visualization/calibration_frame.py`

Current compare behavior:

- world-space geometry uses raw calibration-world coordinates
- no semantic-world transform is currently applied by default
- turntable metadata now records this explicitly

## Current Metadata Contract

Recording metadata:

- built by `qqtt/env/camera/recording_metadata.py`
- schema version: `qqtt_recording_v2`
- includes:
  - `serial_numbers`
  - `calibration_reference_serials`
  - `capture_mode`
  - `streams_present`
  - color/IR intrinsics
  - IR extrinsics
  - depth encoding metadata
  - per-camera exposure/gain/white-balance

Aligned-case metadata:

- used by visualization and alignment workflows
- includes:
  - `serial_numbers`
  - `calibration_reference_serials`
  - `frame_num`
  - `K_color`
  - `depth_scale_m_per_unit`

## Current Preflight / Probe Decision Logic

Probe source:

- `docs/generated/d455_stream_probe_results.json`

Current policy by capture mode:

- `rgbd`
  - no D455 IR-pair preflight gate
  - allowed directly
- `stereo_ir`
  - probe-aware
  - unsupported profile still allowed experimentally with warning
- `both_eval`
  - probe-aware
  - unsupported profile blocked

Current decision logic is now centralized in:

- `qqtt/env/camera/preflight.py`

`record_data.py` now prints an explicit operator-facing preflight summary before recording proceeds.

## Current Comparison Artifact Outputs

Depth panels:

- per-camera `frames/*.png`
- optional `panels.mp4`
- `summary.json`

Reprojection compare:

- per-pair `frames/*.png`
- optional `reprojection.mp4`
- `summary_metrics.json`

Depth video compare:

- per-view native/ffs/side-by-side frame directories
- optional `native.mp4`
- optional `ffs.mp4`
- optional `side_by_side.mp4`
- optional `grid_2x3.mp4`
- `comparison_metadata.json`
- `metrics.json`

Turntable compare:

- `scene_overview_with_cameras.png`
- `scene_overview_calibration_frame.png`
- `turntable_metadata.json`
- `compare_debug_metrics.json`
- `support_metrics.json`
- `source_metrics.json`
- `mismatch_metrics.json`
- geom / rgb / support / source / mismatch frame directories
- optional geom / rgb / support / source / mismatch mp4/gif outputs
- keyframe sheets
- per-camera mask / object-cloud / fused-object debug artifacts

## Current Test Categories

Record / align:

- recording metadata schema
- alignment smoke tests
- FFS alignment smoke tests

Calibration:

- calibration loader smoke tests
- camera-frustum / camera-pose tests

Visualization workflows:

- depth panels
- reprojection compare
- depth-video compare
- turntable compare

Visualization contracts:

- object ROI / refinement
- source attribution / mismatch / support
- projection orientation
- artifact writing
- layout composition
- import layering
- typed contracts

## Current Architecture Pain Points

1. `turntable_compare.py` is still the largest visualization workflow module.
2. `object_compare.py` still mixes pure ROI/masking logic with debug artifact writing.
3. Compare metadata and frame semantics were previously under-documented.
4. Record preflight policy used to be partially duplicated inside `record_data.py`.
5. Some contract tests were still missing for:
   - preflight policy
   - calibration matrix validity
   - explicit frame semantics
   - object-layer alignment invariants
