# 2026-04-07 Camera-Aware Single-Frame Turntable

## Goal

Add a new single-frame native-vs-FFS fused-cloud comparison workflow that:

- visualizes the 3 real calibrated camera poses explicitly
- renders comparisons from view trajectories anchored near those real camera poses
- makes a static keyframe sheet the primary output
- keeps the existing temporal fused comparison workflow intact for secondary use

## Non-Goals

- do not replace existing panel, reprojection, or temporal fused workflows
- do not duplicate existing aligned-case loading or depth deprojection logic
- do not broaden repo scope beyond aligned-case comparison visualization

## Current Repo State Being Reused

- `data_process/visualization/pointcloud_compare.py`
  - case discovery
  - single-frame fused cloud loading via `load_case_frame_cloud`
  - crop helpers
  - fallback renderer
- `data_process/visualization/calibration_io.py`
  - `calibrate.pkl` loading and serial-order mapping
- `scripts/harness/visual_compare_depth_video.py`
  - current user-facing fused-cloud entrypoint

## Architecture Changes

1. Add a reusable camera-frusta helper module driven by real `c2w` transforms.
2. Add a new `turntable_compare.py` workflow module for:
   - single-frame case selection
   - world-space crop / ROI center selection
   - per-camera orbit generation anchored near real camera positions
   - 2x3 board generation
   - keyframe sheet assembly
   - scene overview rendering with camera poses
3. Add a separate harness entrypoint for the new workflow.
4. Keep current temporal compare code available, but not primary for this task.

## Validation Plan

Deterministic:

- add tests for:
  - camera frustum extraction from `c2w`
  - single-frame same-case vs two-case case selection
  - orbit generation near real camera anchors
  - 2x3 board layout / keyframe sheet output
- run `python scripts/harness/check_all.py`

Practical:

- update `docs/generated/depth_visualization_validation.md` with the new single-frame workflow command and outputs

## Risks

- the fallback renderer is point-based, so pose/frustum overlays must stay legible without relying on Open3D
- orthographic and perspective projections need identical semantics between native and FFS tiles
- crop selection must keep the tabletop readable without hiding the camera-pose context

## Acceptance Criteria

- a new single-frame camera-aware compare script exists
- real camera poses are visualized in an overview artifact
- per-column views are anchored near real camera positions
- the primary artifact is a static 2x3 keyframe sheet for one selected frame
- optional orbit video output is supported
- existing point-cloud loading logic is reused rather than duplicated
- deterministic tests pass

## Completion Checklist

- [ ] add camera-frusta helper module
- [ ] add turntable compare workflow module
- [ ] add new harness script
- [ ] add overview + board + keyframe sheet outputs
- [ ] add deterministic tests
- [ ] update docs
- [ ] run deterministic validation

