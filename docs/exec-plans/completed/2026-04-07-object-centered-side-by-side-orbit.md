# 2026-04-07 Object-Centered Side-by-Side Orbit

## Goal

Refactor the existing single-frame turntable comparison so the default professor-facing
result becomes:

- one selected aligned frame
- one native fused cloud and one FFS fused cloud
- a full 360-degree object-centered orbit around the world-space ROI
- a large left-right side-by-side compare
- a much more legible overview showing the real camera frusta and the current orbit camera
- automatic dual outputs:
  - geometry diagnostic video + keyframe sheet
  - RGB reference video + keyframe sheet

## Non-Goals

- do not remove the current temporal fused workflow
- do not duplicate aligned-case point-cloud loading
- do not remove the existing camera-neighborhood compare path; keep it as secondary

## Current Repo State Being Reused

- `data_process/visualization/turntable_compare.py`
  - single-frame case selection
  - world-space crop logic reuse
  - camera-frusta overlay support
- `data_process/visualization/pointcloud_compare.py`
  - fused point-cloud loading
  - fallback renderer
  - projection/view math
- `scripts/harness/visual_compare_turntable.py`
  - existing user-facing single-frame entrypoint

## Architecture Changes

1. Make object-centered 360 orbit the default turntable mode.
2. Make large side-by-side Native vs FFS panels the default layout.
3. Add automatic dual render planning for geometry + RGB outputs in one run.
4. Enlarge and enrich the overview with:
   - real camera frusta
   - orbit path ring
   - current orbit camera
   - ROI / crop box
5. Keep the old 2x3 near-camera board as a secondary mode only.

## Validation Plan

Deterministic:

- add / update tests for:
  - object-centered orbit generation
  - synchronized native/ffs path pairing
  - overview inset generation
  - dual-output planning
  - keyframe sheet generation
- run `python scripts/harness/check_all.py`

Practical:

- render the new default workflow on:
  - `data/native_30_static`
  - `data/ffs_30_static`
- record exact command and output paths under `docs/generated/depth_visualization_validation.md`

## Risks

- larger default videos increase runtime and output size
- overview readability depends on the fallback renderer remaining legible at reduced inset scale
- full-360 default orbit needs a stable start azimuth tied to the real camera layout

## Acceptance Criteria

- default turntable output is large side-by-side Native vs FFS
- default orbit is 360 degrees around the cropped ROI center
- overview clearly shows real camera positions and the orbit camera
- one run automatically writes geometry and RGB videos + keyframe sheets
- native and FFS use the exact same orbit path
- old 2x3 near-camera compare remains optional/secondary
- deterministic tests pass
