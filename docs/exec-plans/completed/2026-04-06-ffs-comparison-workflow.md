# 2026-04-06 FFS Comparison Workflow

## Goal

Finish and harden the current optional Fast-FoundationStereo integration and add a
user-facing native-vs-FFS visual comparison workflow for aligned cases.

## Non-Goals

- do not restart the repo from scratch
- do not reintroduce downstream perception, simulation, Gaussian, or teleop code
- do not replace the RealSense-only path
- do not claim same-take `both_eval` support without hardware evidence

## Current Repo State Being Reused

- `rgbd`, `stereo_ir`, and `both_eval` capture modes already exist
- `realsense`, `ffs`, and `both` alignment backends already exist
- shared FFS production modules already exist in `data_process/depth_backends/`
- proof-of-life and stream capability probe harness code already exists
- deterministic tests already cover probe schema, geometry, quantization, and alignment smoke paths

## Architecture Changes

1. Audit and tighten the current recording / alignment / metadata contracts.
2. Add a robust calibration loader for the repo’s real `calibrate.pkl` schema.
3. Add reusable fused point-cloud comparison modules.
4. Add a user-facing CLI that renders native-vs-FFS comparison frames and videos.
5. Reuse the shared production depth-backend geometry instead of duplicating logic in harness scripts.

## Raw / Aligned Metadata Contract

- raw metadata must remain sufficient for:
  - FFS inference
  - reprojection to color
  - point-cloud generation
  - calibration transform selection
- aligned metadata must explicitly record:
  - `depth_backend_used`
  - `depth_source_for_depth_dir`
  - `depth_encoding`
  - `depth_scale_m_per_unit`
  - `K_color`, `K_ir_left`, `K_ir_right`
  - `T_ir_left_to_color`, `T_ir_left_to_right`
  - `calibration_reference_serials`
  - `ffs_config` when applicable

## Comparison Visualization Plan

- support same-case comparison:
  - one aligned case with `depth/` and `depth_ffs/`
- support two-case fallback comparison:
  - one aligned native-depth case
  - one aligned FFS-depth case
- per frame:
  - decode compatible depth to meters
  - deproject with color intrinsics
  - transform into world with `calibrate.pkl`
  - fuse across cameras
  - render native and FFS clouds from an identical deterministic view
  - save side-by-side frames and optional mp4 / ply outputs

## calibrate.pkl Schema Plan

- inspect the actual producer output and support that real schema only
- document the supported schema and transform convention explicitly
- fail clearly on unsupported / ambiguous schemas

## Validation Plan

Deterministic:

- extend `scripts/harness/check_all.py`
- add software-only tests for:
  - calibration loading
  - point-cloud fusion
  - comparison CLI smoke path

Hardware:

- verify `rgbd -> realsense` still works
- verify `stereo_ir -> ffs` still works
- if `both_eval` remains blocked, validate the comparison CLI in two-case fallback mode

## Risks

- `both_eval` remains probe-gated on the current machine
- `calibrate.pkl` lacks embedded serials, so subset captures must rely on explicit metadata mapping
- offscreen Open3D rendering may not be available everywhere, so a deterministic fallback renderer is required

## Acceptance Criteria

- existing `rgbd -> realsense` path still works
- existing `stereo_ir -> ffs` path still works
- `both_eval` remains honest
- a comparison CLI exists and supports same-case and two-case modes
- calibration loader supports the repo’s real `calibrate.pkl` schema
- deterministic tests pass
- docs and validation notes are updated honestly

## Completion Checklist

- [x] audit and tighten current integration
- [x] add calibration loader
- [x] add fused point-cloud comparison modules
- [x] add comparison CLI
- [x] add deterministic tests
- [x] update docs
- [x] run deterministic validation
- [x] run practical hardware validation
- [x] move this plan to `docs/exec-plans/completed/`

## Progress Log

- 2026-04-06: created active plan for finishing the comparison workflow on top of the existing FFS integration
- 2026-04-06: audited current FFS integration and preserved the existing `rgbd -> realsense` and `stereo_ir -> ffs` paths
- 2026-04-06: added `data_process.visualization.calibration_io` and documented the supported real `calibrate.pkl` schema as camera-to-world transforms
- 2026-04-06: added fused point-cloud comparison production modules and CLI
- 2026-04-06: added deterministic tests for calibration loading, point-cloud fusion, and visual comparison CLI output structure
- 2026-04-06: validated fallback two-case comparison rendering on real aligned native and FFS cases from serial `239222300781`
- 2026-04-06: confirmed `both_eval` still remains blocked on the current machine/profile

## Completion Summary

This step completed by hardening the existing FFS integration and adding a user-facing
comparison visualization workflow without expanding the repo back into downstream pipelines.

Grounded end state:

- `rgbd -> realsense` still works
- `stereo_ir -> ffs` still works
- `both_eval` remains honest and blocked on this machine
- native-vs-FFS visual comparison now works in:
  - same-case mode when `depth_ffs/` exists
  - two-case fallback mode when same-take comparison capture is unavailable
