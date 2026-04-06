# 2026-04-06 FFS Depth Backend Integration

## Goal

Integrate Fast-FoundationStereo into the camera-only QQTT repo as an optional
depth backend while preserving the existing RealSense RGB-D path.

Desired supported flows:

- `record_data.py --capture_mode rgbd`
- `data_process/record_data_align.py --depth_backend realsense`

- `record_data.py --capture_mode stereo_ir`
- `data_process/record_data_align.py --depth_backend ffs`

- `record_data.py --capture_mode both_eval`
- `data_process/record_data_align.py --depth_backend both`

The repo still stops at `record_data_align.py`.

## Non-Goals

- do not reintroduce downstream perception, Gaussian Splatting, simulation, evaluation, or teleop
- do not vendor Fast-FoundationStereo into the repo
- do not duplicate proof-of-life geometry code across production and harness

## Current Repo State

- camera-only repo boundary already established
- proof-of-life FFS scripts exist under `scripts/harness/`
- single-camera D455 proof-of-life is complete
- stream capability probe is complete
- current recording pipeline still only supports color + aligned RealSense depth
- current alignment pipeline still only supports canonical RealSense depth copying

## Current Proof-of-Life Assets To Reuse

- `scripts/harness/ffs_geometry.py`
- `scripts/harness/probe_d455_ir_pair.py`
- `scripts/harness/run_ffs_on_saved_pair.py`
- `scripts/harness/reproject_ffs_to_color.py`
- `scripts/harness/probe_d455_stream_capability.py`
- generated D455 probe and proof-of-life docs under `docs/generated/`

## Integration Architecture

1. Promote FFS geometry and runner logic into shared production modules under
   `data_process/depth_backends/`.
2. Update harness scripts to import the shared production implementation.
3. Extend recording to support optional raw `ir_left` / `ir_right` capture.
4. Keep alignment as the place where FFS depth is generated.
5. Preserve canonical `depth/<cam>/<frame>.npy` compatibility semantics.

## Metadata Contract Changes

Raw recording metadata must preserve old semantics where practical and add:

- `schema_version`
- `logical_camera_names`
- `capture_mode`
- `streams_present`
- `camera_model_per_camera`
- `product_line_per_camera`
- `K_color`
- `K_ir_left`
- `K_ir_right`
- `T_ir_left_to_right`
- `T_ir_left_to_color`
- `ir_baseline_m`
- `depth_scale_m_per_unit`
- `depth_encoding`
- `alignment_target`
- `depth_coordinate_frame`
- emitter request / actual

Aligned metadata must add:

- `depth_backend_used`
- `depth_source_for_depth_dir`
- optional `depth_ffs` metadata
- FFS config fields when applicable

## CLI Changes

Recording:

- `--capture_mode rgbd|stereo_ir|both_eval`
- optional `--emitter on|off|auto`
- optional non-interactive frame-limited recording support if needed for validation

Alignment:

- `--depth_backend realsense|ffs|both`
- `--ffs_repo`
- `--ffs_model_path`
- `--ffs_scale`
- `--ffs_valid_iters`
- `--ffs_max_disp`
- `--write_ffs_float_m`
- `--fail_if_no_ir_stereo`

## Validation Plan

Deterministic:

- extend `scripts/harness/check_all.py`
- add synthetic alignment tests for `ffs` and `both`
- add metadata schema tests

Hardware:

- validate default `rgbd -> realsense`
- validate `stereo_ir -> ffs`
- validate `both_eval -> both` only if the machine supports it

## Risks

- three-camera `rgb_ir_pair` probe was not stable on this machine, so `stereo_ir`
  may exist as an integrated mode without being production-stable here
- three-camera `rgbd_ir_pair` probe was not stable, so `both_eval` may remain unsupported here
- recording metadata expansion must stay backward-compatible enough for existing readers
- FFS runtime still depends on an external repo path and compatible local GPU env

## Acceptance Criteria

- old RGB-D path still works
- new `stereo_ir` capture mode exists
- alignment supports `realsense|ffs|both`
- FFS uses an in-process production runner
- FFS output is explicitly reprojected from IR-left to color coordinates
- canonical compatibility encoding for `depth/` is preserved
- harness scripts reuse shared production logic
- docs/tests/hardware validation are updated honestly

## Completion Checklist

- [x] add production `data_process/depth_backends/` modules
- [x] refactor harness scripts to reuse production modules
- [x] extend recording stack with capture modes and richer metadata
- [x] extend alignment with `realsense|ffs|both`
- [x] add deterministic tests
- [x] update docs
- [x] run deterministic validation
- [x] run real hardware validation
- [x] move this plan to `docs/exec-plans/completed/`

## Progress Log

- 2026-04-06: promoted FFS geometry and runner logic into `data_process/depth_backends/`
- 2026-04-06: updated harness scripts to reuse production geometry / runner code
- 2026-04-06: added `rgbd`, `stereo_ir`, and `both_eval` recording modes with richer raw metadata
- 2026-04-06: added `realsense`, `ffs`, and `both` alignment backends with explicit IR-left to color reprojection
- 2026-04-06: added deterministic tests for recording metadata schema and FFS / both alignment behavior
- 2026-04-06: validated `rgbd -> realsense` on the current 3-camera D455 setup
- 2026-04-06: validated `stereo_ir -> ffs` on D455 serial `239222300781`
- 2026-04-06: confirmed `both_eval` remains blocked by the measured D455 stream capability probe on this machine

## Completion Summary

This integration completed successfully with the following grounded outcome:

- default RealSense RGB-D path remains available
- optional FFS depth generation is now integrated at alignment time
- canonical `depth/` compatibility semantics are preserved
- harness proof-of-life logic now shares production modules instead of duplicating geometry code
- `both_eval` is implemented as an experimental gated mode and is honestly blocked on this machine
