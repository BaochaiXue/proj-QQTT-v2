# 2026-04-05 D455 FFS Proof-of-Life

## Goal

Build a single-camera proof-of-life chain for using external Fast-FoundationStereo (FFS)
with one Intel RealSense D455 inside the camera-only QQTT repo boundary.

The chain must verify:

1. external FFS weights are present
2. the official FFS demo works end-to-end
3. one D455 can provide a saved IR-left / IR-right pair plus geometry metadata
4. FFS can run on that saved pair
5. the FFS result can be converted into:
   - IR-left metric depth
   - color-aligned compatible depth when color metadata is present

## Non-Goals

- do not modify `record_data.py`
- do not modify `data_process/record_data_align.py`
- do not start multi-camera FFS integration
- do not claim same-take RGB + depth + IR-left + IR-right is supported
- do not reintroduce downstream perception, simulation, or rendering code

## Local Path Assumptions

- QQTT repo root: `C:\Users\zhang\proj-QQTT`
- External FFS repo: `C:\Users\zhang\external\Fast-FoundationStereo`
- FFS env: `ffs-standalone`
- QQTT + FFS env: `qqtt-ffs-compat`
- Primary D455 serial for proof-of-life: `239222303506`

## Checkpoint Strategy

- Validate one baseline checkpoint only:
  - `weights/23-36-37/model_best_bp2_serialize.pth`
- Prefer proof-of-life over benchmark breadth
- Save all generated validation notes under `docs/generated/`
- Keep runtime artifacts under local `data/ffs_proof_of_life/`

## Single-Camera Validation Plan

1. Verify the external checkpoint file and config exist.
2. Run the official FFS demo in `ffs-standalone`.
3. Probe one D455 directly with a raw `pyrealsense2` pipeline.
4. Save:
   - `ir_left.png`
   - `ir_right.png`
   - optional `color.png`
   - runtime intrinsics / extrinsics / scale metadata
5. Run FFS on the saved pair.
6. Convert disparity into metric depth in IR-left coordinates.
7. Reproject IR-left depth into color coordinates when color metadata exists.
8. Quantize color-aligned depth into replacement-compatible invalid=0 encoding.

## Geometry / Scale Contract

- Stereo source is one D455 internal pair only.
- IR-left is `infrared` stream index `1`.
- IR-right is `infrared` stream index `2`.
- Raw IR frames must come from the unaligned frameset.
- Initial FFS output lives in IR-left coordinates.
- Color-aligned output requires explicit IR-left to color reprojection.
- Runtime intrinsics and runtime extrinsics from the SDK are authoritative.
- Baseline is derived from runtime `T_ir_left_to_right`, not a hardcoded nominal D455 value.
- Replacement-compatible depth encoding preserves invalid=`0`.
- `depth_scale_m_per_unit` must be recorded explicitly when available.

## Risks

- Official FFS demo is Linux-centric and uses shell / GUI behavior that may fail on Windows.
- FFS may require CUDA and a functioning GPU runtime in `ffs-standalone`.
- Single-camera RGB + IR-left + IR-right may require fallback to 640x480 or IR-only.
- Color-aligned output is blocked if the probe cannot save color from the same sample.

## Acceptance Criteria

- Checkpoint and config exist at documented local paths.
- Official FFS demo produces expected artifacts.
- One D455 sample with IR-left / IR-right is saved with metadata.
- FFS runs on that saved pair.
- Reprojection produces IR-left metric depth and, when possible, color-aligned depth.
- Software-only tests exist for intrinsic formatting, reprojection, and quantization.
- Main QQTT workflow files remain untouched.

## Completion Checklist

- [x] verify checkpoint files
- [x] validate official FFS demo
- [x] add D455 single-camera probe
- [x] save one D455 proof-of-life sample
- [x] run FFS on the saved pair
- [x] reproject to color and quantize compatible depth
- [x] add software-only tests
- [x] update deterministic checks / CI
- [x] move this plan to `docs/exec-plans/completed/`

## Progress Log

- 2026-04-05: created active execution plan and fixed scope for the proof-of-life task
- 2026-04-05: verified external checkpoint `23-36-37` and documented its local path
- 2026-04-05: validated the official FFS demo in `ffs-standalone`
- 2026-04-05: upgraded `ffs-standalone` and `qqtt-ffs-compat` from `torch==2.6.0+cu124` to `torch==2.7.0+cu128` because local `sm_120` hardware could not execute the original build
- 2026-04-05: captured a raw D455 IR-left / IR-right / color sample from serial `239222303506`
- 2026-04-05: ran FFS on the saved pair and saved disparity + IR-left metric depth artifacts
- 2026-04-05: reprojected the FFS result into color coordinates and wrote replacement-compatible `invalid=0` depth encoding
- 2026-04-05: added software-only geometry tests and extended deterministic checks

## Completion Summary

This proof-of-life completed successfully for a single D455.

- Official Fast-FoundationStereo demo: passed
- Single-camera D455 raw IR probe: passed
- Saved-pair FFS inference: passed
- IR-left metric depth conversion: passed
- Color-aligned compatible depth conversion: passed

Main QQTT workflow files remained untouched in this phase.
