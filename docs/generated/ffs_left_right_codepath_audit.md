# FFS Left/Right Codepath Audit

Date: 2026-04-10

## Scope

This audit traces the IR stereo left/right codepath from capture through alignment into Fast-FoundationStereo inference.

It does **not** change:

- `calibrate.pkl` semantics
- calibration serial mapping
- the default production `run_pair(left, right, K_ir_left, baseline_m)` path

## End-to-End Codepath

### 1. Capture

IR stream ordering is explicit in:

- [single_realsense.py](/c:/Users/zhang/proj-QQTT/qqtt/env/camera/realsense/single_realsense.py)

Key points:

- `rs.stream.infrared, 1` is enabled only when `enable_ir_left=True`
- `rs.stream.infrared, 2` is enabled only when `enable_ir_right=True`
- raw frames are read as:
  - `raw_frameset.get_infrared_frame(1)` -> `ir_left`
  - `raw_frameset.get_infrared_frame(2)` -> `ir_right`

So capture-time naming is currently explicit and consistent:

- `ir_left` = RealSense infrared stream index `1`
- `ir_right` = RealSense infrared stream index `2`

### 2. Recording Metadata

IR stereo geometry is recorded in:

- [single_realsense.py](/c:/Users/zhang/proj-QQTT/qqtt/env/camera/realsense/single_realsense.py)
- [recording_metadata.py](/c:/Users/zhang/proj-QQTT/qqtt/env/camera/recording_metadata.py)

Per camera, the repo stores:

- `K_ir_left`
- `K_ir_right`
- `T_ir_left_to_right`
- `T_ir_left_to_color`
- `ir_baseline_m`

These are saved into raw-case `metadata.json`.

### 3. Raw Case Folder Layout

Raw recording writes streams under:

- `data_collect/<case_name>/ir_left/<camera_idx>/<step>.png`
- `data_collect/<case_name>/ir_right/<camera_idx>/<step>.png`

This happens in:

- [camera_system.py](/c:/Users/zhang/proj-QQTT/qqtt/env/camera/camera_system.py)

### 4. Alignment

Alignment copies the raw IR streams forward when present:

- [record_data_align.py](/c:/Users/zhang/proj-QQTT/data_process/record_data_align.py)

Relevant behavior:

- `RAW_IMAGE_STREAMS = ("color", "ir_left", "ir_right")`
- aligned cases preserve:
  - `ir_left/<camera_idx>/<frame>.png`
  - `ir_right/<camera_idx>/<frame>.png`
- aligned `metadata.json` also carries:
  - `K_ir_left`
  - `K_ir_right`
  - `T_ir_left_to_right`
  - `T_ir_left_to_color`
  - `ir_baseline_m`

### 5. FFS Inference During Alignment

The exact FFS callsite is:

- [record_data_align.py](/c:/Users/zhang/proj-QQTT/data_process/record_data_align.py)

Current production call:

- load `ir_left/<camera_idx>/<step>.png` into `left_image`
- load `ir_right/<camera_idx>/<step>.png` into `right_image`
- call:
  - `runner.run_pair(left_image, right_image, K_ir_left=..., baseline_m=...)`

So the current production inference path is still:

- `left = ir_left`
- `right = ir_right`
- intrinsics = `K_ir_left`
- baseline = `ir_baseline_m`

### 6. Harness Proof-of-Life Path

The saved-pair harness follows the same naming:

- [run_ffs_on_saved_pair.py](/c:/Users/zhang/proj-QQTT/scripts/harness/run_ffs_on_saved_pair.py)

It loads:

- `ir_left.png`
- `ir_right.png`

and passes them directly into:

- `runner.run_pair(left_image, right_image, K_ir_left, baseline_m)`

## Current Interpretation

### What looks nominally correct

The repo currently appears **nominally correct** in codepath terms:

- capture naming is explicit
- raw folder naming is explicit
- aligned-case propagation is explicit
- FFS inference currently consumes `ir_left` as left and `ir_right` as right
- metadata keys are consistently named around `left`

### What ambiguity remained before this task

Before this audit pass, the repo still lacked an explicit normal-vs-swapped check.

Also, `FastFoundationStereoRunner.run_pair(...)` clipped disparity to nonnegative before exposing any audit information, which meant:

- a suspicious wrong-order run could be partially hidden
- there was no first-class raw disparity sign summary

## Conclusion

Codepath tracing does **not** reveal an obvious accidental left/right rename or swap.

However, codepath tracing alone is not enough to prove the ordering is correct in practice.

That is why the repo now also includes:

- `scripts/harness/audit_ffs_left_right.py`

which runs the same stereo pair in:

- normal order: `(ir_left, ir_right)`
- swapped order: `(ir_right, ir_left)`

and compares them explicitly.
