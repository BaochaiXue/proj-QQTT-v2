# Demo 3.2 Table-World PCD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Demo 3.2 consume `table_calibrate.pkl` so every generated, saved, and rendered point cloud is in single-camera table/world coordinates.

**Architecture:** Follow FuturePhysTwin preprocessing: lift depth to camera-space xyz, immediately apply the calibrated `camera_to_world_c2w`, then let downstream render/headless/offline paths consume world-space PCD artifacts. The wrapper validates and forwards `--table-calibrate`; the shared delegate loads the transform once after runtime startup and applies it in PCD and tracker 3D lift boundaries.

**Tech Stack:** Python, NumPy, RealSense/FFS Demo 3.x runtime, existing `qqtt.env.camera.table_calibration` loader, unittest harness.

---

### Task 1: Forward Calibration Contract To Delegate

**Files:**
- Modify: `qqtt/demo/single_demo_v3_runtime.py`
- Test: `tests/test_single_demo_v3_runtime.py`

- [x] Add `--table-calibrate <path>` to the live delegate argv when the wrapper validated a table calibration path.
- [x] Update the Demo 3.2 dry-run test so the delegate argv contains the validated table calibration path.

### Task 2: Load And Apply Table-World Transform In Shared Runtime

**Files:**
- Modify: `qqtt/demo/realtime_masked_edgetam_pcd.py`
- Test: `tests/test_single_demo_tapnextpp_overlay.py`

- [x] Add shared delegate CLI `--table-calibrate`.
- [x] Load `camera_to_world_c2w` after the runtime serial is known, validating that the calibration serial matches the active camera.
- [x] Add small helpers to transform Nx3 arrays from camera frame to table/world frame.
- [x] Apply the transform to `MaskedPcdPacket.controller_xyz_m` and `MaskedPcdPacket.object_xyz_m` after enhanced-pt/sync filtering, so saved/rendered PCD is calibrated.
- [x] Apply the same transform to `TrackerMarkerPacket.marker_xyz_m` so same-seq tracking markers remain in the same coordinate frame as PCD.
- [x] Add metadata/contract fields: `pcd_coordinate_frame`, `table_calibration_path`, `table_world_frame_kind`, `table_z_m`.

### Task 3: Verification

**Files:**
- Test: `tests/test_single_demo_v3_runtime.py`
- Test: `tests/test_single_demo_tapnextpp_overlay.py`

- [x] Run focused unit tests for runtime argv/contract and PCD transform helpers.
- [x] Run Demo 3.2 dry-run with `--table-calibrate table_calibrate.pkl` and confirm the delegate/contract says `table_world_z0`.
- [x] Run the smoke validation profile if focused tests pass.
