# Demo 3.1 LocoTrack-S Backend

Status: active

## Goal

Add LocoTrack-S as the fourth Demo 3.1 point-tracker backend through the existing
CPU RGB/mask point-tracker child-process contract. Demo 3.1 keeps depth,
intrinsics, camera-to-world transforms, 2D-to-world lift, fusion, and render in
the main/GPU0 path; LocoTrack runs only as a GPU1 point-tracker adapter that
returns CPU 2D track and visibility packets.

## Scope

1. Add a LocoTrack-S install script for the existing `demo_3_1_max` conda env.
2. Add `locotrack` to `qqtt/tracking/backends` as a normal `PointTrackerAdapter`.
3. Wire Demo 3.1 CLI, dry-run contract, and child-process JSON config fields.
4. Cover serial and batch-views behavior with deterministic fake-model tests.
5. Add rendered profiling harness scripts for LocoTrack serial vs batch-views.

## Non-Goals

- Do not move RealSense depth, intrinsics, c2w, or world lift into LocoTrack.
- Do not add a new runtime process; reuse the existing point-tracker child
  process and adapter factory.
- Do not make Demo 3.2 FFS/LiteTracker changes except where shared adapter tests
  require neutral behavior.
- Do not install or replace the existing CUDA Torch in `demo_3_1_max`.
- Do not claim 4000 points / 45 FPS as measured until the rendered profiling
  harness has real pointcloud evidence.

## Implementation Notes

- Backend name: `locotrack`; aliases include `loco_track`, `locotrack_s`,
  `loco-track`, and `loco-track-s`.
- Model size defaults to `small`; checkpoint defaults are explicit CLI/env paths
  only. No download at adapter import or load time.
- Serial mode is rolling-window inference per camera.
- Batch-views mode is one adapter, one model instance, one model call over
  `[B,T,H,W,3]` for the complete three-camera update.
- LocoTrack output `tracks` are xy; QQTT `TrackingResult` stores yx.
- LocoTrack `occlusion=True` means invisible; QQTT visibility is float32
  `~occlusion`.

## Verification Plan

- `scripts/env/install_locotrack_s_demo_3_1_max.sh --help`
- `python -m py_compile` for modified runtime/backend/harness scripts.
- Focused unit tests:
  - `tests.test_point_tracker_adapters`
  - `tests.test_demo31_cotracker_process_config`
  - `tests.test_demo31_dual_gpu_contract`
- If focused tests pass, run `scripts/harness/check_all.py` in the active demo
  env when available.

## Verification Evidence

- PASS: `python -m py_compile qqtt/tracking/backends/point_tracker_adapter.py qqtt/tracking/backends/locotrack_adapter.py qqtt/demo/demo31_runtime.py qqtt/demo/demo31_cotracker_process.py qqtt/demo/services/profile_schema.py scripts/harness/run_demo31_locotrack_s_profiles.py scripts/harness/summarize_demo31_locotrack_s_profiles.py tests/test_point_tracker_adapters.py tests/test_demo31_dual_gpu_contract.py tests/test_demo31_cotracker_process_config.py`
- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_point_tracker_adapters tests.test_demo31_cotracker_process_config tests.test_demo31_dual_gpu_contract tests.test_profile_schema`
- PASS: `scripts/env/install_locotrack_s_demo_3_1_max.sh --help`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_harness_catalog.py`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/run_demo31_locotrack_s_profiles.py --help`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/summarize_demo31_locotrack_s_profiles.py --help`
- PASS: Demo 3.1 dry-run with `--cotracker-backend locotrack --tracking-backend-execution-mode serial` and `QQTT_DEMO31_TEST_CUDA_COUNT=2`.
- PASS: Demo 3.1 dry-run with `--cotracker-backend locotrack --tracking-backend-execution-mode batch-views` and `QQTT_DEMO31_TEST_CUDA_COUNT=2`.
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`

Not run: full LocoTrack install/checkpoint load, to avoid mutating the local
conda env and cloning external sources during this code pass. The new install
script owns that explicit operator step.
