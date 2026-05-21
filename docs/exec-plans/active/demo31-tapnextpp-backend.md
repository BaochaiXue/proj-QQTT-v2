# Demo 3.1 TAPNext++ Backend

Status: active

## Goal

Add TAPNext++ as a Demo 3.1 point-tracker backend under the existing child-process
PointTrackerAdapter contract. Keep GPU0 responsible for RealSense, masks, depth
fusion, and rendering. Keep GPU1 responsible for CPU RGB/mask input packets and
CPU 2D track/visibility output packets.

The profiling target for this task is explicitly `4096` query points per camera,
which means `12288` total query points across the three Demo 3.1 views.

## Scope

1. Add a `tapnextpp` point-tracker backend and aliases.
2. Add TAPNext++ adapter support for serial and true batch-views.
3. Add Demo 3.1 CLI, dry-run contract, and child-process config wiring.
4. Add an installer for `demo_3_1_max` that does not replace CUDA Torch.
5. Add fake-model tests proving online serial and batch-views call shapes.
6. Add profiling harnesses focused on `4096/view` rendered live Demo 3.1.

## Non-Goals

- Do not move depth, intrinsics, c2w, or world lifting into TAPNext++.
- Do not replace the default backend.
- Do not run or optimize `1365/view` as the primary target.
- Do not silently fall back from batch-views to serial.

## Verification

- Focused unit tests for point tracker adapters and Demo 3.1 dry-run contracts.
- `scripts/env/install_tapnextpp_demo_3_1_max.sh --help`.
- Real TAPNext++ import/checkpoint smoke in `demo_3_1_max` when the install
  script is run.
- Rendered profiling for:
  - `tapnextpp`, `serial`, `4096/view`
  - `tapnextpp`, `batch-views`, `4096/view`

## Implementation Notes

- `tapnextpp` is registered as a `PointTrackerAdapter`, not a new runtime.
- Serial mode owns one `TAPNextPPAdapter` per camera.
- Batch-views mode owns one `TAPNextPPAdapter` with a single TAPNext++ state
  over camera batch dimension `B=3`.
- Adapter I/O stays CPU RGB/mask in and CPU 2D tracks/visibility out; the main
  process still owns RealSense depth, intrinsics, c2w, and world lift.

## Verification Evidence

- PASS: `python -m py_compile qqtt/tracking/backends/tapnextpp_adapter.py qqtt/tracking/backends/point_tracker_adapter.py qqtt/demo/demo31_cotracker_process.py qqtt/demo/demo31_runtime.py tests/test_point_tracker_adapters.py tests/test_demo31_cotracker_process_config.py tests/test_demo31_dual_gpu_contract.py`
- PASS: `conda run --no-capture-output -n demo_3_1_max python -m unittest tests.test_point_tracker_adapters tests.test_demo31_cotracker_process_config tests.test_demo31_dual_gpu_contract`
- PASS: `scripts/env/install_tapnextpp_demo_3_1_max.sh --help`
- PASS: Demo 3.1 dry-run with `--cotracker-backend tapnextpp --tracking-backend-execution-mode serial --cotracker-query-count 4096`.
- PASS: Demo 3.1 dry-run with `--cotracker-backend tapnextpp --tracking-backend-execution-mode batch-views --cotracker-query-count 4096`.
- PASS: Real adapter CUDA smoke on tracker GPU with TAPNext++ checkpoint:
  serial `B=1,N=4` and batch-views `B=3,N=4`.
- PASS: `conda run --no-capture-output -n demo_3_1_max python scripts/harness/check_all.py`.

Not run yet: rendered live RealSense/Open3D profile for `4096/view`; this
requires the physical camera/render session and is the next measurement step.
