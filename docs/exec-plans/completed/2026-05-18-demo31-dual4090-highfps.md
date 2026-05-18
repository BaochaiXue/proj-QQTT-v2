# Demo 3.1 Dual-4090 High-FPS Runtime

## Goal

Clone the Demo 3.0 RealSense CoTracker overlay lineage into a new Demo 3.1
dual-GPU visualization runtime. GPU0 owns three RealSense capture, SAM3.1/HF
EdgeTAM masks, RealSense-depth fusion, and rendering; GPU1 owns CoTracker3
online in a separate process with latest-wins CPU IPC.

## Scope

- Add Demo 3.1 entrypoint, runtime facade, IPC helpers, CoTracker process
  config/launcher helpers, docs, and tests.
- Preserve Demo 3.0 behavior except for reusable helpers.
- Keep Demo 3.1 in sanctioned realtime demo/tracking diagnostic scope.
- Do not introduce FFS into Demo 3.1.
- Do not transfer CUDA tensors between processes.
- Set `CUDA_VISIBLE_DEVICES` before importing GPU runtimes in the Demo 3.1
  entrypoint and CoTracker process path.

## Plan

1. Inspect Demo 3 runtime, CoTracker worker, and existing Demo 3 tests.
2. Add lightweight Demo 3.1 config/contract parsing and dry-run path.
3. Add latest-wins CPU IPC dataclasses and queue helpers.
4. Add CoTracker subprocess config/launcher module that isolates GPU1 via
   environment variables before backend imports.
5. Add a Demo 3.1 runtime facade that reuses Demo 3 validation semantics where
   practical, rejects FFS, reports dual-GPU mapping, and keeps renderer wait
   semantics explicit.
6. Add docs and focused unit tests for contract, IPC, process config, and Demo
   3.0 non-regression.
7. Run deterministic checks in `demo3-max` and commit/push the branch.

## Validation Targets

- `conda run --no-capture-output -n demo3-max python -m unittest -v tests.test_demo31_dual_gpu_contract tests.test_demo31_ipc_latest_wins tests.test_demo31_cotracker_process_config tests.test_demo3_contract tests.test_demo3_cotracker_worker`
- `conda run --no-capture-output -n demo3-max python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda --calibrate-path calibrate.pkl`
- `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --camera-ids 0,1,2`
- `conda run --no-capture-output -n demo3-max python scripts/harness/check_all.py`

## Results

- Added `demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py`
  with pre-import `CUDA_VISIBLE_DEVICES=<mask_gpu>` setup for the main process.
- Added `qqtt/demo/demo31_runtime.py` with Demo 3.1 CLI, dry-run contract,
  dual-GPU validation, FFS rejection, shared-runtime adapter, CPU-only
  CoTracker input packets, and main-process world lift of 2D tracks.
- Added `qqtt/demo/demo31_dual_gpu_ipc.py` with CPU latest-wins queue helpers,
  RGB/mask-only tracking packets, and strict/latest-reuse mask policy helpers.
- Added `qqtt/demo/demo31_cotracker_process.py` with CoTracker process config,
  environment isolation, spawn-based child process loop, and a no-torch
  top-level module contract.
- Added `qqtt/demo/demo31_profile.py` profile summary helpers.
- Added `docs/demo31_dual4090_runtime_contract.md` and `demo_v3_1/README.md`.
- Updated `docs/ARCHITECTURE.md`, `docs/WORKFLOWS.md`, `scripts/harness/check_all.py`,
  and `tests/test_check_all_smoke.py` for Demo 3.1 registration.
- Added tests:
  - `tests.test_demo31_dual_gpu_contract`
  - `tests.test_demo31_ipc_latest_wins`
  - `tests.test_demo31_cotracker_process_config`
- PASS:
  `conda run --no-capture-output -n demo3-max python -m py_compile demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py qqtt/demo/demo31_runtime.py qqtt/demo/demo31_dual_gpu_ipc.py qqtt/demo/demo31_cotracker_process.py qqtt/demo/demo31_profile.py`
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda --calibrate-path calibrate.pkl`
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --camera-ids 0,1,2`
- PASS:
  `conda run --no-capture-output -n demo3-max python -m unittest -v tests.test_demo31_dual_gpu_contract tests.test_demo31_ipc_latest_wins tests.test_demo31_cotracker_process_config tests.test_demo3_contract tests.test_demo3_cotracker_worker`
- PASS:
  `conda run --no-capture-output -n demo3-max python scripts/harness/check_all.py`
  quick profile, including 273 unittest tests.
- Note: unrelated local changes under the Demo 2.3 / CoTracker full-weights
  worklines were present and intentionally excluded from this commit.
