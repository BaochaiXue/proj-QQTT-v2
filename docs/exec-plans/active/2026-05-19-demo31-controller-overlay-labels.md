# Demo 3.1 Controller Overlay Labels

## Goal

Make Demo 3.1 live CoTracker visualization match the intended PhysTwin-style
exhibition: CoTracker tracks dense object-controller union queries, but the
rendered overlay displays only controller-labeled tracks by default.

## Scope

- Carry first-frame query mask membership through the CoTracker overlay packet.
- Add an `overlay_display_scope` contract with default `controller`.
- Filter displayed overlay points by controller label before applying the
  30-points-per-camera display cap.
- Preserve raw `phystwin_dense` union tracking and object/controller sampling
  statistics.
- Update Demo 3 / Demo 3.1 contracts, process config, docs, and smoke tests.

## Non-Goals

- Do not reduce raw CoTracker query count.
- Do not change SAM3.1, EdgeTAM, RealSense fusion, FFS, or Open3D rendering.
- Do not import FuturePhysTwin code or add a runtime dependency.

## Progress

- Implemented first-frame query label propagation in the CoTracker overlay
  packet and Demo 3.1 process IPC.
- Added `overlay_display_scope`, defaulting to `controller`, while preserving
  raw `phystwin_dense` union query sampling.
- Updated Demo 3 / Demo 3.1 contract docs and targeted smoke tests.

## Validation

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  qqtt/demo/cotracker3_overlay_worker.py \
  qqtt/demo/demo31_dual_gpu_ipc.py \
  qqtt/demo/demo31_cotracker_process.py \
  qqtt/demo/demo3_runtime.py \
  qqtt/demo/demo31_runtime.py

conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo3_cotracker_worker \
  tests.test_demo31_cotracker_process_config \
  tests.test_demo3_contract \
  tests.test_demo31_dual_gpu_contract
```
