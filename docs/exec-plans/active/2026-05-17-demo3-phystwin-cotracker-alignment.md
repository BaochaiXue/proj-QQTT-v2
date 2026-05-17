# Demo 3 PhysTwin CoTracker Alignment

## Goal

Align Demo 3's CoTracker-compatible benchmark/export path with the
FuturePhysTwin dense tracking convention while preserving the existing sparse
3D anchor/trail visualization behavior.

## Scope

- Add a PhysTwin dense query mode that uses first-frame union masks and selects
  exactly 5000 query points per camera, with strict failure below 5000 mask
  pixels.
- Match FuturePhysTwin's deterministic dense query sampling: torch `randperm`,
  default seed `42`, and `seed + camera_idx`.
- Support PhysTwin-style nested mask layouts such as `mask/{camera}/*/{frame}.png`
  for benchmark query generation and metrics.
- Make the PhysTwin-compatible benchmark path write `cotracker/{camera}.npz`
  artifacts without changing the existing Demo 3 overlay reader.
- Make the Demo 3 CoTracker backend expose true online streaming updates with
  the CoTracker3 `window_len=16` / `step=8` rolling-buffer contract.
- Update docs and tests to distinguish dense PhysTwin export artifacts from
  sparse realtime/overlay visualization.

## Non-Goals

- Do not put 5000-point dense CoTracker calls into the live Demo 2.2 render hot
  path by default.
- Do not change the Demo 3 overlay board layout or its default sparse display
  point cap.
- Do not import FuturePhysTwin code or add a runtime dependency on that repo.
- Do not make dense CoTracker streaming block the main fused-PCD render path by
  default.

## Validation

- Targeted Demo 3 tracking sampling and harness tests.
- `python scripts/harness/check_all.py` in the default environment if feasible.
- Completed: `python -m py_compile qqtt/tracking/sampling.py scripts/harness/experiments/run_demo3_tracking_backend_benchmark.py tests/test_demo3_tracking_sampling_smoke.py tests/test_demo3_tracking_harness_smoke.py`
- Completed: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_tracking_sampling_smoke tests.test_demo3_tracking_harness_smoke tests.test_demo3_tracking_registry_smoke`
- Completed: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
