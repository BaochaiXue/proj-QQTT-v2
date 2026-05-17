# Demo 3 PhysTwin CoTracker Alignment

## Goal

Align Demo 3's CoTracker-compatible benchmark/export path with the
FuturePhysTwin dense tracking convention while preserving the existing sparse
3D anchor/trail visualization behavior.

## Scope

- Add a PhysTwin dense query mode that uses first-frame union masks and selects
  up to 5000 query points per camera: 5000 when enough mask pixels exist, or
  all available mask pixels when fewer than 5000 exist.
- Match FuturePhysTwin's deterministic dense query sampling: torch `randperm`,
  default seed `42`, and `seed + camera_idx`.
- Support PhysTwin-style nested mask layouts such as `mask/{camera}/*/{frame}.png`
  for benchmark query generation and metrics.
- Make `phystwin_dense` the default Demo 3 benchmark/export query mode.
- Make the PhysTwin-compatible benchmark path write `cotracker/{camera}.npz`
  artifacts without changing the existing Demo 3 overlay reader.
- Make the Demo 3 CoTracker backend expose true online streaming updates with
  the CoTracker3 `window_len=16` / `step=8` rolling-buffer contract.
- Keep the saved-case `cotracker3_online` benchmark/export path on the same
  frame-by-frame `update(frame)` contract instead of a one-shot whole-video
  call.
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
- Completed after online replay change: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_tracking_registry_smoke tests.test_demo3_tracking_harness_smoke tests.test_demo3_tracking_sampling_smoke`
- Completed after online replay change: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Completed after up-to-5000 query cap change: `python -m py_compile qqtt/tracking/sampling.py scripts/harness/experiments/run_demo3_tracking_backend_benchmark.py tests/test_demo3_tracking_sampling_smoke.py tests/test_demo3_tracking_harness_smoke.py`
- Completed after up-to-5000 query cap change: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_tracking_sampling_smoke tests.test_demo3_tracking_harness_smoke tests.test_demo3_tracking_registry_smoke`
- Completed after up-to-5000 query cap change: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Completed after default phystwin_dense change: `python -m py_compile scripts/harness/experiments/run_demo3_tracking_backend_benchmark.py tests/test_demo3_tracking_harness_smoke.py tests/test_demo3_tracking_backend_benchmark_fake_smoke.py`
- Completed after default phystwin_dense change: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_tracking_harness_smoke tests.test_demo3_tracking_backend_benchmark_fake_smoke tests.test_demo3_tracking_sampling_smoke tests.test_demo3_tracking_registry_smoke`
- Completed after default phystwin_dense change: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
