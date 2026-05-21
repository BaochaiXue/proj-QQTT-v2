# Demo 3.1 TAPNext++ Backend

Status: active

## Goal

Add TAPNext++ as a Demo 3.1 point-tracker backend under the existing child-process
PointTrackerAdapter contract. Keep GPU0 responsible for RealSense, masks, depth
fusion, and rendering. Keep GPU1 responsible for CPU RGB/mask input packets and
CPU 2D track/visibility output packets.

The live stress-test target remains `4096` query points per camera, which means
`12288` total query points across the three Demo 3.1 views. The real
approximately-4000-total target is `1365/view` (`4095` total), and summaries
must clearly separate that target from the `4096/view` stress test.

## Scope

1. Add a `tapnextpp` point-tracker backend and aliases.
2. Add TAPNext++ adapter support for serial and true batch-views.
3. Add Demo 3.1 CLI, dry-run contract, and child-process config wiring.
4. Add an installer for `demo_3_1_max` that does not replace CUDA Torch.
5. Add fake-model tests proving online serial and batch-views call shapes.
6. Add profiling harnesses for `1365/view` rendered live Demo 3.1 and keep
   `4096/view` available as a stress repeat.
7. Add group-level timing fields so serial per-camera calls are not compared
   incorrectly with batch-views group calls.
8. Add model-only TAPNext++ benchmark support to separate model cost from
   RealSense/render/IPC cost.

## Non-Goals

- Do not move depth, intrinsics, c2w, or world lifting into TAPNext++.
- Do not replace the default backend.
- Do not change TAPNext++ tracking semantics while adding timing and profiling
  harnesses.
- Do not silently fall back from batch-views to serial.

## Verification

- Focused unit tests for point tracker adapters and Demo 3.1 dry-run contracts.
- `scripts/env/install_tapnextpp_demo_3_1_max.sh --help`.
- Real TAPNext++ import/checkpoint smoke in `demo_3_1_max` when the install
  script is run.
- Rendered profiling for:
  - `tapnextpp`, `serial`, `1365/view`
  - `tapnextpp`, `batch-views`, `1365/view`
  - optional stress repeat at `4096/view`
  - model-only B=1/B=3 recurrent update sweeps

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
- PASS: Fixed TAPNext++ PyTorch adapter preprocessing to match the official
  DAVIS path: RGB frames are `float32 [-1, 1]`, not `[0, 255]`.
- PASS: Fixed TAPNext++ PyTorch raw track parsing to keep the model's native
  `yx` output convention before converting back to QQTT original-frame `yx`.
- PASS: Real TAPNext++ CUDA smoke on tracker GPU with an `848x480` frame and
  query points confirms first-frame output stays near the queries and in image
  bounds.
- PASS: Rendered live RealSense/Open3D serial profile for
  `tapnextpp`, `4096/view`, `120s`:
  `docs/generated/demo31_tapnextpp_rendered_profile/serial_q4096_live_fixed_yx_120s_shared_runtime.json`.
  Shared runtime after-warmup render FPS: `6.373`; rendered groups: `719`.
  Tracker process first render group: `0`; warmup skipped: `0`; render blocked:
  `0`; query counts: `{0: 4096, 1: 4096, 2: 4096}`.
- PASS: Rendered live RealSense/Open3D batch-views profile for
  `tapnextpp`, `4096/view`, `120s`:
  `docs/generated/demo31_tapnextpp_rendered_profile/batch_views_q4096_live_fixed_yx_120s_shared_runtime.json`.
  Shared runtime after-warmup render FPS: `5.730`; rendered groups: `647`.
  Tracker process first render group: `0`; warmup skipped: `0`; render blocked:
  `0`; query counts: `{0: 4096, 1: 4096, 2: 4096}`; batch errors: `0`.
- PASS: Committed summary:
  `docs/generated/demo31_tapnextpp_rendered_profile/summary_q4096_live_fixed_yx.md`.
- PASS: Fixed tracker child-process ready/warmup profiling so the child
  `ready_perf_s` event time is recorded separately from runtime receive time;
  the runtime now drains status events in the startup/background path and in
  tracking input/result hot paths.
- PASS: Added regression coverage:
  `tests.test_demo31_dual_gpu_contract.Demo31DualGpuContractTest.test_tracker_ready_status_records_event_time_not_teardown_time`.
- PASS: `conda run --no-capture-output -n demo_3_1_max python -m unittest tests.test_point_tracker_adapters tests.test_demo31_cotracker_process_config tests.test_demo31_dual_gpu_contract`.
- PASS: `conda run --no-capture-output -n demo_3_1_max python scripts/harness/check_all.py`.
- PASS: Rendered live RealSense/Open3D ready-timing verification for
  `tapnextpp`, `4096/view`, `45s`:
  `docs/generated/demo31_tapnextpp_rendered_profile/summary_q4096_live_readyfix.md`.
  Tracker ready receive times are `7.188s` serial and `4.086s` batch-views;
  first rendered groups remain `0` in both modes with no warmup-skipped or
  render-blocked frames.
- PASS: Added group-level tracker timing fields for serial-vs-batch
  disambiguation: group wall, sum/max model ms per group, per-camera model ms,
  model calls/instances, and total query count.
- PASS: `conda run --no-capture-output -n demo_3_1_max python -m unittest tests.test_demo3_cotracker_worker.Demo3CoTrackerWorkerTest.test_batch_backend_updates_three_cameras_together tests.test_demo3_cotracker_worker.Demo3CoTrackerWorkerTest.test_serial_backend_records_group_model_timing_for_three_cameras tests.test_demo31_tapnextpp_profile_summaries`.
- PASS: `conda run --no-capture-output -n demo_3_1_max python scripts/harness/check_all.py`.
- PASS: Model-only TAPNext++ q1365 smoke on tracker GPU:
  `docs/generated/demo31_tapnextpp_model_only_smoke/summary.md`.
  B=1 q1365 recurrent model p50 `14.270ms`; B=3 q1365 recurrent model p50
  `16.316ms`.
- PASS: Rendered live q1365/q4096 group-timing summary:
  `docs/generated/demo31_tapnextpp_rendered_profile/summary_q1365_q4096_live_group_timing.md`.
  q1365 serial/batch render FPS: `7.253/7.264`; q4096 serial/batch render FPS:
  `6.856/6.409`.
