# Demo 3 FuturePhysTwin Dense Tracking Semantics

## Goal

Fix Demo 3.0 and Demo 3.1 live CoTracker semantics so both online RealSense
runtimes track the object/controller union with FuturePhysTwin-compatible dense
query sampling instead of exposing object-only sparse tracking as the default.

## Scope

- Replace public Demo 3.0 and Demo 3.1 `--track-mode` semantics with
  `--mode exp|demo`.
- Keep both modes online-only, exactly three RealSense, RealSense-depth-only,
  SAM3.1 first-frame initialization, HF EdgeTAM online mask propagation, and
  CoTracker3 online.
- Use object/controller union masks for all CoTracker inputs.
- Make CoTracker query sampling default to `phystwin_dense`, requested count
  `auto`, actual count `min(union_mask_pixels, 5000)` per camera, sampled with
  torch `randperm(seed + camera_idx)`.
- Keep overlay display count separate from raw tracked query count.
- Preserve Demo 3.1 GPU0/GPU1 process split and CPU-only IPC.

## Results

- Demo 3.0 and Demo 3.1 now expose `--mode exp|demo` and no longer expose
  public `--track-mode object-only/controller-only` paths.
- `exp` resolves to shared experiment mode `controller-object-exp` with
  controller prompt/label `towel`; `demo` resolves to `demo-mode` with
  controller prompt/label `hand`.
- Both runtimes always send object/controller union masks to CoTracker.
- CoTracker input packets now preserve union, object, and controller masks.
- `CoTracker3OverlayWorker` defaults to `phystwin_dense`, `auto` query count,
  seed 42, and torch `randperm(seed + camera_idx)` through
  `sample_phystwin_dense`.
- CoTracker initialization now waits until both object and controller masks are
  non-empty, preventing an object-only first query set from being cached.
- Raw tracked query count is separate from `overlay_max_points_per_camera`.
- Demo 3.1 keeps GPU0/GPU1 isolation and CPU-only IPC while returning dense
  query/profile stats from the child process result packet.
- Updated Demo 3.0 and Demo 3.1 runtime docs and high-performance tracking
  backend notes.

## Validation

- PASS:
  `conda run --no-capture-output -n demo3-max python -m py_compile qqtt/demo/demo3_runtime.py qqtt/demo/demo31_runtime.py qqtt/demo/cotracker3_overlay_worker.py qqtt/demo/demo31_dual_gpu_ipc.py qqtt/demo/demo31_cotracker_process.py qqtt/demo/demo31_profile.py tests/test_demo3_contract.py tests/test_demo3_cotracker_worker.py tests/test_demo31_dual_gpu_contract.py tests/test_demo31_ipc_latest_wins.py tests/test_demo31_cotracker_process_config.py`
- PASS:
  `conda run --no-capture-output -n demo3-max python -m unittest -v tests.test_demo3_contract tests.test_demo3_cotracker_worker tests.test_demo31_dual_gpu_contract tests.test_demo31_ipc_latest_wins tests.test_demo31_cotracker_process_config`
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --camera-ids 0,1,2`
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --camera-ids 0,1,2 --mode demo`
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda --calibrate-path calibrate.pkl`
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --track-mode object-only`
  fails with argparse status 2.
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py --track-mode object-only`
  fails with argparse status 2.
- PASS:
  `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --input-video foo.mp4`
  fails with argparse status 2.
- BLOCKED:
  `conda run --no-capture-output -n demo3-max python scripts/harness/check_all.py`
  currently fails because the working tree has unrelated tracked deletions for
  older demo directories (`demo_v0_2`, `demo_v0_3`, `demo_v1`, `demo_v2`,
  `demo_v2_1`, and `demo_v2_1_5`). The quick harness still references modules
  from those paths, so imports such as `demo_v2_1` and `demo_v2_1_5` fail before
  this Demo 3 semantic patch can get a clean full quick-harness result.

