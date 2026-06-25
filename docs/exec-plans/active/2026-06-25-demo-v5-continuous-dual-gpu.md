# Demo v5 Continuous Dual-GPU Realtime

## Goal

Complete `demo_v5` as the realtime end-to-end bridge from fake/live
single-camera capture to one continuous `realtime_phystwin` zero-order then
first-order optimization run.

## Requirements

- Keep all single-camera work on `single-camera`.
- Reference and preserve `data_process_sam3d` processing semantics where Demo
  v5 writes `final_data.pkl` and shape-prior points.
- Strictly adapt to the repo-local `realtime_phystwin` online consumption:
  one `online_dir`, one aggregate `static_data_path`, stable topology fields,
  and contiguous frame ranges.
- Default to warmup dual GPU, then runtime split:
  - physical GPU0: fake/live camera through final data and online chunks
  - physical GPU1: `realtime_phystwin` zero-order and first-order optimization
- Run one continuous optimizer process for the whole online stream, not
  per-chunk optimization.
- Preserve optimizer quality defaults; do not lower iterations, batch windows,
  topology, or tracking semantics to hit cadence.
- Keep the camera-to-final-data publication path at 5 FPS.

## Design

- Make `demo_v5` import its own modules instead of `demo_v4`.
- Keep the topology wire contract as `demo_v4_session_topology_v1` because
  `realtime_phystwin` currently rejects any other topology version.
- Add a v5 optimization orchestration layer in
  `demo_v5/realtime_futurephystwin_chunks.py`.
- Add managed SAM3D worker support for the warmup phase. The managed worker
  runs in `phystwin-max` with `CUDA_VISIBLE_DEVICES=1` and `--device cuda:0`.
- Start continuous optimization from the first committed online chunk via the
  existing `on_chunk_written` callback, after stopping the managed warmup worker
  when it is safe to free GPU1.
- Pass `CUDA_VISIBLE_DEVICES=1` to the optimization process and `--device
  cuda:0` to `train_online_zero_then_first.py`.
- Pass portable relative data paths to the optimization process. Since its
  working directory is `realtime_phystwin/`, generated data paths are
  `../result/demo_v5/...`.
- Do not pass `--stop_when_finished` by default; finite fake-live validation
  should still run first-order optimization to its configured iteration budget
  unless a quick smoke run explicitly lowers `--optimization-iterations`.
- Keep source-headless conversion as a conversion/debug path; require
  `--optimization-mode disabled` there so it cannot be mistaken for realtime.
- Update scope/architecture docs so Demo v5 is explicitly isolated from formal
  recording/alignment products.

## Tasks

- [x] Inspect the current `demo_v5`, `data_process_sam3d`, and
  `realtime_phystwin` contracts.
- [x] Add design and exec-plan documentation.
- [x] Replace `demo_v5` imports and labels that still point at `demo_v4`.
- [x] Add optimization command construction and runtime process management.
- [x] Add managed shape-prior worker command construction and shutdown.
- [x] Wire `on_chunk_written` so optimization starts once and only once for the
  continuous online stream.
- [x] Add v5 tests for defaults, command construction, GPU routing, source
  headless guard, and no per-chunk optimizer starts.
- [x] Update docs/scope/architecture and README for v5.
- [x] Run focused tests, py_compile, scope guards, and smoke validation.
- [x] Run fake-live camera to continuous optimization E2E and compare quality
  against the closest offline FuturePhysTwin baseline.

## Validation Log

- `git branch --show-current` -> `single-camera`.
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin` -> 8 tests OK.
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin tests.test_demo_v4_futurephystwin_chunks tests.test_realtime_masked_edgetam_pcd_filter tests.test_demo32_shape_prior_warmup.Sam3dOnlyCliContractTest.test_shape_prior_worker_parser_accepts_alignment_coverage_override tests.test_demo32_shape_prior_warmup.SingleViewShapeAlignmentTest.test_alignment_config_accepts_observation_coverage_override tests.test_single_demo_tapnextpp_overlay.SingleDemoTapNextOverlayTest.test_headless_capture_metadata_preserves_prepared_only_flag` -> 73 tests OK.
- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5/realtime_futurephystwin_chunks.py demo_v5/realtime_camera_final_data.py qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_demo_v5_realtime_phystwin.py` -> OK.
- `conda run -n demo_2_max --no-capture-output python demo_v5/realtime_futurephystwin_chunks.py --dry-run` -> default contract uses Demo v5 camera entrypoint, camera GPU0, managed SAM3D worker GPU1, optimizer GPU1, and relative `../result/demo_v5/...` paths from `realtime_phystwin/`.
- Fake-live short E2E: `result/demo_v5/e2e_smoke_20260625c`, 2 chunks, optimizer return code 0, shape-prior worker released before optimization.
- Fake-live quality E2E: `result/demo_v5/e2e_quality_20260625`, 5 chunks / 175 aggregate frames, `final_data.pkl` finite, 700 surface points, 1000 interior points, stable topology hash across chunks, optimizer return code 0.
- Quality E2E losses: online zero-order 10 iters best/final `5.50e-4`; online first-order 20 iters best `1.29e-4`, final `1.49e-4`, all first-order iterations consumed 175 frames.
- Closest existing offline FuturePhysTwin reference: `realtime_phystwin/experiments/demo_v4_native_single_gpu_unlimited_chunk_0031/inv_phy_log.log`, first-order best `7.18e-5` after 200 iterations. Demo v5's 20-iteration online result is the same order of magnitude and not an obvious quality collapse.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_scope.py` -> passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_experiment_boundaries.py` -> passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_visual_architecture.py` -> passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` -> 302 tests OK, smoke checks passed.
- `git diff --check` -> OK.
