# Demo v5 Realtime FuturePhysTwin Validation 2026-06-25

## Purpose

Validate Demo v5 as the realtime bridge from Demo v5 fake/live single-camera
capture to repo-local `realtime_phystwin` continuous zero-order then first-order
optimization. The run uses warmup dual GPU: camera/final-data on physical GPU0,
SAM3D warmup on physical GPU1, then `realtime_phystwin` optimization on
physical GPU1 after the managed warmup worker is released.

## Deterministic Checks

```bash
git branch --show-current
```

Outcome: `single-camera`.

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest tests.test_demo_v5_realtime_phystwin
```

Outcome: 8 tests OK.

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest \
    tests.test_demo_v5_realtime_phystwin \
    tests.test_demo_v4_futurephystwin_chunks \
    tests.test_realtime_masked_edgetam_pcd_filter \
    tests.test_demo32_shape_prior_warmup.Sam3dOnlyCliContractTest.test_shape_prior_worker_parser_accepts_alignment_coverage_override \
    tests.test_demo32_shape_prior_warmup.SingleViewShapeAlignmentTest.test_alignment_config_accepts_observation_coverage_override \
    tests.test_single_demo_tapnextpp_overlay.SingleDemoTapNextOverlayTest.test_headless_capture_metadata_preserves_prepared_only_flag
```

Outcome: 73 tests OK.

```bash
conda run -n demo_2_max --no-capture-output \
  python -m py_compile \
    demo_v5/realtime_futurephystwin_chunks.py \
    demo_v5/realtime_camera_final_data.py \
    qqtt/demo/realtime_masked_edgetam_pcd.py \
    tests/test_demo_v5_realtime_phystwin.py
```

Outcome: OK.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py --dry-run
```

Outcome: default contract uses `demo_v5/realtime_camera_final_data.py`, camera
GPU0, managed SAM3D worker GPU1, optimizer GPU1, continuous optimizer mode, and
relative `../result/demo_v5/...` paths from `realtime_phystwin/`.

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/guards/check_scope.py

conda run -n demo_2_max --no-capture-output \
  python scripts/harness/guards/check_experiment_boundaries.py

conda run -n demo_2_max --no-capture-output \
  python scripts/harness/guards/check_visual_architecture.py

git diff --check
```

Outcome: all passed.

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

Outcome: 302 tests OK; smoke checks passed.

## Fake-Live To Optimization Runs

Short E2E:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py \
    --futurephystwin-base-path result/demo_v5/e2e_smoke_20260625c \
    --case-prefix demo_v5_e2e_smoke \
    --shape-prior-endpoint tcp://127.0.0.1:7107 \
    --max-chunks 2 \
    --capture-extra-seconds 80 \
    --optimization-zero-iterations 1 \
    --optimization-iterations 1 \
    --optimization-wait-timeout-s 900 \
    > result/demo_v5/e2e_smoke_20260625c/run.log 2>&1
```

Outcome: exit code 0; 2 chunks; optimizer return code 0; managed SAM3D worker
released before optimization.

Quality E2E:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py \
    --futurephystwin-base-path result/demo_v5/e2e_quality_20260625 \
    --case-prefix demo_v5_quality \
    --shape-prior-endpoint tcp://127.0.0.1:7108 \
    --max-chunks 5 \
    --capture-extra-seconds 120 \
    --optimization-zero-iterations 10 \
    --optimization-iterations 20 \
    --optimization-wait-timeout-s 3600 \
    > result/demo_v5/e2e_quality_20260625/run.log 2>&1
```

Outcome: exit code 0.

- 5 online chunks, 175 aggregate frames.
- Camera command used `demo_v5/realtime_camera_final_data.py`.
- Camera command used `--metadata-demo-version demo_v5` and
  `--metadata-reference-pipeline data_process_sam3d`.
- Camera/final-data path used physical GPU0.
- Managed SAM3D worker used `CUDA_VISIBLE_DEVICES=1`, internal `--device
  cuda:0`, and `--max-observation-to-aligned-p95-m 0.06`.
- `realtime_phystwin` optimizer used `CUDA_VISIBLE_DEVICES=1`, internal
  `--device cuda:0`, and relative data paths from `realtime_phystwin/`.
- Shape-prior worker return code was `-15` because Demo v5 intentionally
  terminated it before starting optimization; optimizer return code was `0`.
- Each chunk had 35 frames, 700 surface points, 1000 interior points, and the
  same topology hash:
  `4a947d23c79ebabad7ec61456bb9b226ef4db81618772656079581f0c84ae913`.
- Aggregate metadata recorded `demo_version=demo_v5`,
  `runtime_product_name=demo_v5_realtime_camera_final_data`,
  `reference_pipeline=data_process_sam3d`, `depth_backend=native-realsense`,
  and `frame_num=175`.
- `final_data.pkl` contained finite arrays:
  `object_points (175, 1991, 3)`, `object_colors (175, 1991, 3)`,
  `controller_points (175, 30, 3)`, `surface_points (700, 3)`,
  `interior_points (1000, 3)`, and fixed query id vectors.

Optimization logs:

- Online zero-order: 10 iterations, best/final loss `5.50e-4`.
- Online first-order: 20 iterations, all on 175 frames, best loss `1.29e-4`,
  final loss `1.49e-4`.
- Closest existing offline FuturePhysTwin first-order reference:
  `realtime_phystwin/experiments/demo_v4_native_single_gpu_unlimited_chunk_0031/inv_phy_log.log`
  reached best loss `7.18e-5` after 200 iterations. The Demo v5 online
  20-iteration result is the same order of magnitude and does not show an
  obvious quality collapse.

Known external dependency note: SAM3D attempted to build `gsplat_cuda` inside
`phystwin-max`, but that environment has no `nvcc`. The SAM3D worker caught the
post-optimization extension failure and still produced the required 700/1000
shape-prior points for Demo v5.
