# 2026-05-10 Demo 2.2 Staged Parallel GPU-Resident Inputs

Status: completed.

## Goal

Add a Demo 2.2 variant that keeps the async filtered fused PCD path but changes
the GPU owner schedule to:

```text
FFS cam0 -> cam1 -> cam2
then EdgeTAM cam0/cam1/cam2 in parallel
then raw fused semantic PCD
then async latest-wins filter
then filtered-only render
```

## Plan

1. Add preset `demo2.2-staged-parallel-5fps`.
2. Use `gpu_pipeline_mode=staged`, `staged_order=ffs-then-parallel-edgetam`,
   replicated EdgeTAM models, and per-camera CUDA streams.
3. Keep FFS TensorRT pinned staging and persistent device input buffers.
4. Extend EdgeTAM pinned staging to use reusable CUDA pixel-value slots.
5. Keep async filter and filtered-only render unchanged.
6. Add deterministic contract tests and run the hardware profile.

## Validation

- PASS: `conda run --no-capture-output -n demo_2_max python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py tests/test_demo_v2_1_three_view_fused_pcd_smoke.py tests/test_demo_v2_2_async_filtered_fused_pcd_smoke.py`
- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
- Hardware profile:

```bash
conda run --no-capture-output -n demo_2_max python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-staged-parallel-5fps \
  --duration-s 120 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_staged_parallel_5fps_profile.json \
  --debug
```

Hardware result:

- `render_fps=2.70` after warmup.
- `raw_fusion_fps=2.71` after warmup.
- `filter_output_fps=2.71` after warmup.
- Demo 2.2 `4.8 FPS` threshold result: FAIL.
- Staged EdgeTAM wall time median: `143.80 ms`.
- Staged EdgeTAM model-time sum median: `374.17 ms`.
- Conclusion: the staged-parallel schedule runs, but three concurrent compiled
  EdgeTAM streams are slower than the prior serialized single-owner model path
  on this RTX 5090 Laptop workload.
