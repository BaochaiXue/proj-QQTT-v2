# 2026-05-10 Demo 2.2 Async Filtered Fused PCD

Status: implementation complete; deterministic checks passed; hardware target failed.

## Goal

Add a pure RTX 5090 Laptop Demo 2.2 path:

```text
3x RealSense RGB+IR @ 5 FPS
-> single-owner GPU thread: FFS TensorRT opt=5 + compiled EdgeTAM
-> raw fused semantic PCD
-> async latest-wins object/controller filter
-> render latest filtered fused PCD only
```

## Plan

1. Add a `demo2.2-async-filter-5fps` preset with local FFS, single-owner GPU scheduling, live SAM3.1 frame-0 initialization, `stuffed animal` object prompt, and `towel` controller prompt.
2. Add a thin Demo 2.2 wrapper under `demo_v2_2/` that delegates to the Demo 2.1 runtime with the Demo 2.2 preset.
3. Split three-camera fusion into raw semantic PCD build and async latest-wins filtering while preserving the old synchronous path for Demo 2.1 presets.
4. Extend profile summaries with raw fusion FPS, filter output FPS, filtered render FPS, filter replacement counts, and Demo 2.2 PASS/FAIL threshold.
5. Add deterministic smoke tests and run the requested checks.

## Validation

- `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_check_all_smoke`: passed.
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`: passed.
- Hardware run:

```bash
conda run --no-capture-output -n demo_2_max python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-async-filter-5fps \
  --duration-s 120 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_async_filter_5fps_profile.json \
  --debug
```

Hardware result:

- filtered render FPS after warmup: `2.64`
- raw fusion FPS after warmup: `2.64`
- filter output FPS after warmup: `2.64`
- complete fused groups after warmup: `213 / 221`
- PASS threshold: `4.80 FPS`
- result: `FAIL`
- bottleneck class: `upstream_supply`
