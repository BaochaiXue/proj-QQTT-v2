# 2026-05-10 Demo 2.1.5 RealSense Depth

## Goal

Create a Demo 2.1.5 entrypoint copied from the current Demo 2.2 public CLI shape, but using native RealSense RGB-D depth instead of FFS depth. Keep the rest of the Demo 2.2 async filtered fused PCD behavior unchanged where it still applies.

## Plan

1. Add Demo 2.1.5 preset constants in the shared Demo 2.1 runtime.
2. Add a RealSense-native depth group path that converts aligned `depth` uint16 frames into color-space meters and feeds the existing fused PCD builder.
3. Add `demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py` as the thin public CLI wrapper.
4. Add smoke tests for CLI translation, contract defaults, thread specs, and native depth group conversion.
5. Add the new wrapper to `check_all.py` help coverage.
6. Add an operator-facing warm-cache mode for Demo 2.1.5 so EdgeTAM/HF/torch.compile and SAM3.1 can be warmed before the formal live run.
7. Record per-task parallel initialization start/end/duration timing to diagnose whether EdgeTAM prewarm and SAM3.1 preload actually overlap.

## Validation

- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_1_5_realsense_depth_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --dry-run`
- PASS: `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --warm-cache-only --warm-cache-repeat 1 --warm-cache-json-output docs/generated/demo2_1_5_init_cache_warmup_probe.json --debug`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`

## Results

- Added Demo 2.1.5 wrapper and default RealSense-depth contract.
- Added warm-cache-only path for EdgeTAM/HF/torch.compile and SAM3.1 preload.
- Added per-task parallel-init timing records under `init_profile.parallel_init`.
- Warm-cache probe completed in 29390.44 ms without opening cameras.
