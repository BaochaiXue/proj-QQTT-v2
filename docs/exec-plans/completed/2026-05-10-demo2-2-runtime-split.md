# 2026-05-10 Demo 2.2 Runtime Split

## Goal

Stop Demo 2.2 and Demo 2.1.5 public entrypoints from importing the old Demo 2.1 monolith directly. Add a dedicated Demo 2.2 runtime module that owns the Demo 2.2 / 2.1.5 public contract, while leaving the historical Demo 2.1 entrypoint in place for old tests and backwards compatibility.

## Plan

1. Add `demo_v2_2/runtime.py` as the dedicated Demo 2.2 / 2.1.5 runtime boundary.
2. Move Demo 2.2 / 2.1.5 wrapper imports to `demo_v2_2.runtime`.
3. Move Demo 2.2 and Demo 2.1.5 smoke tests to the new runtime module.
4. Add regression checks that the wrappers no longer directly import `demo_v2_1`.
5. Document the split and the remaining follow-up: moving internal implementation out of the legacy monolith in smaller modules.

## Validation

- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_5_realsense_depth_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`

## Results

- Added `demo_v2_2/runtime.py` as the dedicated Demo 2.2 / 2.1.5 runtime import boundary.
- Updated Demo 2.2 and Demo 2.1.5 wrappers to import `demo_v2_2.runtime` instead of the old Demo 2.1 entrypoint.
- Updated Demo 2.2 and Demo 2.1.5 smoke tests to exercise the new boundary.
- Added regression tests ensuring the wrappers do not directly import `demo_v2_1.realtime_three_view_masked_fused_pcd`.
- Documented the split in `docs/generated/demo2_2_runtime_split.md`.
