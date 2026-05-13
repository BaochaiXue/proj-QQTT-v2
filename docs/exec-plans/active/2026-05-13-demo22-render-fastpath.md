# Demo 2.2 Render Fastpath

## Goal

Improve Demo 2.2 pointcloud rendering without treating lower point count as the primary speed lever.

## Scope

- Audit the current Open3D render path and record it under `docs/generated/`.
- Add a small render fastpath support module for latest-only buffering and micro-profile summaries.
- Wire Demo 2.2 runtime flags for render backend/copy/profile controls.
- Keep the existing PCD compute/filter quality unchanged by default.
- Add deterministic unit tests for the new render buffer/profile behavior and CLI contract.

## Non-Goals

- Do not change FFS, EdgeTAM, or mask postprocess quality settings.
- Do not run long live RealSense profiles until the code path is validated.
- Do not make display-only LOD the default optimization.

## Validation

- `python -m py_compile demo_v2_2/*.py demo_v2_1/*.py`
- `python -m unittest -v tests.test_demo22_render_fastpath tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `python scripts/harness/check_all.py`
