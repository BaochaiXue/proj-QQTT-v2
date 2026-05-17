# Demo 2.2 Dependency Separation And Harness Engineering Cleanup

## Goal

Make Demo 2.2 own its runtime import boundary instead of depending on other Demo 2.x entrypoints, while keeping the current async filtered fused-PCD behavior and render fastpath contract intact.

## Scope

- Replace the current `demo_v2_2.runtime` re-export layer with a Demo 2.2-owned runtime module.
- Remove Demo 2.1's dependency on Demo 2.2 render helpers so the version folders do not import each other.
- Keep Demo 2.2 public CLI aliases stable and preserve the current object/controller convention (`stuffed animal` / `towel`).
- Put Demo 2.2 benchmark/harness engineering files in the documented harness catalog flow.
- Update tests/docs that encode the old dependency or harness-file assumptions.

## Non-Goals

- Do not change formal recording/alignment outputs.
- Do not change camera, FFS, EdgeTAM, or tracking quality defaults except where needed to preserve existing Demo 2.2 behavior.
- Do not move external engines, checkpoints, generated profile JSON, or hardware artifacts into source code.

## Validation

- `python -m py_compile demo_v2_2/*.py demo_v2_1/*.py qqtt/demo/*.py scripts/harness/*.py`
- `python -m unittest -v tests.test_demo22_render_fastpath tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke tests.test_demo_v2_1_5_realsense_depth_smoke tests.test_check_all_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
