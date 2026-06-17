# Demo 3.x Table-World Z Diagnostics Implementation Plan

> **For agentic workers:** Use TDD. Add failing tests before production code and keep existing uncommitted Demo 3.x PCD-filter changes intact.

**Goal:** Make Demo 3.1, Demo 3.2, and Demo 3.3 expose table-world Z diagnostics, and make their visual PCD/tracking modes apply the table-Z filter by default at 0 mm signed clearance.

**Architecture:** Reuse the existing `table_calibrate.pkl` / `table_world_z0` contract. Build PCD with existing PT/enhanced-PT behavior first, transform the output to table-world coordinates, compute world-Z stats and threshold candidates, then remove table-band points in Demo 3.1/3.2/3.3 visual modes unless `--disable-table-z-filter` is passed. Headless capture stores stats and metadata so offline RGB overlay videos can compare before/after without rerunning the demo. The tabletop is `table_z_m = 0.0`; the current table calibration uses negative Z as the workspace side (`table_z_above_direction = negative`), so diagnostics and filtering operate on signed clearance instead of assuming positive Z is up.

## Tasks

- [x] Add tests for Demo 3.1 default table-world contract and delegate argv.
- [x] Add tests for world-Z quantile stats, threshold candidate counts, explicit filtering, and default visual-mode filtering.
- [x] Add tests for headless metadata `camera_to_world_c2w` and per-frame `world_z_stats.jsonl`.
- [x] Add tests for table-world PCD projection, RGB overlay removed-point rendering, and threshold sweep summary output.
- [x] Implement shared runtime CLI flags and Z diagnostics helpers.
- [x] Extend headless capture writer with metadata and per-frame stats output.
- [x] Extend the headless render helper with table-world projection and RGB overlay sweep mode.
- [x] Fix table-Z diagnostics/filter direction after sweep evidence showed the current calibration's target PCD lies on negative Z above the table.
- [x] Change visual-mode runtime defaults: Demo 3.1/3.2/3.3 `pcd` and `tracking`
  modes enable table-Z filtering at 0 mm signed clearance, with
  `--disable-table-z-filter` for ablations.
- [x] Update Demo 3.1/3.2/3.3 docs and workflows.
- [x] Run focused unit tests, then smoke validation if focused tests pass.

## Verification

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Results:

- PASS: focused unit tests, 134 tests after adding signed table-clearance direction coverage and default visual-mode filtering.
- PASS: smoke validation after default 0 mm visual-mode filtering, 233 tests plus curated help/guard checks.
- Re-run table-Z overlay sweep on the same Demo 3.2 headless capture with
  `table_z_above_direction = negative`: 20 mm removed 191,168 / 3,955,690
  points (4.83%) with 0 zero-kept frames. The old positive-Z assumption removed
  99.995% at 20 mm with 251 zero-kept frames, confirming the sign-direction bug.
