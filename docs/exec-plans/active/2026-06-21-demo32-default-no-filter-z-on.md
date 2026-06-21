# Demo 3.2 No-Filter Z-On Default

**Goal:** Make Demo 3.2/3.3 headless capture and live visual demo defaults
use unfiltered residual PCD with table-Z deletion enabled.

## Plan

- Update Demo 3.x runtime preset defaults so Demo 3.2/3.3 headless capture and
  `--demo-visual-mode pcd|tracking` default to
  `--pcd-filter-preset original`, which maps object/controller filters to
  `none` and caps to `0`.
- Keep explicit `--pcd-filter-preset pt` and `--pcd-filter-preset enhanced-pt`
  overrides working for both live visual modes and headless capture.
- Keep the existing default table-Z behavior enabled at threshold `0.0` for
  live visual modes and headless capture.
- Update tests and operator docs to describe `filter none + table-Z ON` as the
  default and `{pt,enhanced-pt}` as explicit ablation/cleanup presets.

## Validation

- Run focused Demo 3.x runtime tests for visual defaults and headless capture.
- Run the single-demo / side-by-side focused test set.
- Run smoke harness validation.
