# Demo 3.2 PCD Filter Preset Query Residual Plan

## Goal

Expose a high-level Demo 3.x PCD filter preset that controls object/controller
PCD and TAPNext++ query initialization together.

## Planned Changes

- Add `--pcd-filter-preset {original,pt,enhanced-pt}` to the single Demo 3.x
  wrapper. The preset maps to both object and controller filters, overrides
  Demo 3.2/3.3 visual-mode defaults, and rejects conflicting explicit
  per-class filter options.
- Preserve source pixel `yx` indices through backprojection, voxel cap, PT
  filter, enhanced-PT filter, and filter fallback paths.
- Build residual object/controller masks from filtered survivor pixels and
  initialize TAPNext++ queries only from those residual masks.
- Keep existing lower-level per-class filter controls available when the new
  high-level preset is not supplied.

## Validation

- Add failing runtime tests for preset defaults, overrides, conflicts, contract,
  and delegate argv.
- Add failing delegate/filter tests for `yx` preservation, stride mapping,
  residual query sampling, and fail-fast residual shortages.
- Run focused unit tests, Demo 3.2 dry-runs for all presets, and the smoke
  validation profile before committing and pushing `origin single-camera`.
