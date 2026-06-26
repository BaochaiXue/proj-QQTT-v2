# Demo v5 Backup Bundle Support Gate Fix

## Goal

Keep Demo v5 fail-closed while making fixed-anchor backup bundle recovery resilient to processed-mask and motion-filter false negatives. Do not relax the invalid quality gate.

## Implementation

- Update `StreamingControllerAnchorSelector` so backup bundle initialization uses first-frame raw visibility, depth validity, finite/nonzero 3D, local radius, and spread as hard eligibility. Use `controller_mask` only as a ranking preference.
- Use frame-level motion validity in primary and bundle decisions. The last frame uses the previous transition validity; a single-frame chunk treats motion as valid.
- Change bundle recovery hard support to raw-visible + depth-valid + finite/nonzero. Processed mask and frame-level motion become support weights with default fail weight `0.6`.
- Add per-anchor-frame bundle layer diagnostics for raw-visible, depth-valid, processed-mask-valid, frame-motion-valid, and used support counts.
- Change chunk markers so normal writes `READY`, degraded writes `DEGRADED`, invalid writes `INVALID`; invalid never writes `READY`.
- Default degraded chunks to diagnostic-only. Add `--allow-degraded-online` to permit degraded online append explicitly. Invalid chunks remain skipped.

## Validation

- Add red tests for widened bundle initialization, weighted bundle recovery, last-frame frame-level motion, marker policy, degraded publish defaults, and new diagnostic matrix persistence.
- Run targeted unittest modules:
  - `tests.test_phystwin_strict_product`
  - `tests.test_demo_v5_realtime_phystwin`
- Run smoke validation:
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
- Rerun full fake-live Demo v5 with realtime PhysTwin optimization disabled, regenerate the diagnostic video and bundle report, and copy artifacts to Downloads.
