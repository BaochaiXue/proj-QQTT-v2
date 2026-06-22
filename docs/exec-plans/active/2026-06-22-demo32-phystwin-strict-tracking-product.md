# Demo 3.2 PhysTwin-Strict Tracking Product Backend

## Goal

Add a headless workstation product backend named `phystwin-strict-tracking` that
uses the current Demo 3.2 stack, but emits PhysTwin-compatible query, tracking,
3D track, sampling, pickle, and visualization artifacts.

## Guardrails

- Keep TAPNext++ as the tracker model backend.
- Keep EdgeTAM as the mask backend.
- Keep RealSense/FFS as the depth backend.
- Do not introduce CoTracker as a runtime dependency.
- P0 is finite-window headless/offline finalization only; live panel remains the
  provisional realtime overlay.

## Implementation Steps

- Add product-backend CLI/metadata/delegate fields:
  `--tracking-product-backend {realtime-overlay,phystwin-strict-tracking}` and
  `--phystwin-strict-output-dir`.
- Validate strict P0 only for fake-live headless runs with TAPNext++.
- Add a focused PhysTwin-like compatibility/finalizer module:
  first-frame union query sampling, TAPNext++ track export shape conversion,
  `processed_masks.pkl`, dense world-space PCD grids, strict object/controller
  motion filtering, controller FPS 30, object 5 mm volume sampling, and formal
  videos.
- Hook headless writer/runtime to accumulate the required per-frame artifacts
  and run the strict finalizer when the capture completes.
- Update Demo 3.2 docs and external dependency notes.
- Add tests for parser/delegate metadata and strict helper behavior.

## Validation

```bash
conda run -n demo_2_max --no-capture-output python -m unittest \
  tests.test_single_demo_v3_runtime \
  tests.test_single_demo_tapnextpp_overlay \
  tests.test_realtime_masked_edgetam_pcd_filter

conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```
