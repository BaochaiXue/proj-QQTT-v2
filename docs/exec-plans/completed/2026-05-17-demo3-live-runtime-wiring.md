# Demo 3 Live Runtime Wiring

## Goal

Move Demo 3 past the contract scaffold by wiring non-dry-run execution into the
real three-view runtime path: three RealSense cameras, RealSense depth only, HF
EdgeTAM masks, fused PCD rendering, and an optional async/latest-wins CoTracker3
overlay stage.

## Scope

- Keep Demo 2.2 behavior unchanged.
- Map Demo 3 CLI arguments into the shared Demo 2.x three-view runtime with
  RealSense depth and no FFS options in the Demo 3 public surface.
- Add runtime injection seams so tests can verify non-dry-run calls real
  runtime code paths without hardware.
- Add a latest-wins input slot and thread wrapper for CoTracker3 overlay work.
- Fix live CoTracker robustness so empty first masks do not permanently cache
  empty query points.
- Add stale overlay timeout handling helpers.
- Add connected RealSense / serial / calibration validation helpers that can be
  tested without hardware.
- Update docs/tests to state that live overlay sampling is visualization
  sampling, not byte-identical FuturePhysTwin dense export.

## Non-Goals

- Do not implement FuturePhysTwin `track_process_data.pkl` or inverse physics.
- Do not add FFS depth, FFS TensorRT, FFS remote, or FFS fallback to Demo 3.
- Do not default to dense 5000 visualized tracks in live rendering.
- Do not run hardware in CI.

## Validation

- Unit tests for the Demo 3 runtime adapter, fake live runtime invocation, and
  fake overlay worker threading.
- Existing Demo 3 contract, CoTracker worker, overlay lift tests.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.

## Outcome

- `Demo3Runtime.run()` now validates live RealSense/serial/calibration inputs
  and maps Demo 3 CLI options into the shared three-view runtime with
  RealSense-only depth.
- Demo 3 starts a sidecar CoTracker3 overlay thread using latest-wins input and
  output slots. The renderer consumes the latest non-stale overlay without
  calling CoTracker synchronously.
- The adapter publishes grouped RGB/mask/RealSense-depth packets to CoTracker
  from the fusion path and appends lifted overlay anchors to the render packet
  when fresh world-space overlay points are available.
- Empty first masks no longer permanently cache empty query points.
- Live profile summaries are built from runtime stats and CoTracker/overlay
  samples rather than returning an all-zero scaffold.
- Added deterministic tests for live adapter invocation, connected
  RealSense/calibration validation, empty-mask recovery, stale overlay timeout,
  and the background overlay thread.
- Completed validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.
