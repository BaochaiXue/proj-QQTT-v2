# Demo 3.1 Multi-Backend Point Trackers

## Goal

Upgrade Demo 3.1 from a CoTracker3-only overlay path to a point-tracker
backend layer that can route the GPU1 child process to:

- `cotracker3_online`
- `trackon2`
- `litetracker`

The main process architecture remains unchanged: GPU0 owns RealSense capture,
HF EdgeTAM masks, native-depth fusion, Open3D rendering, and 2D-to-world lift;
GPU1 owns the latest-wins point-tracker child process.

## Constraints

- Do not change Demo 2.3.
- Do not change FFS, EdgeTAM, renderer, or semantic fusion behavior.
- Preserve Demo 3.1 overlay label accounting, `overlay_display_scope`, status
  queue readiness events, lift-input cache, object-volume controls, and GPU
  sampling fields.
- Keep the public `--cotracker-*` CLI flags for compatibility, while adding
  point-tracker backend fields to contract/profile output.
- Camera views are a batch dimension, never time frames.

## Implementation

1. Add tracker backend constants and CLI choices to Demo 3.1.
2. Add `--tracking-backend-execution-mode`,
   `--tracker-batch-query-count-policy`, and backend-specific path flags.
3. Extend `CoTrackerProcessConfig` JSON roundtrip with tracker backend fields.
4. Add a small point-tracker adapter interface and adapters for CoTracker3,
   Track-On2, and LiteTracker.
5. Route the child process through the adapter factory while preserving the
   existing overlay worker and output packet schema.
6. Add environment bootstrap docs/script for `demo_3_1_max`.
7. Add tests for CLI/contract/config/adapters and run deterministic checks.

## Validation

- Demo 3.1 dry-run shows backend and execution-mode contract fields.
- CoTracker3 serial path remains compatible.
- CoTracker3 batch-views path remains available through `auto` and
  `batch-views`.
- Track-On2 and LiteTracker adapters fail clearly when their external repos or
  checkpoints are not configured.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
  passes.
