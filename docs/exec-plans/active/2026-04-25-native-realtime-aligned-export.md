# Native Realtime Aligned Export

Add a native RealSense RGB-D realtime baseline exporter that writes one growing
PhysTwin-compatible case under `data/different_types_real_time/`.

## Scope

- Add a top-level `record_data_realtime_align.py` CLI.
- Use native RealSense `rgbd` capture only; no FFS path.
- Keep the case directory compatible with the formal `data/different_types`
  contract: `color/`, `depth/`, `calibrate.pkl`, and legacy `metadata.json`
  only.
- Write FPS/stability logs outside the formal case under `_logs/`.

## Plan

1. Reuse `CameraSystem.get_observation()` for synchronized latest RGB-D frame
   sets.
2. Normalize `calibrate.pkl` into case camera order using the same serial
   mapping semantics as formal exports.
3. Write accepted frame sets with sequential frame IDs independent of RealSense
   step IDs.
4. Reject duplicate step tuples and timestamp spreads above the configured sync
   tolerance.
5. Add software-only tests with a fake camera system for output contract,
   metadata, duplicate/reject behavior, and FPS log placement.
6. Document the new baseline workflow and include the CLI/test in deterministic
   checks.

## Validation

- `python -m unittest -v tests.test_record_data_realtime_align_smoke`
- `python scripts/harness/check_all.py`
