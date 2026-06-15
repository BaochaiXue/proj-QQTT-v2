# Demo 3.2 Visual Mode Render-Only Semantics

## Goal

Make Demo 3.2/3.3 `demo_visual_mode` choose only what is rendered, not which
pipeline stages run. Both `pcd` and `tracking` modes should run FFS, EdgeTAM,
enhanced-pt PCD, and TAPNext++ query tracking so the reported FPS reflects the
full realtime pipeline.

## Planned Changes

- Keep TAPNext++ enabled by default for both Demo 3.2/3.3 visual modes.
- Reject visual-mode runs that explicitly disable the tracker.
- Keep strict same-seq PCD/tracker pairing for both modes.
- Hide tracker marker layers in `pcd` mode while still using tracker telemetry
  in HUD/debug output.
- Update dry-run contracts, tests, and Demo 3.2 docs to describe render-only
  visual mode semantics.

## Validation

- Update targeted unit tests for runtime contracts and strict-pair worker use.
- Run `tests.test_single_demo_v3_runtime` and
  `tests.test_single_demo_tapnextpp_overlay`.
- Run `scripts/harness/check_all.py`.

## Status

- Implemented and validated.

## Results

- PASS: `python -m py_compile qqtt/demo/single_demo_v3_runtime.py qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_single_demo_v3_runtime.py tests/test_single_demo_tapnextpp_overlay.py`
- PASS: `python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay`
- PASS: Demo 3.2 dry-run for `--demo-visual-mode pcd` and `tracking` both report `tracker_backend=tapnextpp` and `tracker_sync_policy=strict_same_seq_latest_wins`.
- PASS: Demo 3.2 dry-run with `--demo-visual-mode pcd --tracker-backend none` fails fast.
- PASS: `python scripts/harness/check_all.py`
