# Demo 3.x Fake-Live Default 5fps

## Goal

Change the official Demo 3.x fake-live default replay cadence to 5fps while
preserving explicit `--replay-fps` overrides and metadata-driven recording
replay behavior.

## Planned Changes

- Default official fake-live wrapper runs to `--replay-fps 5.0` when omitted.
- Preserve explicit positive `--replay-fps` values as CLI overrides.
- Preserve explicit `--replay-fps 0` as metadata FPS.
- Keep recording replay defaults metadata-driven.
- Update tests and user-facing fake-live docs.

## Validation

- Run focused runtime/replay unit tests.
- Dry-run Demo 3.1/3.2/3.3 fake-live defaults.
- Run quick harness.

## Status

- Implemented and validated.

## Results

- PASS: `python -m unittest tests.test_single_demo_v3_runtime tests.test_recorded_rgbd_replay_source`
- PASS: Demo 3.1 / 3.2 / 3.3 fake-live dry-run defaults report `replay_fps=5.0` and `replay_fps_source=default_fake_live`.
- PASS: Explicit `--replay-fps 0` dry-run resolves to recording metadata FPS.
- PASS: `python scripts/harness/check_all.py`
