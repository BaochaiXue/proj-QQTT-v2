# Demo v4 Chunk Seconds Contract

## Goal

Make Demo v4 chunk duration an explicit operator-facing time setting:
default 5 seconds, configurable to other durations, while keeping frame-count
override available for tests and advanced workflows.

## Steps

- [x] Add tests for default 5 second chunk timing and derived frame count.
- [x] Add tests for custom `--chunk-seconds` changing chunk frames and Demo 3.2
  capture duration.
- [x] Add tests that explicit `--chunk-frame-count` overrides derived frame
  count but still requires a valid positive chunk time/FPS contract.
- [x] Add validation for non-positive chunk seconds, replay FPS, and frame
  counts.
- [x] Update Demo v4 docs to describe time-first chunk configuration.
- [x] Run focused Demo v4 tests and smoke validation.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks`
- `git diff --check`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
