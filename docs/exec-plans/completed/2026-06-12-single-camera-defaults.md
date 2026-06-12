# Single-Camera Defaults

## Objective

Make the `single-camera` branch default to one RealSense camera while preserving
the existing aligned-case metadata/layout contract and keeping three-camera
baseline work isolated from `main`.

## Scope

- Change shared camera defaults from three cameras to one camera.
- Update camera entrypoint help text so preview, calibration, recording, and
  realtime aligned export describe the single-camera branch behavior.
- Update docs/workflow snippets that define the branch-local single-camera
  operator path.
- Add deterministic smoke coverage for the one-camera defaults.
- Adjust harness references for `demo_v2_2` and `demo_v2_3` because those
  three-view entrypoint files are already removed in this worktree.

## Non-Goals

- Rewrite the shared three-view fused-PCD runtime internals.
- Change the aligned case directory or metadata compatibility contract.
- Fake or automate manual hardware calibration.

## Validation Plan

- Focused parser/default tests.
- Harness/catalog checks affected by removed demo entrypoints.
- Default `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
  if the current single-camera branch surface is internally consistent.
- Full `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full`
  because docs, harness entries, and demo command surfaces changed together.

## Status

- 2026-06-12: Started on branch `single-camera` after confirming
  `git pull --ff-only origin main` is up to date.
- 2026-06-12: Changed shared defaults to one camera, updated branch docs and
  harness coverage, and validated quick plus full deterministic checks.
