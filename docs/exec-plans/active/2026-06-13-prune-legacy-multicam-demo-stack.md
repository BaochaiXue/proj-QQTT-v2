# Prune Legacy Multicamera Demo Stack

## Objective

Remove historical three-camera, dual-GPU, batch=3, Demo 2.x/3.x tracking, and
shape-prior surfaces from the `single-camera` branch so the branch keeps only
the current single-camera demo/runtime and aligned-case diagnostics.

## Scope

- Delete legacy `demo_v3*` three-view entrypoint folders.
- Delete historical `qqtt.demo` three-view / Demo 2.2 / Demo 2.3 / Demo 3.1+
  runtime modules that are not needed by `demo_v3*`.
- Delete tests and harness scripts that only keep those historical paths alive.
- Remove docs and generated evidence tied to Demo 2.3, Demo 3.1/3.2/3.3,
  dual-GPU, batch=3, or shape-prior work.
- Update scope/docs/checks to describe the remaining single-camera branch.

## Non-Goals

- Remove the current single-camera RealSense, FFS, EdgeTAM, or aligned
  native-vs-FFS comparison workflows.
- Remove core camera calibration/recording/alignment compatibility.
- Remove `main`; it remains the protected three-camera baseline.

## Validation

- Focused import and smoke tests for single-camera demos and current checks.
- Default deterministic `scripts/harness/check_all.py`.

## Status

- 2026-06-13: Started on branch `single-camera` after `git pull --ff-only
  origin main` reported up to date.
