# 2026-04-19 Live 3-Cam FFS Benchmark

## Goal

Redesign FFS performance evaluation around the real `cameras_viewer_FFS.py` online path so timing reflects simultaneous 3-camera processing rather than single-camera saved-pair inference.

## Non-Goals

- no TensorRT integration in this pass
- no change to canonical aligned outputs
- no change to point-cloud / floating-point diagnostics
- no automatic optimization of viewer scheduling yet

## Files To Touch

- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- `docs/WORKFLOWS.md`
- `docs/HARDWARE_VALIDATION.md`
- `docs/generated/README.md`
- new live-benchmark validation note under `docs/generated/`

## Implementation Plan

1. add timed-run support to `cameras_viewer_FFS.py`
2. add periodic runtime stats logging for:
   - aggregate capture fps
   - aggregate FFS fps
   - per-camera capture fps
   - per-camera FFS fps
   - latest inference ms
   - latest capture-to-result sequence gap
3. keep stats formatting in pure helpers so deterministic smoke tests can cover it
4. run real 3-camera live experiments with the current best offline configs
5. document the realistic workflow and measured outcomes separately from the saved-pair benchmark

## Validation Plan

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- timed 3-camera live runs with `--duration-s` and `--stats-log-interval-s`
- `python scripts/harness/check_all.py`
