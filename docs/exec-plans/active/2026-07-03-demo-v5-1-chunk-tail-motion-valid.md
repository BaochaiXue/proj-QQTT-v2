# Demo v5.1 Chunk Tail Motion Valid

## Requirement

Problem:
Origin motion validity is a forward test over `t -> t+1`, so the final frame
inside each online chunk cannot be tested at publication time. The current
published motion-valid arrays therefore mark every chunk tail as false, which
causes consumers that render `visibility & motion_valid` to hide otherwise
visible object points.

Required final behavior:
Temporarily publish the untestable chunk-tail row as motion-valid for object,
selected controller, and controller candidate arrays. Keep the underlying
origin `motion_consistency()` semantics unchanged.

Constraints:
Stay on `single-camera`. Do not add new keys or CLI/config switches. Keep this
as an explicit temporary production-layer publishing rule.

## Plan

- [x] Confirm branch and note the `origin/main` fast-forward check cannot
  complete for the current `single-camera` branch.
- [x] Publish chunk-tail motion-valid rows as true after recovery decisions are
  complete.
- [x] Add tests for unchanged origin motion semantics and temporary published
  tail semantics across chunk 0 and later chunks.
- [x] Run focused and smoke validation.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/tracking.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v5_1_tracking`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
- Result: focused tracking tests passed, and smoke validation passed with 106
  tests.
