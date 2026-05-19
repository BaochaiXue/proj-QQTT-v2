# Demo 3.1 EdgeTAM Session Sync

## Goal

Make the Demo 2.3 bounded HF EdgeTAM live-session memory fix explicit in the
Demo 3 and Demo 3.1 adapters without changing their runtime architecture.

## Plan

- Confirm whether Demo 3.1 already delegates mask/fusion/render to the shared
  three-view runtime.
- Expose the EdgeTAM live-session keep-frame limit through Demo 3 and Demo 3.1
  CLI contracts.
- Forward the limit into the shared runtime argv used by Demo 3 / Demo 3.1.
- Update contract docs and smoke tests so the compatibility is visible.
- Run deterministic checks, commit only scoped files, and push `main`.

## Validation

- `python -m py_compile qqtt/demo/demo3_runtime.py qqtt/demo/demo31_runtime.py`
- Focused Demo 3 / Demo 3.1 contract tests
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Outcome

- Demo 3 and Demo 3.1 expose `--edgetam-live-session-keep-frames`.
- Demo 3 forwards that limit into the shared three-view runtime argv, so Demo
  3.1 inherits the bounded HF EdgeTAM session behavior through its Demo 3
  adapter.
- Demo 3 / 3.1 contracts and profile summaries now report the keep-frame limit
  and pruning state.
- Validation passed with the focused contract tests, dry-runs, and quick
  deterministic harness.
