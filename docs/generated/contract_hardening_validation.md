# Contract Hardening Validation

Date: 2026-04-09

## What Was Wrong Before

### Issue A

- `AGENTS.md` said aligned comparison visualization was in scope, but the invariants still said the repo stopped at `data_process/record_data_align.py`.
- the AGENTS file map still pointed only at the older `visual_compare_depth_video.py` path instead of the actual current compare entrypoints.

### Issue B

- `record_data.py` had a meaningful two-stage preflight flow, but the operator-facing contract was not explicit enough.
- when serials were not provided, the first preflight status was provisional, but that was not clearly labeled.
- blocked vs warning-only behavior existed, but the policy table was not cleanly exposed as a reusable contract.

## What Is Now Enforced

### Issue A

- `AGENTS.md` now states clearly that:
  - the primary data product stops at aligned cases
  - aligned native-vs-FFS comparison visualization is still in scope as a diagnostic utility
- the AGENTS file map now points to:
  - `visual_compare_depth_panels.py`
  - `visual_compare_reprojection.py`
  - `visual_compare_depth_video.py`
  - `visual_compare_turntable.py`

### Issue B

- record preflight policy is explicit in `qqtt/env/camera/preflight.py`
- current policy remains unchanged:
  - `rgbd`: allowed directly
  - `stereo_ir`: warning-only when unsupported
  - `both_eval`: blocked when unsupported
- `record_data.py` now prints clearer staged summaries:
  - `before camera discovery` when serials are pending
  - `before camera startup` when explicit serials are already known
  - `after camera discovery` once actual serials are resolved
- `unknown` support state now has explicit messaging, including the missing-probe-file case

## Tests That Prove The Fix

Added or updated:

- `tests/test_agents_scope_contract_smoke.py`
- `tests/test_record_preflight_policy_smoke.py`
- `tests/test_record_data_preflight_message_smoke.py`

These cover:

- AGENTS scope/file-map consistency
- explicit serials + blocked `both_eval`
- discovered serials + blocked `both_eval`
- discovered serials + `stereo_ir` warning path
- missing probe file / unknown support state

## Deterministic Validation

Command run:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness/check_all.py
```

Result:

- passed

## Compatibility Note

This pass intentionally did not change:

- `calibrate.pkl` format
- calibration producer behavior
- `load_calibration_transforms(...)` default semantics
- calibration serial-mapping semantics
