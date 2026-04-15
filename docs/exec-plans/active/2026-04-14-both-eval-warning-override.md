# 2026-04-14 Both Eval Warning Override

## Goal

Allow `record_data.py --capture_mode both_eval` to proceed on the current machine even when the latest D455 stream probe marks the profile unsupported, while keeping the operator-facing state explicitly experimental.

## Non-Goals

- no claim that `both_eval` is now fully supported or stable
- no change to the underlying probe results
- no change to `rgbd` or `stereo_ir` semantics
- no change to aligned-case depth backend behavior

## Files To Touch

- `qqtt/env/camera/preflight.py`
- `record_data.py` tests that currently assume `both_eval` is blocked
- current operator docs and current generated validation notes

## Implementation Plan

1. change `both_eval` preflight policy from `block_if_probe_fails` to warning-allowed
2. preserve the explicit experimental warning path in `record_data.py`
3. update tests to assert `experimental_warning` instead of `blocked`
4. update README / workflows / hardware validation / current generated validation note to match the new operator contract

## Validation Plan

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_record_preflight_policy_smoke`
- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_record_data_preflight_message_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
