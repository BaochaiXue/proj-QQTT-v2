# Demo 3.1 Controller PCD Cap Before Query And Fusion

## Goal

Prevent the large towel/controller mask from dominating Demo 3.1 by capping the controller mask/PCD contribution per camera before CoTracker query-point initialization and before fused PCD construction.

## Plan

- [x] Add a deterministic per-camera controller mask cap shared by Demo 3 and Demo 3.1.
- [x] Expose the cap in CLI/runtime contracts, defaulting to fewer than 5000 controller points per view.
- [x] Apply the capped controller mask before tracking input publication and before calling the shared fused PCD builder.
- [x] Add smoke tests that prove tracking and fusion inputs see the capped controller mask.
- [x] Run targeted tests, deterministic harness, and diff checks.

## Notes

- This is not an overlay-display cap. The cap must happen before query selection and before PCD fusion.
- The current experiment controller is `towel`; the formal demo controller remains `hand`.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m py_compile qqtt/demo/demo3_runtime.py qqtt/demo/demo31_runtime.py qqtt/demo/services/profile_schema.py tests/test_demo3_contract.py tests/test_demo31_dual_gpu_contract.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_cotracker_worker tests.test_demo31_cotracker_process_config`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_contract tests.test_demo31_dual_gpu_contract tests.test_profile_schema`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
