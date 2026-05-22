# Demo 3.1 Default TAPNext++ Backend

## Goal

Make Demo 3.1 default to the `tapnextpp` point-tracker backend while keeping
Demo 3.2 on its LiteTracker default.

## Scope

- Changed Demo 3.1 parser/default contract backend from `cotracker3_online` to
  `tapnextpp`.
- Preserved explicit overrides for `cotracker3_online`, `litetracker`,
  `trackon2`, and `locotrack`.
- Added default TAPNext++ repo/checkpoint values so the dry-run contract and
  child process config are explicit.
- Updated focused contract/docs tests so the default backend is visible in dry
  run output and process config.

## Outcome

Demo 3.1 now reports `cotracker_backend=tapnextpp`,
`tracker_backend_family=tapnext`, and
`tracking_backend_online_semantics=stateful_frame_by_frame` by default.
Demo 3.2 still reports its LiteTracker batch-view default.

## Validation

- `python -m py_compile qqtt/demo/demo31_runtime.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract tests.test_demo32_trackable_mask_filter tests.test_demo3_cotracker_worker`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
