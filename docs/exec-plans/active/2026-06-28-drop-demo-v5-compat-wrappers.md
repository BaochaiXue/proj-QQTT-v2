# Drop Demo v5 Compatibility Wrappers

Goal: remove explicit legacy compatibility wrapper modules from Demo v5 and Demo v5.1 so runtime code uses canonical module names directly.

## Scope

- Delete top-level Demo v5 and Demo v5.1 Python files whose module docstring declares `Legacy compatibility wrapper`.
- Keep canonical implementations unchanged:
  - `data_process_chunk_writer.py`
  - `realtime_data_process_track.py`
  - `chunked_final_data_aggregate.py`
  - `visualize_track.py`
  - `realtime_dense_track.py`
  - `realtime_data_process_sam3d.py`
- Add a deterministic harness guard that fails if Demo v5 or Demo v5.1 gains new explicit legacy wrapper modules.
- Treat `demo_v5_1/realtime_dense_track.py` as a local Demo v5.1 runtime
  implementation, not a thin entrypoint that forwards to
  `qqtt.demo.realtime_masked_edgetam_pcd` or `demo_v5`.
- Later user direction expanded the cleanup beyond wrapper files for Demo v5.1:
  remove aggregate invariant compatibility matching so chunks are compared by
  the exact current contract, not by scalar/array or dtype-tolerant fallback
  rules.

## Checklist

- [x] Add guard and verify it fails against the current wrappers.
- [x] Delete explicit compatibility wrapper modules.
- [x] Register the guard in the harness catalog and validation runner.
- [x] Run focused guard, catalog, compile, and available validation checks.
- [x] Add a failing test that rejects a thin Demo v5.1 dense-track wrapper.
- [x] Copy the realtime masked PCD runtime into `demo_v5_1/realtime_dense_track.py`.
- [x] Make the copied v5.1 dense-track runtime prefer the current checkout on
      `sys.path`.
- [x] Remove the Demo v5.1 aggregate `_arrays_match` / `_require_matching_value`
      compatibility matcher and replace it with exact invariant checks.

## Validation Results

- Red check: `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_demo_v5_no_compat_wrappers.py` failed before deletion and listed 10 Demo v5 / Demo v5.1 wrapper modules.
- Green check: `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_demo_v5_no_compat_wrappers.py` passed after deletion.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_harness_catalog.py` passed.
- `conda run -n demo_2_max --no-capture-output python -m py_compile scripts/harness/guards/check_demo_v5_no_compat_wrappers.py demo_v5/data_process_chunk_writer.py demo_v5/realtime_data_process_track.py demo_v5/chunked_final_data_aggregate.py demo_v5/chunked_final_data_output.py demo_v5/realtime_dense_track.py demo_v5/realtime_data_process_sam3d.py demo_v5/visualize_track.py demo_v5_1/data_process_chunk_writer.py demo_v5_1/realtime_data_process_track.py demo_v5_1/chunked_final_data_aggregate.py demo_v5_1/chunked_final_data_output.py demo_v5_1/realtime_dense_track.py demo_v5_1/realtime_data_process_sam3d.py demo_v5_1/visualize_track.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_experiment_boundaries.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_visual_architecture.py` passed.
- `conda run -n demo_2_max --no-capture-output python -m scripts.harness.guards.check_scope` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` ran help checks and guards, then failed in the unittest batch because this workspace snapshot has no top-level `tests/` package and Python imported stale tests from `/home/xinjie/proj-QQTT-v2/tests` with unrelated missing experiment imports and color override expectations.
- Red check for local dense-track runtime:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_legacy_key_cleanup.DemoV5LegacyKeyCleanupTests.test_demo_v51_realtime_dense_track_is_local_runtime`
  failed while `demo_v5_1/realtime_dense_track.py` was still a 29-line wrapper.
- Green check after copying the runtime:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_legacy_key_cleanup.DemoV5LegacyKeyCleanupTests.test_demo_v51_realtime_dense_track_is_local_runtime`
  passed.
- `conda run -n demo_2_max --no-capture-output python demo_v5_1/realtime_dense_track.py --help`
  initially failed because the copied repo-root probe allowed an older checkout
  to win imports; it passed after the v5.1 local entrypoint was changed to put
  the current checkout first on `sys.path`.
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_legacy_key_cleanup tests.test_demo_v5_1_default_config` passed after updating config tests to import the current `demo_v5_1.main` entrypoint.
- `conda run -n demo_2_max --no-capture-output python demo_v5_1/main.py --dry-run` passed.
- Red check for aggregate invariant cleanup:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_aggregate_invariants`
  failed while `_arrays_match` and `_require_matching_value` still existed and
  scalar-string/singleton-array plus dtype-mismatched arrays were accepted.
- Green check:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_aggregate_invariants tests.test_demo_v5_1_split_payload tests.test_validation_smoke_manifest`
  passed after replacing the generic matcher with exact JSON/scalar/array
  invariant checks.
- Related regression:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_legacy_key_cleanup tests.test_demo_v5_1_default_config tests.test_demo_v5_1_aggregate_invariants tests.test_demo_v5_1_split_payload tests.test_demo_v5_1_tools_io tests.test_validation_smoke_manifest`
  passed.
- Fresh smoke validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed.
