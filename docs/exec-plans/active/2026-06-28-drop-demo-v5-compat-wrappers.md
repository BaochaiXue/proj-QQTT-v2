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
- Do not remove schema/data-field compatibility logic such as legacy key aliases; this cleanup is only about redundant import/entrypoint wrapper files.

## Checklist

- [x] Add guard and verify it fails against the current wrappers.
- [x] Delete explicit compatibility wrapper modules.
- [x] Register the guard in the harness catalog and validation runner.
- [x] Run focused guard, catalog, compile, and available validation checks.

## Validation Results

- Red check: `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_demo_v5_no_compat_wrappers.py` failed before deletion and listed 10 Demo v5 / Demo v5.1 wrapper modules.
- Green check: `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_demo_v5_no_compat_wrappers.py` passed after deletion.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_harness_catalog.py` passed.
- `conda run -n demo_2_max --no-capture-output python -m py_compile scripts/harness/guards/check_demo_v5_no_compat_wrappers.py demo_v5/data_process_chunk_writer.py demo_v5/realtime_data_process_track.py demo_v5/chunked_final_data_aggregate.py demo_v5/chunked_final_data_output.py demo_v5/realtime_dense_track.py demo_v5/realtime_data_process_sam3d.py demo_v5/visualize_track.py demo_v5_1/data_process_chunk_writer.py demo_v5_1/realtime_data_process_track.py demo_v5_1/chunked_final_data_aggregate.py demo_v5_1/chunked_final_data_output.py demo_v5_1/realtime_dense_track.py demo_v5_1/realtime_data_process_sam3d.py demo_v5_1/visualize_track.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_experiment_boundaries.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_visual_architecture.py` passed.
- `conda run -n demo_2_max --no-capture-output python -m scripts.harness.guards.check_scope` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` ran help checks and guards, then failed in the unittest batch because this workspace snapshot has no top-level `tests/` package and Python imported stale tests from `/home/xinjie/proj-QQTT-v2/tests` with unrelated missing experiment imports and color override expectations.
