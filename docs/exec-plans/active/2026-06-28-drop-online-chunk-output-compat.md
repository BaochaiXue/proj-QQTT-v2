# Drop Demo v5 Online Chunk Output Compatibility Plan

Goal: remove the legacy `online_chunk_output.py` import shim support from Demo v5 and Demo v5.1 so callers use the canonical `chunked_final_data_output.py` module directly.

## Scope

- Delete `demo_v5/online_chunk_output.py`.
- Delete `demo_v5_1/online_chunk_output.py` from the current Demo v5.1 workspace.
- Remove `DemoV4OnlineOutputWriter` and `DemoV5OnlineOutputWriter` aliases from `demo_v5/chunked_final_data_output.py` and `demo_v5_1/chunked_final_data_output.py`.
- Keep `ChunkedFinalDataWriter`, `build_online_chunk`, and existing online chunk output behavior unchanged.
- Do not touch Demo v4 compatibility paths in this cleanup.

## Steps

- [x] Add a failing focused test that expects `demo_v5.online_chunk_output` to be absent and verifies `ChunkedFinalDataWriter` remains the canonical writer.
- [x] Remove the Demo v5 and Demo v5.1 compatibility wrapper files.
- [x] Remove old writer aliases from canonical modules.
- [x] Update active docs that still point validation commands or runtime paths at `online_chunk_output.py`.
- [x] Run the focused Demo v5 unit test and the repository smoke validation profile.

## Validation

- Red check: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin.DemoV5RealtimePhysTwinTest.test_legacy_online_chunk_output_imports_are_removed` failed before implementation because `demo_v5.online_chunk_output` still imported.
- Green check: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin.DemoV5RealtimePhysTwinTest.test_legacy_online_chunk_output_imports_are_removed` passed after removal.
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin` passed: 48 tests.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed: 302 tests.
- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5/chunked_final_data_output.py demo_v5_1/chunked_final_data_output.py` passed.
