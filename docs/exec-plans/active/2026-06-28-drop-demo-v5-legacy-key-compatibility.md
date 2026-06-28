# Drop Demo v5 Legacy Key Compatibility

Goal: remove Demo v5 and Demo v5.1 data_process legacy key aliases so only
canonical query, track, topology, and case-root field names are accepted.

## Scope

- Delete `demo_v5/data_process_schema.py` and `demo_v5_1/data_process_schema.py`.
- Replace all `normalize_data_process_keys` calls in Demo v5/v5.1 runtime code
  with plain `dict(...)` copies where a local mutable mapping is needed.
- Keep Demo v4 and `vendor/realtime_phystwin` legacy contracts untouched.
- Preserve existing non-key cleanup work already present in the tree.

## Validation

- Add focused tests proving old `controller_anchor_*`/`controller_fps_indices`
  payloads are rejected instead of normalized.
- Add static source coverage for removed legacy key helper names and exact key
  spellings under `demo_v5` and `demo_v5_1`.
- Run targeted pytest, compileall, and the repository smoke validation profile.

## Results

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_legacy_key_cleanup.py -q` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_demo_v5_no_compat_wrappers.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_harness_catalog.py` passed.
- `conda run -n demo_2_max --no-capture-output python -m compileall -q demo_v5 demo_v5_1 tests/test_demo_v5_legacy_key_cleanup.py` passed.
- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_phystwin_strict_product.py -q` could not run because this workspace has no `tests/test_phystwin_strict_product.py`.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed help checks and guards, then failed in the unittest batch due this workspace importing/missing unrelated `tests.*` modules from `/home/xinjie/proj-QQTT-v2/tests` plus existing camera color override expectation mismatches.
