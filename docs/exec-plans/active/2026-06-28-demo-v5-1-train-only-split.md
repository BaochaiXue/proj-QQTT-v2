# Demo v5.1 Single-Chunk Split Cleanup

Goal: remove the generic `_split_payload` helper and stop emitting
`split.json` from Demo v5.1 single-chunk data-process cases. Single chunks are
already atomic windows, so they should not carry an internal train/test split.
The aggregate static case may still emit `split.json`, but it writes it
directly at the aggregate output point and uses every frame as training data.

## Scope

- Remove single-chunk split generation and validation from
  `demo_v5_1/data_process_chunk_writer.py`.
- Keep aggregate static-case split generation in
  `demo_v5_1/chunked_final_data_aggregate.py`, with all frames assigned to
  `train` and an empty `test` range, without a reusable `_split_payload`
  helper.
- Add focused coverage proving chunk cases no longer emit `split.json`.

## Checklist

- [x] Add a failing focused test for single-chunk split removal.
- [x] Remove single-chunk split generation and validation.
- [x] Keep aggregate split train-only.
- [x] Remove the generic `_split_payload` helper from Demo v5.1 code.
- [x] Run focused tests and compile checks.
- [x] Re-run smoke validation after removing chunk-level split files.

## Validation Results

- Red check:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_split_payload`
  failed before the chunk cleanup because
  `demo_v5_1.data_process_chunk_writer` still exposed `_split_payload`.
- Green check:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_split_payload`
  passed after removing chunk-level `split.json` and the generic
  `_split_payload` helper.
- Related regression:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_split_payload tests.test_demo_v5_legacy_key_cleanup tests.test_demo_v5_1_default_config tests.test_demo_v5_1_tools_io`
  passed.
- Compile check:
  `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/data_process_chunk_writer.py demo_v5_1/chunked_final_data_aggregate.py tests/test_demo_v5_1_split_payload.py`
  passed.
- Smoke validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed.
