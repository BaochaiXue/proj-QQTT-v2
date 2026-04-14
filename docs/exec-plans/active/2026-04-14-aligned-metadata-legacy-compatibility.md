# 2026-04-14 Aligned Metadata Legacy Compatibility

## Goal

Restore old `proj-QQTT` aligned-case compatibility for:

- `metadata.json`
- `calibrate.pkl`

while keeping current QQTT aligned metadata fields available through a sidecar file.

## Non-Goals

- no change to raw recording metadata under `data_collect/`
- no change to `calibrate.pkl`
- no change to aligned frame directory contents
- no migration of arbitrary historical exports outside current aligned-case roots

## Files To Touch

- `data_process/record_data_align.py`
- aligned metadata helper / visualization metadata loader
- visualization test fixtures and record-align smoke tests
- docs and deterministic check wiring

## Implementation Plan

1. write future aligned outputs as:
   - `metadata.json` with only legacy aligned fields
   - `metadata_ext.json` with QQTT extension fields
2. make aligned metadata loaders merge both files in memory when the sidecar exists
3. keep reads backward-compatible with older unsplit aligned metadata files
4. update fixtures/tests to follow the split contract where aligned metadata is involved
5. sync current aligned cases under `data/` to the new split format when requested

## Validation Plan

- `python -m unittest -v tests.test_record_data_align_smoke`
- `python -m unittest -v tests.test_record_data_align_ffs_smoke`
- `python -m unittest -v tests.test_record_data_align_both_smoke`
- `python -m unittest -v tests.test_aligned_metadata_loader_smoke`
- `python scripts/harness/check_all.py`
