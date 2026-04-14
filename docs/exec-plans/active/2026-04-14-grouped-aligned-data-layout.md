# 2026-04-14 Grouped Aligned Data Layout

## Goal

Reorganize aligned cases so downstream tasks can consume grouped layouts of the form:

- `data/<type>/<case_name>/`

while keeping visualization and case-loading workflows able to resolve grouped cases.

## Non-Goals

- no change to aligned case payload schema inside each case directory
- no deletion of auxiliary streams such as `depth_ffs_float_m/`, `ir_left/`, or `ir_right/`
- no change to `result/` handling

## Files To Touch

- grouped aligned-case resolution in visualization IO
- docs for grouped raw/aligned layouts
- deterministic smoke coverage for grouped case resolution
- local aligned data directories for the current static cases

## Implementation Plan

1. allow aligned-case resolution under `aligned_root` by:
   - direct relative subpath such as `static/my_case`
   - unique basename search when the grouped layout is nested
2. add smoke coverage for grouped same-case and two-case resolution plus ambiguity detection
3. update workflow and validation docs to describe grouped layouts
4. move the current static aligned cases under `data/static/`

## Validation Plan

- `python -m unittest -v tests.test_grouped_aligned_case_resolution_smoke`
- `python scripts/harness/check_all.py`
- manual inspection that current static cases live under `data/static/`
