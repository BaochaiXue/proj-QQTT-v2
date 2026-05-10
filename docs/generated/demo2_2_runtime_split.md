# Demo 2.2 Runtime Split

Date: 2026-05-10

## What Changed

Demo 2.2 and Demo 2.1.5 public wrappers now import the dedicated `demo_v2_2.runtime` boundary instead of importing `demo_v2_1.realtime_three_view_masked_fused_pcd` directly.

Updated public entrypoints:

- `demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py`
- `demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py`

New runtime boundary:

- `demo_v2_2/runtime.py`

## Why

`demo_v2_1/realtime_three_view_masked_fused_pcd.py` has accumulated historical Demo 2.1 presets, probes, compatibility switches, and Demo 2.2 / 2.1.5 behavior in one file. The first cleanup step is to stop new demos from treating that file as their public API.

## Current Boundary

`demo_v2_2.runtime` owns the Demo 2.2 / 2.1.5 import surface:

- Demo 2.2 / 2.1.5 presets
- runtime class alias
- contract helpers
- parser / preset helpers
- GPU sampling helper
- packet dataclasses used by tests

The old Demo 2.1 file remains the internal implementation source for now, so existing Demo 2.1 tests and historical entrypoints are not broken. The next cleanup pass can move internals from the legacy file into smaller modules behind this boundary.

## Validation

- `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_5_realsense_depth_smoke`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
