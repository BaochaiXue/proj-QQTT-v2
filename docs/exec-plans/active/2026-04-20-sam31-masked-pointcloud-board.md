# 2026-04-20 SAM31 Masked Pointcloud Board

## Goal

Add an independent single-frame point-cloud compare workflow that uses SAM 3.1 object masks to compare `Native` and `FFS` before vs after background suppression under one fixed Open3D view.

## Non-Goals

- no PhysTwin runtime dependency or cross-project import
- no change to canonical aligned `depth/` outputs
- no turntable default-path changes
- no live-viewer integration
- no floating-point postprocess replacement in this pass

## Files To Touch

- new `data_process/visualization/workflows/masked_pointcloud_compare.py`
- new `scripts/harness/visual_compare_masked_pointcloud.py`
- `scripts/harness/check_all.py`
- `tests/visualization_test_utils.py`
- new mask compare smoke tests
- workflow / architecture / harness docs

## Implementation Plan

1. add a dedicated workflow that:
   - resolves one compare frame
   - loads unmasked fused `Native` and `FFS` clouds
   - loads per-camera union masks from `sam31_masks`
   - filters per-camera clouds with existing pixel-mask helpers
   - computes one shared crop and one shared oblique Open3D render view
   - writes one `2x2` board plus debug overlays and fused PLYs
2. keep SAM 3.1 generation outside the workflow in the harness CLI:
   - reuse existing `sam31_masks` when present
   - optionally generate a workflow-local sidecar when missing
   - never mutate aligned case metadata or depth outputs
3. add deterministic tests for:
   - mask union loading
   - require-existing failure
   - reuse-or-generate helper invocation
   - end-to-end workflow output contract
4. wire the new CLI and tests into deterministic checks
5. update docs so this workflow is explicit and separate from turntable/triplet flows

## Validation Plan

- `python scripts/harness/visual_compare_masked_pointcloud.py --help`
- `python -m unittest -v tests.test_masked_pointcloud_compare_smoke`
- `python scripts/harness/check_all.py`
