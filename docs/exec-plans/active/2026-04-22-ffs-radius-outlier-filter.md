## Goal

Replace the repo's current FFS "main output stays raw" behavior with an optional Open3D radius-outlier filter that runs on each per-camera color-aligned FFS depth frame during alignment, writes the filtered result as the main FFS output, and archives the unfiltered FFS depth for workflows that still need true raw-vs-filtered comparisons.

## Non-Goals

- no live viewer integration in `cameras_viewer_FFS.py`
- no change to native RealSense `depth/` behavior when `--depth_backend both`
- no cross-camera fused filtering; filtering stays per camera and per aligned frame
- no change to the existing native-like auxiliary postprocess contract beyond making it consume the filtered main FFS depth

## Files To Touch

- new shared FFS radius-outlier helper under `data_process/depth_backends/`
- `data_process/record_data_align.py`
- `data_process/aligned_case_metadata.py`
- `data_process/visualization/io_case.py`
- triplet compare workflows that expose `ffs_raw`
- fixture helpers, alignment / loader / triplet / cleanup tests
- README / WORKFLOWS / ARCHITECTURE

## Validation Plan

- new helper unit tests for radius-outlier filtering semantics
- alignment smoke coverage for `--ffs_radius_outlier_filter` under `ffs` and `both`
- loader smoke coverage for `depth_source="ffs_raw"` archive preference and legacy fallback
- triplet workflow smoke coverage that preserves `ffs_raw` semantics on new archived cases
- cleanup smoke coverage that deletes `*_original*` FFS archive directories
- `python scripts/harness/check_all.py`
