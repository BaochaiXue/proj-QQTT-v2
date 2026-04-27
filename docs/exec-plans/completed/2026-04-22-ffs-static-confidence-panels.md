# 2026-04-22 FFS Static Confidence Panels

## Goal

Add a dedicated offline static-round confidence visualization workflow that
reruns PyTorch Fast-FoundationStereo on static round frame `0`, derives
per-pixel confidence proxies from classifier logits, and renders masked `3x3`
boards for the three static FFS rounds.

## Non-Goals

- no TensorRT confidence path
- no native-vs-FFS comparison board in this workflow
- no new SAM generation policy; reuse existing static stuffed-animal masks
- no change to aligned-case primary outputs

## Files To Touch

- `data_process/depth_backends/fast_foundation_stereo.py`
- `data_process/depth_backends/geometry.py`
- `data_process/depth_backends/__init__.py`
- `data_process/visualization/workflows/ffs_confidence_panels.py`
- `scripts/harness/visualize_ffs_static_confidence_panels.py`
- `tests/test_ffs_confidence_panels_smoke.py`
- `tests/test_ffs_tensorrt_single_engine_smoke.py`
- `docs/WORKFLOWS.md`
- `scripts/harness/README.md`
- `scripts/harness/check_all.py`
- this exec plan

## Implementation Plan

1. add small, testable confidence helpers under the FFS backend:
   - logits -> `max_softmax` and `margin`
   - hook-based classifier capture during PyTorch forward
   - confidence maps upsampled/unpadded back to final disparity shape
2. add a scalar reprojection helper that aligns arbitrary IR-left scalar maps to
   color coordinates using the same nearest-depth winner policy as
   `align_depth_to_color`
3. add a new static-only visualization workflow that:
   - resolves the three static FFS rounds
   - loads frame `0`
   - reruns FFS once per camera
   - aligns depth and both confidence metrics to color
   - applies the existing static stuffed-animal masks
   - writes per-round `margin` and `max_softmax` `3x3` boards plus summary JSON
4. expose the workflow through a thin harness CLI with repo-local defaults for
   FFS repo/model path and static output root
5. add software-only tests for confidence formulas, scalar reprojection, static
   path/mask resolution, and workflow artifact writing
6. update user-facing docs and deterministic harness coverage for the new CLI

## Validation Plan

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_confidence_panels_smoke tests.test_ffs_tensorrt_single_engine_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/visualize_ffs_static_confidence_panels.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`

## Risks

- the confidence proxy depends on internal classifier logits and is not a
  calibrated probability
- the hook-based PyTorch path assumes the upstream classifier module is invoked
  exactly once per forward pass in the current model code
