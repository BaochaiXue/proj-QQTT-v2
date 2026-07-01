# Demo v5.1 Reuse SAM3.1 For Shape Prior

## Requirement

Use the main-process SAM3.1 image runtime for the shape-prior high-resolution
object segmentation instead of launching a second SAM3.1 subprocess.

## Plan

- Keep `image_upscale.py` as the shape-prior subprocess that writes
  `shape/high_resolution.png`.
- Run high-resolution SAM3.1 segmentation by directly calling
  `demo_v5_1.sam31_image_segmentation.segment_image_to_origin_rgba()` from the
  main process shape-prior client.
- Cache and retain the frame-0 SAM3.1 image model whenever shape-prior warmup is
  enabled, so the shape-prior object prompt reuses that model cache.
- Update docs and focused tests so the shape-prior command chain no longer
  contains a SAM3.1 segmentation subprocess.

## Validation

- Passed: `python -m py_compile demo_v5_1/sam31_image_segmentation.py
  demo_v5_1/main_warmup.py demo_v5_1/shape_prior_warmup.py
  demo_v5_1/main_data_processing.py
  tests/test_demo_v5_1_shape_prior_simplification.py`
- Passed: `python -m pytest tests/test_demo_v5_1_shape_prior_simplification.py -q`
- Passed: `python scripts/harness/validation/run.py --profile smoke`
