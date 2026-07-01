# Demo v5.1 Origin Shape-Prior Input

## Goal

Make Demo v5.1 build its shape-prior input like the origin data process:

- first upscale the frame-0 object crop;
- then segment the high-resolution image with Demo v5.1-owned SAM3.1 code;
- then feed `shape/masked_image.png` into SAM3D.

## Changes

- Move the SAM3.1 image segmentation dependency used by Demo v5.1 into
  `demo_v5_1`.
- Add origin-compatible RGBA output helpers to `sam31_image_segmentation.py`.
- Remove the old `shape/sam3d_input_rgba.png` warmup input path.
- Pass the runtime object prompt into the shape-prior client.
- Update the Demo v5.1 pipeline docs and focused tests.

## Validation

- Passed: `python -m py_compile demo_v5_1/sam31_image_segmentation.py
  demo_v5_1/shape_prior_warmup.py
  tests/test_demo_v5_1_shape_prior_simplification.py`
- Passed: `python -m pytest tests/test_demo_v5_1_shape_prior_simplification.py -q`
- Passed: `python scripts/harness/validation/run.py --profile smoke`
