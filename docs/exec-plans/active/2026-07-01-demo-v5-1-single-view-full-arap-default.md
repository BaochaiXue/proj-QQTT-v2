# Demo v5.1 Single-View Full-ARAP Alignment

## Goal

Demo v5.1 shape-prior alignment always runs the same full PhysTwin quality
path for every available camera count:

- keypoint ARAP deformation (`deform_ARAP`);
- ray-casting ARAP registration (`deform_ARAP_ray_registration`) including the
  above-table clamp.

The older one-camera rigid-prior bypass is no longer a supported runtime mode.
Keeping it as a selectable branch made the warmup behavior depend on a demo
flag instead of the aligned-case data contract, and it let SAM3D shape error
pass straight into `final_data.pkl` surface/interior points.

## Changes

- `demo_v5_1/shape_prior_align.py`:
  - keep `align_full_vendor_compatible` as the only final alignment path;
  - keep the processed-mask camera-count validation;
  - remove the one-camera rigid-prior bypass function, CLI flag, and selector.
- `tests/test_demo_v5_1_shape_prior_simplification.py`: assert that the bypass
  symbols and selector do not return.

## Validation

- `python -m py_compile demo_v5_1/shape_prior_align.py`
- `python -m unittest -v tests.test_demo_v5_1_shape_prior_simplification`
- `python scripts/harness/validation/run.py --profile smoke`
