# Demo v5.1 Single-View Full-ARAP Alignment Default

## Goal

Disable the single-camera conservative alignment gate by default so the
one-camera warmup runs the same shape-prior alignment quality path as the
original multi-camera PhysTwin flow:

- keypoint ARAP deformation (`deform_ARAP`);
- ray-casting ARAP registration (`deform_ARAP_ray_registration`) including the
  above-table clamp.

Commit `5d74ac8` made `camera_count == 1` return the rigidly posed SAM3D mesh
untouched (`align_single_view_conservative`). That preserved the mesh prior but
let any SAM3D shape error pass straight into `final_data.pkl`
surface/interior points. The user decided the default must match origin
quality; the conservative behavior stays available as an explicit opt-in.

## Changes

- `demo_v5_1/shape_prior_align.py`:
  - add `--single_view_alignment {full,conservative}` with default `full`;
  - select `align_single_view_conservative` only when `camera_count == 1`
    and the flag is `conservative`; every other case (including one camera by
    default) runs the origin full flow;
  - rename `align_multiview_vendor_compatible` to
    `align_full_vendor_compatible` since it is no longer multi-view-only, and
    drop the artificial `camera_count == 2` error (the full flow iterates
    `c2ws`/`w2cs` generically);
  - rewrite the divergence comment to document the new default and the
    opt-in.
- `tests/test_demo_v5_1_shape_prior_simplification.py`: replace the two gate
  tests with assertions for the full-flow default and the conservative
  opt-in.
- `demo_v5_1/pipeline.md`: document the align stage default and the opt-in
  flag.

## Validation

- `python -m py_compile demo_v5_1/shape_prior_align.py
  tests/test_demo_v5_1_shape_prior_simplification.py`
- `python -m pytest tests/test_demo_v5_1_shape_prior_simplification.py -q`
- `python scripts/harness/validation/run.py --profile smoke`
