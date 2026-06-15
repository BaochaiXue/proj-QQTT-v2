# Demo 3.2 PCD Mask Erosion

## Problem

Demo 3.2 FFS point clouds can still show a thin ring of floor/background
points around the sloth and controller. The async point-cloud filter can be too
stale to render every frame, so the renderer sometimes falls back to raw PCD
that already contains FFS/mask-boundary depth artifacts.

## Plan

- Add an explicit realtime masked-PCD option, `--pcd-mask-erode-pixels`, that
  shrinks controller/object masks before RGB-D backprojection.
- Keep the default at `0` for generic Demo 3/3.1 behavior.
- When Demo 3.2/3.3 enables FFS PCD filtering, apply a small default erosion
  unless the operator explicitly overrides it.
- Include the value in metadata, dry-run contract, and delegate argv so it is
  visible in logs and reproducible from harness output.
- Add tests for mask erosion behavior, validation, wrapper forwarding, and FFS
  surface defaults.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime tests.test_realtime_masked_edgetam_pcd_filter`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Manual Demo 3.2 fake-live smoke with `--enable-pcd-filter --debug`.
