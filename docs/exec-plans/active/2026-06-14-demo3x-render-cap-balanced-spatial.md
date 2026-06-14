# Demo 3.x Balanced Render Cap

## Summary

Demo 3.x masks and PCD packets can still contain both controller hands while the
Open3D display layer caps each layer to 5000 points. The current render cap sorts
points spatially and takes a linear slice, but it does not explicitly balance the
cap across spatial buckets. When one hand/controller region is sparse and another
region is dense, the sparse region can become visually underrepresented even
though the underlying PCD packet still has points.

## Plan

- Replace raw spatial-order slicing in `cap_render_points` with deterministic
  coarse spatial bucket balancing.
- Keep the public `--render-max-points-per-layer` contract unchanged.
- Add tests that sparse separated regions keep a visible quota under a heavy
  render cap.
- Re-run targeted tests and harness checks.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m py_compile qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_single_demo_tapnextpp_overlay.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay tests.test_single_demo_v3_runtime tests.test_realtime_masked_edgetam_pcd_filter tests.test_check_all_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full`
- Non-headless Demo 3.2 fake-live smoke:
  `timeout 95s conda run -n demo_2_max --no-capture-output python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --input-source fake-live --replay-fps 30 --enable-pcd-filter --debug --duration-s 55 --ffs-trt-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864`
  exited cleanly.
