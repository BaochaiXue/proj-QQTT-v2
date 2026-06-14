# Demo 3.2 FFS TensorRT Serialization

## Summary

Demo 3.2 fake-live runs local FFS TensorRT depth and TAPNext++ marker overlay in the
same process. The PCD worker and tracker worker can both request local FFS depth for
the same mask frame, which allows two threads to enter the same TensorRT execution
context concurrently. On local runs this eventually trips TensorRT/Myelin with
`execute_async_v3 returned failure` and stops the demo.

## Plan

- Serialize local FFS TensorRT runner access inside the Demo 3.x runtime.
- Cache a small number of color-aligned FFS depth frames by replay sequence so the
  tracker lift can reuse depth already computed by the PCD worker.
- Keep remote FFS behavior unchanged.
- Add focused unit tests proving same-sequence FFS depth is cached and different
  sequences still compute fresh depth.
- Run targeted unit tests, quick harness, and a real Demo 3.2 fake-live window run.

## Validation

- PASS: `python -m py_compile qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_single_demo_tapnextpp_overlay.py`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay tests.test_single_demo_v3_runtime tests.test_realtime_masked_edgetam_pcd_filter tests.test_check_all_smoke`
- PASS: `git diff --check`
- PASS: Demo 3.2 fake-live non-headless window run reached the 3600-frame EOF and exited with code 0 using the local two-stage FFS TensorRT engine under `/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864`.
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full`
