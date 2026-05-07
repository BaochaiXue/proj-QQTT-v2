# Demo 2.1 Staged FFS Then Parallel EdgeTAM

Status: implemented; live benchmark pending.

## Goal

Add a Demo 2.1 staged GPU pipeline mode:

```text
CaptureGroup
  -> FFS stage: cam0 -> cam1 -> cam2, one shared TensorRT runner/context owner
  -> EdgeTAM stage: cam0/cam1/cam2 in parallel, one session/model per camera
  -> CompleteInferenceGroup
  -> fusion/filter/render
```

## Non-Goals

- no FFS model / valid iters / TensorRT engine changes
- no RealSense-depth fallback
- no saved-mask fallback
- no semantic filter changes
- no object/controller union before filtering

## Files

- `demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `tests/test_demo_v2_1_three_view_fused_pcd_smoke.py`
- `demo_v2_1/README.md`
- generated Demo 2.1 validation docs

## Validation

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`

## Validation Result

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py tests/test_demo_v2_1_three_view_fused_pcd_smoke.py` passed.
- `conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke` passed.
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py` passed.
