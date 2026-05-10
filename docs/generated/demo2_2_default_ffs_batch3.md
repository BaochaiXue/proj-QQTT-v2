# Demo 2.2 Default FFS Batch3

Status: implemented.

Demo 2.2's default wrapper now resolves the `demo2.2-async-filter-5fps` preset
to the isolated FFS TensorRT batch=3 engine path.

## Default Command

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --duration-s 40 \
  --profile-warmup-exclude-s 20
```

Resolved FFS contract:

```text
trt_batch_size: 3
batch3_isolated_artifact: true
trt_model_dir: result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3
```

## Rollback

Use this explicit override to return Demo 2.2 to the previous batch=1 engine:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --ffs-trt-batch-size 1
```

The global Demo 2.1 parser default remains batch=1. Only the Demo 2.2
async-filter preset defaults to batch=3.

## Reference Profile

The batch=3 hardware profile from the previous validation remains the current
reference:

```text
render FPS:          5.37
FFS cycle median:   67.38 ms
GPU owner median:  177.28 ms
```

Source report:
`docs/generated/demo2_2_ffs_batch3_trt_build_and_profile.md`.
