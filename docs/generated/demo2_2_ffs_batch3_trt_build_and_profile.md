# Demo 2.2 FFS Batch3 TensorRT Build And Profile

## Build

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python scripts/ffs_trt/build_batch3_5090_engine.py --debug
```

Artifact path:

```text
result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3
```

Build result:

```text
status: pass
static batch size: 3
model: 20-30-48
valid_iters: 4
engine input: 480x864
builderOptimizationLevel: 5
feature build: 23014.24 ms
post build: 45790.88 ms
```

The batch=3 artifacts are under `result/` and are not committed to git. The
existing batch=1 TensorRT artifact path is unchanged.

## Runtime Smoke

Synthetic 3-sample `FastFoundationStereoTensorRTRunner.run_batch()` after one
warmup:

```text
elapsed: 49.07 ms
outputs: 3
depth shape: 480x848
all outputs finite and positive
input staging: pinned
```

## Demo 2.2 Profile

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-async-filter-5fps \
  --duration-s 40 \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_2_async_filter_batch3_ffs_20s_warmup_20s_formal_profile.json \
  --debug
```

The Demo 2.2 async-filter preset now defaults to `--ffs-trt-batch-size 3`; the
flag was required when this profile was first generated, but is no longer
needed for the default wrapper path.

Profile output:

```text
docs/generated/demo2_2_async_filter_batch3_ffs_20s_warmup_20s_formal_profile.md
docs/generated/demo2_2_async_filter_batch3_ffs_20s_warmup_20s_formal_profile.json
```

Key result after warmup:

```text
render FPS:           5.37
raw fusion FPS:       5.38
filter FPS:           5.37
capture group FPS:   13.22
FFS batch median:    49.02 ms
FFS cycle median:    67.38 ms
EdgeTAM median:     109.17 ms
GPU owner median:   177.28 ms
```

Compared with the previous batch=1 parallel-init profile:

```text
FFS cycle median:     90.67 -> 67.38 ms
GPU owner median:    197.08 -> 177.28 ms
render FPS:            4.92 -> 5.37
```

Batch=3 improves the median path enough to clear the 5 FPS local demo target,
but p90/p95 still show limited headroom because EdgeTAM remains sequential and
there are occasional FFS/filter spikes.
