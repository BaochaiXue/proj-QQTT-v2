# Demo 2.2 CLI Simplification

Status: implemented.

Demo 2.2 now has its own small public CLI in:

```bash
demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py
```

The wrapper still delegates to the Demo 2.1 runtime, but the daily `--help`
surface no longer exposes old profiling and scheduling internals such as
`--fusion-target-fps`, `--gpu-pipeline-mode`, or `--single-owner-order`.

## Recommended Commands

Default Demo 2.2 contract:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --dry-run
```

Hardware profile:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --duration-s 60 \
  --warmup-s 40 \
  --profile-json-output docs/generated/demo2_2_profile.json \
  --debug
```

Object-only debug run:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --object-only \
  --duration-s 30
```

## Public Aliases

| Demo 2.2 public flag | Underlying runtime flag |
| --- | --- |
| `--warmup-s` | `--profile-warmup-exclude-s` |
| `--min-depth-m` | `--depth-min-m` |
| `--max-depth-m` | `--depth-max-m` |
| `--object-only` | `--track-mode object-only` |
| `--controller-object` | `--track-mode controller-object` |
| `--experimental-edgetam-batch-vision` | `--edgetam-batch-vision-encoder` |
| `--experimental-staged-parallel` | `--preset demo2.2-staged-parallel-5fps` |
| `--ffs-batch-size` | `--ffs-trt-batch-size` |

## Compatibility

Unknown flags are still passed through to the underlying Demo 2.1 parser, so old
benchmark commands continue to work. Use:

```bash
python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py --advanced-help
```

to show the full legacy/runtime argument list.

## Default Contract

The default Demo 2.2 entry remains:

```text
3 cameras @ 15 FPS
FFS TensorRT batch=3
HF EdgeTAM shared model + one streaming session per camera
single-owner GPU path: FFS then EdgeTAM
SAM3.1 first-frame object/controller initialization
async latest-wins filter
filtered fused PCD render only
```
