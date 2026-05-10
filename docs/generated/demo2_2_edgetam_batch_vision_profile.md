# Demo 2.2 EdgeTAM Batch Vision Encoder Probe

Status: implemented and profiled.

This probe adds an explicit Demo 2.2 option:

```bash
--edgetam-batch-vision-encoder
```

The option batches the three camera RGB frames through
`HF EdgeTAMVideoModel.get_image_features()` once, splits the returned
`HW x batch x C` feature tensors, and writes each slice into the matching
per-camera `EdgeTamVideoInferenceSession` cache. The downstream video tracker
state remains independent per camera.

## Command

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --edgetam-batch-vision-encoder \
  --no-parallel-init \
  --duration-s 60 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_async_filter_batchvision_40s_warmup_20s_formal_profile.json \
  --debug
```

`--no-parallel-init` was used for this hardware run because a first attempt hit
a SAM3.1 / EdgeTAM concurrent import race in `torchvision.ops` during parallel
initialization. The steady-state timings below are still valid for the batch
vision path.

## Result

| Metric | Batch3 default reference | Batch vision probe |
| --- | ---: | ---: |
| render FPS | `5.37` | `5.25` |
| GPU owner median | `177.28 ms` | `182.64 ms` |
| FFS cycle median | `67.38 ms` | `73.69 ms` |
| EdgeTAM cycle median | `109.17 ms` | `107.46 ms` |
| EdgeTAM batch vision model median | n/a | `13.34 ms` |
| EdgeTAM batch vision total median | n/a | `20.97 ms` |
| EdgeTAM per-camera model median | n/a | `~27.5 ms` |

## Interpretation

The feature-cache path works: the per-camera EdgeTAM forward no longer pays the
full per-camera image encoder cost, and the profile records a separate batch
vision encoder stage. However, total EdgeTAM cycle median only improves by
about `1.7 ms` versus the previous batch3 default profile.

This is not enough to make it the default. Most remaining EdgeTAM time is in the
per-camera video tracker path: memory attention, mask decoder, memory encoder,
HF session bookkeeping, post-processing, and Python overhead. FFS/filter tail
latency also still affects render FPS.

## Current Recommendation

Keep Demo 2.2 default as:

```text
FFS TensorRT batch=3 + HF EdgeTAM shared model, per-camera sessions
```

Keep `--edgetam-batch-vision-encoder` as an explicit experiment option. The next
useful optimization is not just batching the image encoder; it would need a
deeper batched video-tracker scheduler or lower-overhead non-HF runtime for the
memory/decoder path.
