# HF EdgeTAM Streaming Validation

- Timestamp UTC: `2026-05-02T02:57:54.360333+00:00`
- Status: `pass`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Torch: `2.11.0+cu130`
- Torch CUDA: `13.0`
- Transformers: `5.7.0`

## API

- `EdgeTamVideoModel`: import OK
- `Sam2VideoProcessor`: import OK
- `EdgeTamVideoInferenceSession`: import OK
- Session init mode: `processor`
- Session class: `transformers.models.sam2_video.modeling_sam2_video.Sam2VideoInferenceSession`
- Streaming session initialized without a complete video.
- Frame 0 point prompt was added with `original_size`.

## Synthetic Streaming Smoke

- Frames: `3`
- Frame size: `540x960`
- Dtype: `bfloat16`
- Mean latency ms: `252.950`
- Median latency ms: `36.505`
- Min latency ms: `18.597`
- Max latency ms: `703.749`

## Per Frame

| frame | latency_ms | mask_shape | mask_mean |
| ---: | ---: | --- | ---: |
| 0 | 703.749 | `[1, 1, 540, 960]` | -11.886794 |
| 1 | 36.505 | `[1, 1, 540, 960]` | -19.489017 |
| 2 | 18.597 | `[1, 1, 540, 960]` | -18.420883 |

## Conclusion

HF EdgeTAMVideo streaming proof-of-life passed on synthetic frames. This confirms the session-style protocol works separately from the patched official EdgeTAM backend.
