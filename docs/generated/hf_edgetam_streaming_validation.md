# HF EdgeTAM Streaming Validation

- Timestamp UTC: `2026-05-02T03:00:34.744918+00:00`
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
- Session init mode: `edgetam`
- Session class: `transformers.models.edgetam_video.modeling_edgetam_video.EdgeTamVideoInferenceSession`
- Streaming session initialized without a complete video.
- Frame 0 point prompt was added with `original_size`.

## Synthetic Streaming Smoke

- Frames: `10`
- Frame size: `540x960`
- Dtype: `bfloat16`
- Mean latency ms: `74.951`
- Median latency ms: `18.239`
- Min latency ms: `16.594`
- Max latency ms: `566.514`

## Per Frame

| frame | latency_ms | mask_shape | mask_mean |
| ---: | ---: | --- | ---: |
| 0 | 566.514 | `[1, 1, 540, 960]` | -11.886794 |
| 1 | 37.136 | `[1, 1, 540, 960]` | -20.810583 |
| 2 | 19.677 | `[1, 1, 540, 960]` | -18.138641 |
| 3 | 18.047 | `[1, 1, 540, 960]` | -20.348970 |
| 4 | 17.977 | `[1, 1, 540, 960]` | -19.220907 |
| 5 | 16.594 | `[1, 1, 540, 960]` | -17.227367 |
| 6 | 18.389 | `[1, 1, 540, 960]` | -21.149357 |
| 7 | 18.089 | `[1, 1, 540, 960]` | -21.719955 |
| 8 | 19.591 | `[1, 1, 540, 960]` | -19.132315 |
| 9 | 17.495 | `[1, 1, 540, 960]` | -18.460169 |

## Conclusion

HF EdgeTAMVideo streaming proof-of-life passed on synthetic frames. This confirms the session-style protocol works separately from the patched official EdgeTAM backend.
