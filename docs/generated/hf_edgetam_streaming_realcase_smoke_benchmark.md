# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-02T04:36:04.767708+00:00`
- Status: `pass`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Torch: `2.11.0+cu130`
- Transformers: `5.7.0`
- Jobs: `3/3` passed

## Contract

- Uses real aligned QQTT color frames, not synthetic frames.
- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.
- Measures frame-by-frame streaming with persistent inference sessions.
- This remains an experimental benchmark, not a production backend.

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box | 1 | 56.35 | 18.59 | 32.08 | 34.06 | 0.9834 |
| mask | 1 | 42.74 | 19.30 | 36.88 | 42.07 | 0.9884 |
| point | 1 | 20.29 | 19.88 | 49.97 | 54.81 | 0.9623 |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| still_object_round1 | 0 | point | 3 | 20.29 | 19.88 | 21.08 | 49.97 | 0.9639 | 0.9651 | 0 |
| still_object_round1 | 0 | box | 3 | 56.35 | 18.59 | 18.91 | 32.08 | 0.9695 | 0.9699 | 0 |
| still_object_round1 | 0 | mask | 3 | 42.74 | 19.30 | 19.60 | 36.88 | 0.9887 | 0.9907 | 0 |
