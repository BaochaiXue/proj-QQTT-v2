# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-02T04:38:40.772662+00:00`
- Status: `pass`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Torch: `2.11.0+cu130`
- Transformers: `5.7.0`
- Jobs: `1/1` passed

## Contract

- Uses real aligned QQTT color frames, not synthetic frames.
- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.
- Measures frame-by-frame streaming with persistent inference sessions.
- This remains an experimental benchmark, not a production backend.

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_mask | 1 | 23.31 | 24.23 | 41.80 | 58.48 | 0.9989 |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| still_object_round1 | 0 | previous_mask | 3 | 23.31 | 24.23 | 25.71 | 41.80 | 0.9874 | 0.9897 | 0 |
