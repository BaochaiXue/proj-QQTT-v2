# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-03T19:11:52.869148+00:00`
- Status: `pass`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Compile mode: `vision-reduce-overhead`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Torch: `2.11.0+cu130`
- Transformers: `5.7.0`
- Jobs: `3/3` passed

## Contract

- Uses real aligned QQTT color frames, not synthetic frames.
- Reads one PNG frame at a time and calls the HF streaming model with a persistent inference session.
- Does not pass a full video path, MP4, or offline video-folder input to EdgeTAM.
- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.
- Measures frame-by-frame streaming with persistent inference sessions.
- This remains an experimental benchmark, not a production backend.

## Compile

- Mode: `vision-reduce-overhead`
- Enabled: `True`
- Applied targets: `vision_encoder`
- `torch.compile` mode: `reduce-overhead`
- `fullgraph`: `False`
- `dynamic`: `False`

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mask | 3 | 16.48 | 11.47 | 86.19 | 96.19 | 0.3251 |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sloth_set_2_motion_ffs | 0 | mask | 93 | 16.48 | 11.47 | 12.87 | 85.43 | nan | nan | 0 |
| sloth_set_2_motion_ffs | 1 | mask | 93 | 15.09 | 11.48 | 12.50 | 86.40 | nan | nan | 0 |
| sloth_set_2_motion_ffs | 2 | mask | 93 | 16.64 | 11.45 | 12.30 | 86.19 | nan | nan | 0 |
