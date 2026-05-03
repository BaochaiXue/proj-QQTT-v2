# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-03T19:16:29.768578+00:00`
- Status: `pass`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Compile mode: `none`
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

- Mode: `none`
- Enabled: `False`
- Applied targets: `none`
- `torch.compile` mode: `None`
- `fullgraph`: `False`
- `dynamic`: `False`

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mask | 3 | 23.27 | 16.24 | 60.83 | 65.22 | 0.3251 |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sloth_set_2_motion_ffs | 0 | mask | 93 | 23.27 | 18.38 | 20.38 | 55.17 | nan | nan | 0 |
| sloth_set_2_motion_ffs | 1 | mask | 93 | 20.27 | 16.24 | 17.71 | 60.83 | nan | nan | 0 |
| sloth_set_2_motion_ffs | 2 | mask | 93 | 23.38 | 16.02 | 17.23 | 61.70 | nan | nan | 0 |
