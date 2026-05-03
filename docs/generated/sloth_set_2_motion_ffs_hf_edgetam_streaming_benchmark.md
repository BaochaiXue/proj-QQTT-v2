# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-03T18:34:37.857322+00:00`
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
- Reads one PNG frame at a time and calls the HF streaming model with a persistent inference session.
- Does not pass a full video path, MP4, or offline video-folder input to EdgeTAM.
- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.
- Measures frame-by-frame streaming with persistent inference sessions.
- This remains an experimental benchmark, not a production backend.

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mask | 3 | 23.60 | 18.93 | 51.88 | 55.80 | 0.3251 |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sloth_set_2_motion_ffs | 0 | mask | 93 | 23.60 | 18.78 | 20.42 | 52.62 | nan | nan | 0 |
| sloth_set_2_motion_ffs | 1 | mask | 93 | 25.93 | 18.93 | 21.14 | 51.88 | nan | nan | 0 |
| sloth_set_2_motion_ffs | 2 | mask | 93 | 23.54 | 20.24 | 23.07 | 49.26 | nan | nan | 0 |
