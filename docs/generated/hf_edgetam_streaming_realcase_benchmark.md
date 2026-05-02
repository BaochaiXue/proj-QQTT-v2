# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-02T04:38:07.339672+00:00`
- Status: `pass`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Torch: `2.11.0+cu130`
- Transformers: `5.7.0`
- Jobs: `63/63` passed

## Contract

- Uses real aligned QQTT color frames, not synthetic frames.
- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.
- Measures frame-by-frame streaming with persistent inference sessions.
- This remains an experimental benchmark, not a production backend.

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box | 21 | 18.69 | 19.13 | 51.72 | 55.75 | 0.9606 |
| mask | 21 | 26.18 | 19.33 | 50.39 | 54.78 | 0.9671 |
| point | 21 | 18.68 | 19.33 | 50.88 | 55.00 | 0.9064 |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| still_object_round1 | 0 | point | 30 | 19.13 | 18.99 | 21.19 | 52.07 | 0.9727 | 0.9737 | 0 |
| still_object_round1 | 0 | box | 30 | 53.06 | 18.19 | 19.93 | 50.98 | 0.9701 | 0.9706 | 0 |
| still_object_round1 | 0 | mask | 30 | 42.63 | 18.52 | 20.18 | 51.71 | 0.9903 | 0.9917 | 0 |
| still_object_round1 | 1 | point | 30 | 17.77 | 18.64 | 21.29 | 53.14 | 0.3795 | 0.3804 | 0 |
| still_object_round1 | 1 | box | 30 | 17.83 | 18.28 | 20.85 | 54.25 | 0.9851 | 0.9854 | 0 |
| still_object_round1 | 1 | mask | 30 | 27.11 | 19.33 | 20.95 | 50.78 | 0.9851 | 0.9836 | 0 |
| still_object_round1 | 2 | point | 30 | 18.31 | 19.14 | 23.21 | 50.59 | 0.0001 | 0.0000 | 0 |
| still_object_round1 | 2 | box | 30 | 18.10 | 18.74 | 20.32 | 53.05 | 0.9746 | 0.9724 | 0 |
| still_object_round1 | 2 | mask | 30 | 28.28 | 19.64 | 25.65 | 48.14 | 0.9772 | 0.9792 | 0 |
| still_object_round2 | 0 | point | 30 | 19.00 | 19.51 | 21.26 | 50.73 | 0.0422 | 0.0425 | 0 |
| still_object_round2 | 0 | box | 30 | 19.27 | 19.06 | 20.72 | 52.32 | 0.9720 | 0.9736 | 0 |
| still_object_round2 | 0 | mask | 30 | 27.14 | 18.74 | 21.08 | 51.95 | 0.9739 | 0.9737 | 0 |
| still_object_round2 | 1 | point | 30 | 18.99 | 19.25 | 20.42 | 51.96 | 0.5790 | 0.5880 | 0 |
| still_object_round2 | 1 | box | 30 | 17.76 | 19.31 | 23.61 | 50.86 | 0.9697 | 0.9752 | 0 |
| still_object_round2 | 1 | mask | 30 | 26.86 | 18.94 | 21.42 | 51.49 | 0.9756 | 0.9765 | 0 |
| still_object_round2 | 2 | point | 30 | 18.22 | 19.25 | 20.69 | 51.57 | 0.0125 | 0.0125 | 0 |
| still_object_round2 | 2 | box | 30 | 19.89 | 19.00 | 19.86 | 52.78 | 0.8935 | 0.9082 | 0 |
| still_object_round2 | 2 | mask | 30 | 24.72 | 19.56 | 22.58 | 50.29 | 0.9823 | 0.9703 | 0 |
| still_object_round3 | 0 | point | 30 | 17.73 | 18.82 | 20.62 | 52.72 | 0.9869 | 0.9856 | 0 |
| still_object_round3 | 0 | box | 30 | 18.69 | 18.48 | 20.46 | 53.70 | 0.9882 | 0.9882 | 0 |
| still_object_round3 | 0 | mask | 30 | 23.36 | 19.22 | 24.05 | 50.24 | 0.9912 | 0.9912 | 0 |
| still_object_round3 | 1 | point | 30 | 18.68 | 19.04 | 20.60 | 52.19 | 0.3610 | 0.3725 | 0 |
| still_object_round3 | 1 | box | 30 | 18.47 | 18.11 | 19.86 | 54.19 | 0.8671 | 0.8416 | 0 |
| still_object_round3 | 1 | mask | 30 | 25.75 | 18.71 | 20.59 | 52.37 | 0.9817 | 0.9558 | 0 |
| still_object_round3 | 2 | point | 30 | 18.49 | 18.79 | 20.20 | 53.25 | 0.0024 | 0.0025 | 0 |
| still_object_round3 | 2 | box | 30 | 17.67 | 18.91 | 20.18 | 52.77 | 0.9128 | 0.8564 | 0 |
| still_object_round3 | 2 | mask | 30 | 25.20 | 19.28 | 21.96 | 50.67 | 0.9544 | 0.8948 | 0 |
| still_object_round4 | 0 | point | 30 | 19.92 | 20.12 | 23.05 | 48.35 | 0.0308 | 0.0308 | 0 |
| still_object_round4 | 0 | box | 30 | 18.97 | 19.44 | 21.25 | 51.43 | 0.9760 | 0.9788 | 0 |
| still_object_round4 | 0 | mask | 30 | 22.84 | 20.22 | 22.21 | 49.26 | 0.9785 | 0.9822 | 0 |
| still_object_round4 | 1 | point | 30 | 22.23 | 19.85 | 21.14 | 50.09 | 0.2874 | 0.2908 | 0 |
| still_object_round4 | 1 | box | 30 | 20.04 | 19.63 | 20.88 | 51.01 | 0.9668 | 0.9817 | 0 |
| still_object_round4 | 1 | mask | 30 | 25.35 | 19.33 | 22.23 | 50.39 | 0.9861 | 0.9689 | 0 |
| still_object_round4 | 2 | point | 30 | 17.96 | 19.33 | 20.72 | 51.73 | 0.1499 | 0.1478 | 0 |
| still_object_round4 | 2 | box | 30 | 18.46 | 19.28 | 20.45 | 51.55 | 0.9759 | 0.9688 | 0 |
| still_object_round4 | 2 | mask | 30 | 26.61 | 19.63 | 20.53 | 50.51 | 0.9682 | 0.9657 | 0 |
| still_rope_round1 | 0 | point | 30 | 18.95 | 19.83 | 21.24 | 50.42 | 0.0010 | 0.0015 | 0 |
| still_rope_round1 | 0 | box | 30 | 18.80 | 19.64 | 21.82 | 50.55 | 0.9520 | 0.9468 | 0 |
| still_rope_round1 | 0 | mask | 30 | 24.48 | 19.76 | 23.75 | 49.24 | 0.9774 | 0.9692 | 0 |
| still_rope_round1 | 1 | point | 30 | 18.97 | 19.52 | 21.30 | 50.88 | 0.6565 | 0.6616 | 0 |
| still_rope_round1 | 1 | box | 30 | 17.99 | 19.31 | 21.40 | 51.79 | 0.9452 | 0.9452 | 0 |
| still_rope_round1 | 1 | mask | 30 | 24.44 | 18.56 | 19.69 | 53.13 | 0.9487 | 0.9457 | 0 |
| still_rope_round1 | 2 | point | 30 | 19.13 | 19.66 | 22.75 | 50.39 | 0.0002 | 0.0002 | 0 |
| still_rope_round1 | 2 | box | 30 | 18.42 | 19.76 | 22.20 | 50.71 | 0.9495 | 0.9461 | 0 |
| still_rope_round1 | 2 | mask | 30 | 26.69 | 19.94 | 27.78 | 48.29 | 0.9734 | 0.9701 | 0 |
| still_rope_round2 | 0 | point | 30 | 17.38 | 19.86 | 21.91 | 50.27 | 0.0371 | 0.0367 | 0 |
| still_rope_round2 | 0 | box | 30 | 18.22 | 19.10 | 21.67 | 51.94 | 0.9624 | 0.9701 | 0 |
| still_rope_round2 | 0 | mask | 30 | 28.74 | 19.16 | 20.90 | 51.07 | 0.9514 | 0.9622 | 0 |
| still_rope_round2 | 1 | point | 30 | 18.58 | 19.19 | 20.89 | 51.41 | 0.0001 | 0.0001 | 0 |
| still_rope_round2 | 1 | box | 30 | 18.18 | 18.94 | 22.38 | 52.23 | 0.9679 | 0.9635 | 0 |
| still_rope_round2 | 1 | mask | 30 | 28.59 | 19.93 | 21.18 | 49.49 | 0.9688 | 0.9632 | 0 |
| still_rope_round2 | 2 | point | 30 | 21.15 | 19.91 | 21.43 | 49.37 | 0.9762 | 0.9673 | 0 |
| still_rope_round2 | 2 | box | 30 | 18.98 | 20.08 | 22.32 | 49.11 | 0.9798 | 0.9712 | 0 |
| still_rope_round2 | 2 | mask | 30 | 25.34 | 19.76 | 22.17 | 48.77 | 0.9792 | 0.9709 | 0 |
| ffs_dynamics_round1 | 0 | point | 30 | 18.17 | 19.70 | 21.18 | 50.68 | 0.9749 | 0.9760 | 0 |
| ffs_dynamics_round1 | 0 | box | 30 | 19.43 | 19.38 | 21.56 | 50.99 | 0.9727 | 0.9741 | 0 |
| ffs_dynamics_round1 | 0 | mask | 30 | 24.31 | 19.02 | 20.96 | 51.60 | 0.9775 | 0.9798 | 0 |
| ffs_dynamics_round1 | 1 | point | 30 | 17.61 | 19.81 | 22.38 | 49.85 | 0.8399 | 0.8354 | 0 |
| ffs_dynamics_round1 | 1 | box | 30 | 19.08 | 20.00 | 21.81 | 49.61 | 0.9670 | 0.9709 | 0 |
| ffs_dynamics_round1 | 1 | mask | 30 | 26.18 | 19.75 | 21.29 | 49.59 | 0.9673 | 0.9721 | 0 |
| ffs_dynamics_round1 | 2 | point | 30 | 18.96 | 18.96 | 20.63 | 52.41 | 0.9432 | 0.9445 | 0 |
| ffs_dynamics_round1 | 2 | box | 30 | 19.97 | 19.13 | 21.16 | 51.72 | 0.9167 | 0.9181 | 0 |
| ffs_dynamics_round1 | 2 | mask | 30 | 28.00 | 20.40 | 23.49 | 47.97 | 0.9846 | 0.9858 | 0 |
