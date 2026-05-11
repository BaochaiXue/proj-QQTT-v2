# Demo 3 Tracking Benchmark Profile

- case: both_30_still_object_round1_20260428
- backends: fake
- cameras: 0, 1, 2
- query_points: 100, 256, 512, 1024
- frames_requested: 30
- total_wall_ms: 1221.909
- frame_load_ms_total: 674.238
- mask_load_ms_total: 107.713
- max_rss_mb: 697.777
- torch_cuda_peak_mb: 0.000

## Serial Group FPS

| Backend | Points | Group FPS | E2E p50 ms | E2E p95 ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| fake | 100 | 33932.438 | 0.030 | 0.033 | serial scheduling |
| fake | 256 | 29502.007 | 0.030 | 0.043 | serial scheduling |
| fake | 512 | 23598.262 | 0.036 | 0.064 | serial scheduling |
| fake | 1024 | 20582.199 | 0.045 | 0.062 | serial scheduling |

## Per-Camera Rows

| Backend | Camera | Points | Frames | E2E ms | Visible | Inside Mask | Depth Valid | Lifted | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fake | 0 | 100 | 30 | 0.033 | 1.000 | 0.992 | 0.069 | 6.867 | backend_load_ms=0.005 |
| fake | 1 | 100 | 30 | 0.026 | 1.000 | 1.000 | 0.946 | 94.567 | backend_load_ms=0.005 |
| fake | 2 | 100 | 30 | 0.030 | 1.000 | 0.996 | 0.986 | 98.633 | backend_load_ms=0.005 |
| fake | 0 | 256 | 30 | 0.030 | 1.000 | 0.991 | 0.057 | 14.700 | backend_load_ms=0.005 |
| fake | 1 | 256 | 30 | 0.026 | 1.000 | 0.994 | 0.962 | 246.333 | backend_load_ms=0.005 |
| fake | 2 | 256 | 30 | 0.045 | 1.000 | 0.996 | 0.967 | 247.600 | backend_load_ms=0.005 |
| fake | 0 | 512 | 30 | 0.036 | 1.000 | 0.997 | 0.072 | 36.867 | backend_load_ms=0.005 |
| fake | 1 | 512 | 30 | 0.068 | 1.000 | 0.999 | 0.935 | 478.933 | backend_load_ms=0.005 |
| fake | 2 | 512 | 30 | 0.024 | 1.000 | 0.994 | 0.969 | 495.967 | backend_load_ms=0.005 |
| fake | 0 | 1024 | 30 | 0.045 | 1.000 | 0.998 | 0.071 | 72.600 | backend_load_ms=0.005 |
| fake | 1 | 1024 | 30 | 0.036 | 1.000 | 0.998 | 0.950 | 972.933 | backend_load_ms=0.005 |
| fake | 2 | 1024 | 30 | 0.064 | 1.000 | 0.997 | 0.965 | 988.167 | backend_load_ms=0.005 |
