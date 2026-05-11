# Demo 3 Tracking Benchmark Profile

- case: both_30_still_object_round1_20260428
- backends: cotracker3_online
- cameras: 0, 1, 2
- query_points: 100, 256, 512, 1024
- frames_requested: 30
- total_wall_ms: 11143.320
- frame_load_ms_total: 598.344
- mask_load_ms_total: 96.329
- max_rss_mb: 1764.199
- torch_cuda_peak_mb: 2605.316

## Serial Group FPS

| Backend | Points | Group FPS | E2E p50 ms | E2E p95 ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| cotracker3_online | 100 | 1.853 | 388.160 | 824.270 | serial scheduling |
| cotracker3_online | 256 | 1.959 | 514.239 | 515.883 | serial scheduling |
| cotracker3_online | 512 | 1.365 | 734.804 | 736.198 | serial scheduling |
| cotracker3_online | 1024 | 0.859 | 1164.977 | 1170.442 | serial scheduling |

## Per-Camera Rows

| Backend | Camera | Points | Frames | E2E ms | Visible | Inside Mask | Depth Valid | Lifted | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cotracker3_online | 0 | 100 | 30 | 872.727 | 0.998 | 0.991 | 0.068 | 6.833 | backend_load_ms=976.407 |
| cotracker3_online | 1 | 100 | 30 | 357.956 | 1.000 | 1.000 | 0.946 | 94.567 | backend_load_ms=976.407 |
| cotracker3_online | 2 | 100 | 30 | 388.160 | 1.000 | 0.996 | 0.986 | 98.633 | backend_load_ms=976.407 |
| cotracker3_online | 0 | 256 | 30 | 514.239 | 1.000 | 0.993 | 0.056 | 14.433 | backend_load_ms=976.407 |
| cotracker3_online | 1 | 256 | 30 | 516.065 | 1.000 | 0.994 | 0.962 | 246.333 | backend_load_ms=976.407 |
| cotracker3_online | 2 | 256 | 30 | 501.044 | 1.000 | 0.996 | 0.967 | 247.600 | backend_load_ms=976.407 |
| cotracker3_online | 0 | 512 | 30 | 736.353 | 1.000 | 0.996 | 0.072 | 36.833 | backend_load_ms=976.407 |
| cotracker3_online | 1 | 512 | 30 | 727.246 | 1.000 | 0.999 | 0.935 | 478.833 | backend_load_ms=976.407 |
| cotracker3_online | 2 | 512 | 30 | 734.804 | 1.000 | 0.993 | 0.969 | 495.967 | backend_load_ms=976.407 |
| cotracker3_online | 0 | 1024 | 30 | 1171.049 | 1.000 | 0.997 | 0.071 | 72.967 | backend_load_ms=976.407 |
| cotracker3_online | 1 | 1024 | 30 | 1157.669 | 1.000 | 0.997 | 0.950 | 972.933 | backend_load_ms=976.407 |
| cotracker3_online | 2 | 1024 | 30 | 1164.977 | 1.000 | 0.996 | 0.965 | 988.167 | backend_load_ms=976.407 |
