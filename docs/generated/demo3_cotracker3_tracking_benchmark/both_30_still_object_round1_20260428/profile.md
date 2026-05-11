# Demo 3 Tracking Benchmark Profile

- case: both_30_still_object_round1_20260428
- backends: cotracker3_online
- cameras: 0, 1, 2
- query_points: 100, 256, 512, 1024
- frames_requested: 30
- total_wall_ms: 15239.101
- frame_load_ms_total: 657.400
- mask_load_ms_total: 93.028
- max_rss_mb: 1763.617
- torch_cuda_peak_mb: 2605.316

## Serial Group FPS

| Backend | Points | Group FPS | E2E p50 ms | E2E p95 ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| cotracker3_online | 100 | 1.211 | 586.150 | 1247.001 | serial scheduling |
| cotracker3_online | 256 | 1.402 | 713.287 | 719.718 | serial scheduling |
| cotracker3_online | 512 | 0.996 | 1012.185 | 1033.870 | serial scheduling |
| cotracker3_online | 1024 | 0.652 | 1539.325 | 1548.438 | serial scheduling |

## Per-Camera Rows

| Backend | Camera | Points | Frames | E2E ms | Visible | Inside Mask | Depth Valid | Lifted | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cotracker3_online | 0 | 100 | 30 | 1320.429 | 0.998 | 0.991 | 0.068 | 6.833 | backend_load_ms=1492.178 |
| cotracker3_online | 1 | 100 | 30 | 570.717 | 1.000 | 1.000 | 0.946 | 94.567 | backend_load_ms=1492.178 |
| cotracker3_online | 2 | 100 | 30 | 586.150 | 1.000 | 0.996 | 0.986 | 98.633 | backend_load_ms=1492.178 |
| cotracker3_online | 0 | 256 | 30 | 713.287 | 1.000 | 0.993 | 0.056 | 14.433 | backend_load_ms=1492.178 |
| cotracker3_online | 1 | 256 | 30 | 705.595 | 1.000 | 0.994 | 0.962 | 246.333 | backend_load_ms=1492.178 |
| cotracker3_online | 2 | 256 | 30 | 720.433 | 1.000 | 0.996 | 0.967 | 247.600 | backend_load_ms=1492.178 |
| cotracker3_online | 0 | 512 | 30 | 1012.185 | 1.000 | 0.996 | 0.072 | 36.833 | backend_load_ms=1492.178 |
| cotracker3_online | 1 | 512 | 30 | 1036.279 | 1.000 | 0.999 | 0.935 | 478.833 | backend_load_ms=1492.178 |
| cotracker3_online | 2 | 512 | 30 | 963.178 | 1.000 | 0.993 | 0.969 | 495.967 | backend_load_ms=1492.178 |
| cotracker3_online | 0 | 1024 | 30 | 1549.451 | 1.000 | 0.997 | 0.071 | 72.967 | backend_load_ms=1492.178 |
| cotracker3_online | 1 | 1024 | 30 | 1510.039 | 1.000 | 0.997 | 0.950 | 972.933 | backend_load_ms=1492.178 |
| cotracker3_online | 2 | 1024 | 30 | 1539.325 | 1.000 | 0.996 | 0.965 | 988.167 | backend_load_ms=1492.178 |
