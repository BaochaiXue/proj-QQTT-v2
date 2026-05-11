# Demo 3 Tracking Benchmark Profile

- case: both_30_still_object_round1_20260428
- backends: cotracker3_online
- cameras: 0
- query_points: 10
- frames_requested: 2
- total_wall_ms: 1836.051
- frame_load_ms_total: 19.703
- mask_load_ms_total: 1.781
- max_rss_mb: 1451.137
- torch_cuda_peak_mb: 2309.697

## Serial Group FPS

| Backend | Points | Group FPS | E2E p50 ms | E2E p95 ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| cotracker3_online | 10 | 1.495 | 668.835 | 668.835 | serial scheduling |

## Per-Camera Rows

| Backend | Camera | Points | Frames | E2E ms | Visible | Inside Mask | Depth Valid | Lifted | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cotracker3_online | 0 | 10 | 2 | 668.835 | 1.000 | 1.000 | 0.100 | 1.000 | backend_load_ms=1108.029 |
