# Demo 3.1 TAPNext++ Model-Only Benchmark Summary

This excludes RealSense, masks, Open3D, IPC, lift, and render. `recurrent_update_ms_*` is the adapter-reported TAPNext++ model update time.

| batch_size | query_count_per_view | total_query_count | image_size | autocast_dtype | compile | first_update_ms | first_update_model_ms | recurrent_update_ms_p50 | recurrent_update_ms_p95 | preprocess_ms_p50 | preprocess_ms_p95 | postprocess_ms_p50 | postprocess_ms_p95 | cuda_event_ms_p50 | cuda_event_ms_p95 | wall_ms_p50 | wall_ms_p95 | measured_wall_fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1365 | 1365 | 256x256 | fp16 | no | 333.020 | 317.116 | 14.270 | 16.210 | 0.266 | 0.850 | 5.999 | 8.176 | 14.270 | 16.210 | 20.630 | 23.587 | 45.314 |
| 3 | 1365 | 4095 | 256x256 | fp16 | no | 61.421 | 17.951 | 16.316 | 19.396 | 0.948 | 1.450 | 42.293 | 42.573 | 16.316 | 19.396 | 59.597 | 63.363 | 16.301 |
