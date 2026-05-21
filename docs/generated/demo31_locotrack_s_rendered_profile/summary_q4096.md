# Demo 3.1 LocoTrack-S Rendered Profile: q4096 only

query_count_per_camera = 4096; duration = 60s; render-mode = pointcloud; overlay-display-scope = controller; overlay-max-points-per-camera = 0.

| execution_mode | window_frames | locotrack_batch_size | rendered_fps | tracker_publish_fps | tracker_model_ms_p50 | tracker_model_ms_p95 | tracker_e2e_ms_p50 | tracker_e2e_ms_p95 | rendered_groups | tracker_result_count | tracker_input_drop_count | overlay_blocked_count | gpu0_mem_p95_gb | gpu1_mem_p95_gb | gpu0_util_p95 | gpu1_util_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serial | 4 | 1 | 1.072 | 1.073 | 920.201 | 959.419 | 922.539 | 961.823 | 55 | 56 | 0 | 754 | 4.397 | 2.970 | 46.000 | 59.000 |
| serial | 8 | 1 | 0.836 | 0.841 | 1208.535 | 1261.710 | 1212.182 | 1265.406 | 43 | 44 | 0 | 764 | 4.397 | 3.182 | 45.000 | 75.000 |
| serial | 12 | 1 | 0.681 | 0.688 | 1554.412 | 1598.868 | 1559.265 | 1603.823 | 35 | 36 | 0 | 784 | 4.397 | 3.421 | 47.000 | 80.000 |
| batch-views | 4 | 3 | 1.845 | 1.851 | 531.436 | 555.474 | 534.089 | 558.692 | 97 | 98 | 0 | 731 | 4.397 | 4.059 | 47.000 | 83.000 |
| batch-views | 8 | 3 | 1.064 | 1.075 | 939.851 | 978.956 | 951.391 | 989.804 | 56 | 57 | 0 | 760 | 4.397 | 4.616 | 46.000 | 92.000 |
| batch-views | 12 | 3 | 0.770 | 0.784 | 1388.719 | 1437.231 | 1398.136 | 1446.759 | 41 | 42 | 0 | 790 | 4.397 | 5.340 | 46.000 | 95.000 |

Notes:
- rendered_fps comes from the shared runtime render loop (`summary_after_warmup.render_fps`).
- tracker latency/FPS comes from the Demo 3.1 LocoTrack child-process summary.
- RealSense camera 239222300412 reported a repeated 5000 ms frame timeout during runs, so these are live-system measurements with that camera caveat.
