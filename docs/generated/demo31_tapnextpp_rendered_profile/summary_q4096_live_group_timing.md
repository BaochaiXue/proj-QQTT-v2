# Demo 3.1 TAPNext++ Rendered Profile Summary

- `q1365/view` is the real ~4000-total target: `1365 * 3 = 4095` points.
- `q4096/view` is a stress test: `4096 * 3 = 12288` points.
- 45 FPS requires recurrent tracker latency at or below `22.2 ms`.
- Treat a rendered profile as valid only when `rendered_groups_after_warmup > 0`.

| execution_mode | query_count_per_camera | total_query_count_across_views | target_class | rendered_fps | rendered_groups_after_warmup | valid_rendered_profile | tracker_publish_fps | tracker_group_wall_ms_p50 | tracker_group_wall_ms_p95 | tracker_model_ms_sum_per_group_p50 | tracker_model_ms_sum_per_group_p95 | tracker_model_ms_max_per_group_p50 | tracker_model_ms_max_per_group_p95 | per_camera_model_ms_p50_by_camera | model_calls_per_group | model_instances_expected | model_instances_actual | tracker_model_ms_p50 | tracker_model_ms_p95 | tracker_e2e_ms_p50 | tracker_e2e_ms_p95 | input_drop_count | result_drop_count | stale_overlay_count | lift_cache_miss_count | gpu0_mem_used_gb | gpu1_mem_used_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch-views | 4096 | 12288 | stress_12288_total | 6.409 | 243 | yes | 6.407 | 150.765 | 153.385 | 147.866 | 149.219 | 147.866 | 149.219 | {"0":147.86578369140625,"1":147.86578369140625,"2":147.86578369140625} | 1 | 1 | 1 | 147.866 | 149.219 | 150.950 | 153.588 | 0 | 0 | 0 | 0 | 3.398 | 8.438 |
| serial | 4096 | 12288 | stress_12288_total | 6.856 | 261 | yes | 6.856 | 140.253 | 143.504 | 136.046 | 138.072 | 47.548 | 48.580 | {"0":45.202857971191406,"1":43.37895965576172,"2":47.431434631347656} | 3 | 3 | 3 | 136.046 | 138.072 | 140.440 | 143.685 | 0 | 0 | 0 | 0 | 3.398 | 8.188 |
