# Demo 3.3 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
- target FPS: `30.00`
- capture group target FPS: `30.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- EdgeTAM live session keep frames: `64`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `5.58`
- raw fusion FPS after warmup: `5.58`
- filter output FPS after warmup: `5.58`
- fusion FPS after warmup: `5.58`
- stage period p50 after warmup: `118.36 ms`
- display packet period p50 after warmup: `158.48 ms`
- groups after warmup: `7841`
- complete fused groups after warmup: `1564`
- rendered groups after warmup: `1563`
- complete group ratio after warmup: `0.199`
- stage drop count after warmup: `6`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.42`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Shape Prior Warmup

- enabled: `False`
- status: `disabled`
- case dir: `result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/<run_id>/case`
- object points0: `0`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10723.27` |
| EdgeTAM model load ms | `n/a` |
| EdgeTAM compile wrap ms | `n/a` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `n/a` |
| SAM3.1 model load ms | `n/a` |
| SAM3.1 cam0 segment ms | `n/a` |
| SAM3.1 cam1 segment ms | `n/a` |
| SAM3.1 cam2 segment ms | `n/a` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `n/a` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `33.65` |
| time to first rendered group s | `33.69` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `1194`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `87.00` | `97.00` | `98.00` | `99.00` |
| `memory_util_pct` | `47.00` | `59.00` | `60.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `7943.06` | `7970.00` | `8365.44` |
| `power_w` | `298.55` | `345.32` | `358.16` | `374.49` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2655.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `66.00` | `89.00` | `90.00` | `91.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.57` | `54.32` | `60.42` | `291.20` |
| `display_packet_publish_period_ms` | `158.48` | `282.63` | `328.31` | `625.93` |
| `edgetam_stage_publish_period_ms` | `69.77` | `99.18` | `108.14` | `326.45` |
| `ffs_stage_publish_period_ms` | `60.90` | `101.84` | `113.51` | `356.75` |
| `filter_output_publish_period_ms` | `158.53` | `285.48` | `330.49` | `620.58` |
| `fusion_publish_period_ms` | `158.53` | `285.48` | `330.49` | `620.58` |
| `gpu_owner_publish_period_ms` | `118.35` | `230.96` | `294.41` | `605.10` |
| `raw_fusion_publish_period_ms` | `158.41` | `278.00` | `300.10` | `621.36` |
| `render_period_ms` | `158.30` | `285.69` | `329.81` | `625.82` |
| `stage_join_publish_period_ms` | `118.36` | `230.96` | `294.41` | `605.10` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `7.98` | `15.81` | `21.81` | `64.55` |
| `edgetam_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `64.46` | `69.68` | `71.35` | `110.05` |
| `gpu_owner_ffs_cycle_ms` | `61.83` | `64.92` | `65.98` | `110.05` |
| `gpu_owner_edgetam_cycle_ms` | `63.82` | `69.48` | `71.10` | `91.36` |
| `raw_fusion_total_ms` | `9.56` | `15.64` | `17.33` | `29.14` |
| `fusion_total_ms` | `90.00` | `101.32` | `253.74` | `317.22` |
| `filter_total_ms` | `79.40` | `89.20` | `244.84` | `306.13` |
| `filter_input_age_ms` | `79.44` | `89.22` | `244.87` | `306.15` |
| `object_enhanced_pt_ms` | `45.09` | `51.22` | `53.51` | `271.09` |
| `controller_pt_filter_ms` | `34.48` | `40.29` | `43.82` | `263.27` |
| `render_total_ms` | `3.19` | `4.31` | `4.92` | `18.58` |
| `render_queue_wait_ms` | `31.68` | `45.08` | `47.76` | `68.69` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.11` | `0.37` | `0.60` | `2.75` |
| `render_cpu_format_ms` | `0.36` | `0.80` | `1.11` | `16.85` |
| `render_open3d_points_update_ms` | `0.10` | `0.15` | `0.22` | `2.39` |
| `render_open3d_colors_update_ms` | `0.10` | `0.27` | `0.33` | `16.53` |
| `render_open3d_update_geometry_ms` | `2.70` | `3.62` | `3.85` | `10.29` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `1.39` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `6815` | `271.09` | `24000` | `9110` |
| `7153` | `257.94` | `24000` | `9184` |
| `6021` | `256.21` | `24000` | `9114` |
| `6105` | `254.61` | `24000` | `9203` |
| `6663` | `245.61` | `24000` | `9047` |
| `4979` | `243.57` | `24000` | `9100` |
| `7226` | `243.12` | `24000` | `9069` |
| `6436` | `242.68` | `24000` | `9233` |
| `6509` | `241.79` | `24000` | `9140` |
| `4687` | `240.40` | `24000` | `8887` |
