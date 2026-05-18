# Demo 2.1 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
- target FPS: `30.00`
- capture group target FPS: `30.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `11.87`
- raw fusion FPS after warmup: `11.89`
- filter output FPS after warmup: `11.88`
- fusion FPS after warmup: `11.88`
- stage period p50 after warmup: `69.35 ms`
- display packet period p50 after warmup: `69.16 ms`
- groups after warmup: `492`
- complete fused groups after warmup: `217`
- rendered groups after warmup: `216`
- complete group ratio after warmup: `0.441`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `18.13`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `11554.85` |
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
| time to first complete group s | `31.98` |
| time to first rendered group s | `31.99` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `106`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `50.00` | `56.00` | `57.00` | `63.00` |
| `memory_util_pct` | `13.00` | `43.50` | `45.00` | `49.00` |
| `memory_used_mb` | `3671.88` | `12927.28` | `12939.72` | `12941.12` |
| `power_w` | `164.49` | `279.90` | `286.23` | `290.01` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10501.00` | `10501.00` |
| `temperature_c` | `60.00` | `62.00` | `63.00` | `64.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.69` | `50.65` | `54.07` | `204.72` |
| `display_packet_publish_period_ms` | `69.16` | `117.83` | `122.48` | `256.97` |
| `edgetam_stage_publish_period_ms` | `61.63` | `72.15` | `76.02` | `221.93` |
| `ffs_stage_publish_period_ms` | `45.94` | `64.53` | `66.83` | `213.75` |
| `filter_output_publish_period_ms` | `69.16` | `117.83` | `122.47` | `256.96` |
| `fusion_publish_period_ms` | `69.16` | `117.83` | `122.47` | `256.96` |
| `gpu_owner_publish_period_ms` | `69.35` | `118.22` | `127.75` | `235.53` |
| `raw_fusion_publish_period_ms` | `69.34` | `118.22` | `127.76` | `235.53` |
| `render_period_ms` | `70.32` | `117.31` | `128.91` | `263.54` |
| `stage_join_publish_period_ms` | `69.35` | `118.22` | `127.75` | `235.53` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `9.30` | `25.93` | `28.37` | `31.99` |
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
| `gpu_owner_total_ms` | `55.14` | `58.58` | `59.92` | `75.02` |
| `gpu_owner_ffs_cycle_ms` | `40.13` | `42.85` | `44.89` | `58.52` |
| `gpu_owner_edgetam_cycle_ms` | `55.14` | `58.58` | `59.92` | `75.02` |
| `raw_fusion_total_ms` | `12.98` | `25.17` | `27.05` | `45.39` |
| `fusion_total_ms` | `63.72` | `77.84` | `93.25` | `227.25` |
| `filter_total_ms` | `49.69` | `54.63` | `58.46` | `203.64` |
| `filter_input_age_ms` | `50.31` | `55.56` | `58.97` | `204.45` |
| `object_enhanced_pt_ms` | `33.07` | `38.24` | `39.86` | `187.62` |
| `controller_pt_filter_ms` | `16.28` | `19.39` | `20.62` | `23.18` |
| `render_total_ms` | `2.17` | `4.52` | `5.33` | `18.41` |
| `render_queue_wait_ms` | `4.89` | `9.37` | `9.82` | `14.28` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.19` | `0.29` | `2.33` |
| `render_cpu_format_ms` | `0.29` | `0.54` | `0.74` | `4.53` |
| `render_open3d_points_update_ms` | `0.07` | `0.10` | `0.13` | `0.89` |
| `render_open3d_colors_update_ms` | `0.10` | `0.22` | `0.34` | `4.37` |
| `render_open3d_update_geometry_ms` | `1.76` | `4.15` | `4.82` | `18.11` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `925` | `187.62` | `48104` | `11675` |
| `987` | `183.36` | `48096` | `11692` |
| `682` | `183.30` | `48080` | `11737` |
| `1170` | `180.90` | `48051` | `11545` |
| `1043` | `177.65` | `48139` | `11797` |
| `1107` | `175.72` | `48009` | `11639` |
| `869` | `174.21` | `48115` | `11673` |
| `806` | `173.22` | `48053` | `11695` |
| `571` | `172.38` | `48058` | `11774` |
| `737` | `171.12` | `48084` | `11699` |
