# Demo 3.2 performance profile

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
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `22.43`
- filter output FPS after warmup: `22.43`
- fusion FPS after warmup: `22.43`
- stage period p50 after warmup: `44.45 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `3377`
- complete fused groups after warmup: `2367`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.701`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `30.00`
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
| camera startup ms | `10936.52` |
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
| time to first complete group s | `28.92` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `490`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `54.00` | `63.00` | `65.00` | `67.00` |
| `memory_util_pct` | `13.00` | `51.00` | `52.00` | `55.00` |
| `memory_used_mb` | `4175.38` | `6863.27` | `6893.74` | `9167.38` |
| `power_w` | `169.60` | `303.17` | `304.84` | `309.65` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `63.00` | `66.00` | `67.00` | `68.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.53` | `38.07` | `38.90` | `194.32` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `60.85` | `72.42` | `74.69` | `209.13` |
| `ffs_stage_publish_period_ms` | `44.47` | `48.48` | `49.90` | `188.23` |
| `filter_output_publish_period_ms` | `44.65` | `49.14` | `50.42` | `190.93` |
| `fusion_publish_period_ms` | `44.66` | `49.14` | `50.41` | `190.93` |
| `gpu_owner_publish_period_ms` | `44.45` | `48.63` | `49.68` | `189.45` |
| `raw_fusion_publish_period_ms` | `44.44` | `48.58` | `49.67` | `189.45` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `44.45` | `48.63` | `49.68` | `189.45` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `16.37` | `28.32` | `28.52` | `44.21` |
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
| `gpu_owner_total_ms` | `56.70` | `60.18` | `61.08` | `14419.40` |
| `gpu_owner_ffs_cycle_ms` | `39.33` | `40.88` | `41.46` | `45.65` |
| `gpu_owner_edgetam_cycle_ms` | `56.70` | `60.18` | `61.08` | `14419.40` |
| `raw_fusion_total_ms` | `10.13` | `12.92` | `13.63` | `19.01` |
| `fusion_total_ms` | `23.60` | `27.16` | `28.10` | `33.20` |
| `filter_total_ms` | `13.13` | `15.40` | `16.02` | `20.14` |
| `filter_input_age_ms` | `13.75` | `16.08` | `16.74` | `20.49` |
| `object_enhanced_pt_ms` | `4.17` | `5.25` | `5.83` | `8.00` |
| `controller_pt_filter_ms` | `8.77` | `10.75` | `11.42` | `15.27` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_queue_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_cpu_format_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_points_update_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_colors_update_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `786` | `8.00` | `24000` | `6244` |
| `3329` | `7.81` | `24000` | `6224` |
| `1282` | `7.79` | `24000` | `6258` |
| `1255` | `7.70` | `24000` | `6214` |
| `2522` | `7.46` | `24000` | `6232` |
| `2646` | `7.35` | `24000` | `6262` |
| `1484` | `7.29` | `24000` | `6183` |
| `1306` | `7.27` | `24000` | `6202` |
| `3305` | `7.21` | `24000` | `6227` |
| `2212` | `7.15` | `24000` | `6217` |
