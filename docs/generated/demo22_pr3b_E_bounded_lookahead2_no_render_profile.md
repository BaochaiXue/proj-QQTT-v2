# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `0.00`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `3.90`
- filter output FPS after warmup: `3.66`
- fusion FPS after warmup: `3.66`
- stage period p50 after warmup: `251.40 ms`
- display packet period p50 after warmup: `250.66 ms`
- groups after warmup: `1223`
- complete fused groups after warmup: `319`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.261`
- stage drop count after warmup: `60`
- raw fused pending replacements total: `0`
- render buffer dropped total: `406`
- target deficit: `15.00`
- bottleneck class: `upstream_supply`
- GPU pipeline: `overlapped-stages`
- single-owner order: `cross_group_overlap`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `7610.12` |
| camera startup ms | `6124.49` |
| EdgeTAM model load ms | `2374.03` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1372.51` |
| EdgeTAM warmup/first forward ms | `227.11` |
| SAM3.1 model load ms | `9456.37` |
| SAM3.1 cam0 segment ms | `1135.91` |
| SAM3.1 cam1 segment ms | `527.67` |
| SAM3.1 cam2 segment ms | `483.35` |
| FFS runner init ms | `5360.19` |
| FFS first run ms | `1095.85` |
| session init + prompt add ms | `27.67` |
| SAM3.1 release cleanup ms | `245.42` |
| time to first complete group s | `18.12` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `177`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `62.00` | `72.40` | `75.00` | `80.00` |
| `memory_util_pct` | `23.00` | `27.00` | `28.00` | `30.00` |
| `memory_used_mb` | `9103.10` | `12000.70` | `12258.30` | `12423.10` |
| `power_w` | `130.72` | `164.37` | `206.17` | `232.74` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `180.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `68.00` | `70.00` | `71.00` | `72.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.82` | `82.13` | `93.02` | `374.40` |
| `display_packet_publish_period_ms` | `250.66` | `297.72` | `484.61` | `987.56` |
| `edgetam_stage_publish_period_ms` | `250.67` | `270.01` | `279.08` | `487.41` |
| `ffs_stage_publish_period_ms` | `76.81` | `107.84` | `112.33` | `332.59` |
| `filter_output_publish_period_ms` | `250.64` | `297.73` | `484.62` | `987.55` |
| `fusion_publish_period_ms` | `250.65` | `297.73` | `484.62` | `987.55` |
| `gpu_owner_publish_period_ms` | `251.40` | `269.71` | `283.03` | `485.49` |
| `raw_fusion_publish_period_ms` | `251.38` | `269.69` | `283.04` | `485.47` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `251.40` | `269.71` | `283.03` | `485.49` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `24.13` | `50.57` | `59.54` | `79.12` |
| `edgetam_model_ms` | `43.94` | `51.55` | `54.41` | `84.37` |
| `edgetam_preprocess_ms` | `1.46` | `1.93` | `2.57` | `16.49` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.05` | `0.09` | `0.12` | `2.87` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `17.41` | `22.97` | `24.19` | `30.95` |
| `edgetam_total_ms` | `62.95` | `70.89` | `73.30` | `106.11` |
| `ffs_cycle_ms` | `76.75` | `107.49` | `111.66` | `332.53` |
| `ffs_batch_ms` | `51.72` | `56.37` | `58.15` | `257.52` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `61.40` | `72.60` | `78.00` | `297.46` |
| `edgetam_batch_vision_total_ms` | `71.39` | `81.39` | `87.83` | `307.39` |
| `edgetam_batch_vision_preprocess_ms` | `4.39` | `5.79` | `7.70` | `49.46` |
| `edgetam_cam0_model_ms` | `37.10` | `46.19` | `50.81` | `84.37` |
| `edgetam_cam1_model_ms` | `44.37` | `51.69` | `54.41` | `64.69` |
| `edgetam_cam2_model_ms` | `47.36` | `53.07` | `55.29` | `63.38` |
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
| `ffs_stage_ms` | `1.85` | `2.83` | `3.23` | `23.57` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.85` | `2.83` | `3.23` | `23.57` |
| `ffs_cam1_stage_ms` | `1.85` | `2.83` | `3.23` | `23.57` |
| `ffs_cam2_stage_ms` | `1.85` | `2.83` | `3.23` | `23.57` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `250.59` | `269.74` | `279.06` | `487.38` |
| `gpu_owner_ffs_cycle_ms` | `75.67` | `85.49` | `94.09` | `283.08` |
| `gpu_owner_edgetam_cycle_ms` | `250.59` | `269.74` | `279.06` | `487.38` |
| `raw_fusion_total_ms` | `9.76` | `12.86` | `13.80` | `16.81` |
| `fusion_total_ms` | `44.68` | `59.81` | `63.94` | `282.57` |
| `filter_total_ms` | `35.83` | `48.78` | `52.48` | `272.15` |
| `filter_input_age_ms` | `36.35` | `49.14` | `53.26` | `272.52` |
| `object_enhanced_pt_ms` | `25.16` | `37.85` | `41.75` | `255.74` |
| `controller_pt_filter_ms` | `10.47` | `13.12` | `14.26` | `19.23` |
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
| `786` | `255.74` | `50823` | `14282` |
| `1667` | `251.42` | `50797` | `14282` |
| `1171` | `238.32` | `50803` | `14377` |
| `1326` | `234.72` | `50612` | `14176` |
| `985` | `231.76` | `50743` | `14195` |
| `1494` | `224.16` | `30794` | `9552` |
| `485` | `219.82` | `30731` | `9567` |
| `320` | `214.12` | `30807` | `9413` |
| `768` | `49.85` | `50761` | `14344` |
| `1671` | `49.48` | `50804` | `14366` |
