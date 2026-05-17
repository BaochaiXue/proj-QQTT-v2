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
- raw fusion FPS after warmup: `5.80`
- filter output FPS after warmup: `5.80`
- fusion FPS after warmup: `5.80`
- stage period p50 after warmup: `164.83 ms`
- display packet period p50 after warmup: `164.56 ms`
- groups after warmup: `1268`
- complete fused groups after warmup: `528`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.416`
- stage drop count after warmup: `18`
- raw fused pending replacements total: `0`
- render buffer dropped total: `638`
- target deficit: `15.00`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `8430.77` |
| camera startup ms | `8790.43` |
| EdgeTAM model load ms | `2538.16` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1718.15` |
| EdgeTAM warmup/first forward ms | `90.67` |
| SAM3.1 model load ms | `9836.69` |
| SAM3.1 cam0 segment ms | `505.04` |
| SAM3.1 cam1 segment ms | `274.82` |
| SAM3.1 cam2 segment ms | `227.97` |
| FFS runner init ms | `8290.74` |
| FFS first run ms | `1156.36` |
| session init + prompt add ms | `4.84` |
| SAM3.1 release cleanup ms | `244.17` |
| time to first complete group s | `22.33` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `182`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `53.00` | `59.00` | `59.00` | `63.00` |
| `memory_util_pct` | `15.00` | `18.90` | `19.00` | `26.00` |
| `memory_used_mb` | `11379.10` | `15907.70` | `16354.80` | `16757.10` |
| `power_w` | `120.09` | `147.40` | `164.25` | `245.80` |
| `sm_clock_mhz` | `232.00` | `1110.00` | `1110.00` | `1110.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `62.00` | `65.00` | `66.00` | `67.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.86` | `86.05` | `93.16` | `288.03` |
| `display_packet_publish_period_ms` | `164.56` | `181.32` | `194.61` | `397.94` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `164.56` | `181.36` | `194.61` | `397.93` |
| `fusion_publish_period_ms` | `164.56` | `181.36` | `194.61` | `397.93` |
| `gpu_owner_publish_period_ms` | `164.83` | `181.24` | `192.74` | `382.19` |
| `raw_fusion_publish_period_ms` | `164.65` | `181.65` | `192.99` | `383.62` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `36.97` | `56.00` | `56.69` | `65.32` |
| `edgetam_model_ms` | `23.31` | `27.41` | `29.19` | `54.01` |
| `edgetam_preprocess_ms` | `1.04` | `1.22` | `1.28` | `2.05` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.04` | `0.06` | `0.07` | `0.39` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.84` | `1.22` | `2.72` | `9.93` |
| `edgetam_total_ms` | `24.67` | `28.60` | `30.30` | `55.11` |
| `ffs_cycle_ms` | `69.63` | `76.74` | `80.93` | `286.19` |
| `ffs_batch_ms` | `49.66` | `53.64` | `57.04` | `254.72` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `12.39` | `14.98` | `16.09` | `24.13` |
| `edgetam_batch_vision_total_ms` | `19.90` | `23.32` | `24.68` | `30.54` |
| `edgetam_batch_vision_preprocess_ms` | `3.11` | `3.66` | `3.84` | `6.16` |
| `edgetam_cam0_model_ms` | `23.80` | `28.01` | `29.44` | `54.01` |
| `edgetam_cam1_model_ms` | `23.10` | `27.31` | `28.90` | `35.31` |
| `edgetam_cam2_model_ms` | `23.14` | `27.16` | `28.65` | `40.53` |
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
| `ffs_stage_ms` | `2.18` | `3.23` | `3.59` | `5.40` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.18` | `3.23` | `3.59` | `5.40` |
| `ffs_cam1_stage_ms` | `2.18` | `3.23` | `3.59` | `5.40` |
| `ffs_cam2_stage_ms` | `2.18` | `3.23` | `3.59` | `5.40` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `164.67` | `180.80` | `191.40` | `382.15` |
| `gpu_owner_ffs_cycle_ms` | `69.63` | `76.74` | `80.93` | `286.19` |
| `gpu_owner_edgetam_cycle_ms` | `94.19` | `103.73` | `108.27` | `137.80` |
| `raw_fusion_total_ms` | `10.28` | `11.95` | `12.49` | `14.88` |
| `fusion_total_ms` | `50.69` | `56.97` | `59.57` | `261.58` |
| `filter_total_ms` | `40.02` | `46.22` | `49.28` | `249.39` |
| `filter_input_age_ms` | `40.60` | `46.87` | `49.78` | `249.55` |
| `object_enhanced_pt_ms` | `32.96` | `38.91` | `41.14` | `242.01` |
| `controller_pt_filter_ms` | `7.02` | `8.11` | `8.44` | `10.09` |
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
| `691` | `242.01` | `50385` | `14015` |
| `762` | `240.33` | `50356` | `14017` |
| `613` | `237.24` | `50341` | `14117` |
| `1478` | `232.04` | `50346` | `14078` |
| `1046` | `231.00` | `50365` | `14057` |
| `1554` | `230.56` | `50423` | `14117` |
| `209` | `229.00` | `50137` | `14086` |
| `1627` | `227.76` | `50383` | `14039` |
| `1331` | `227.66` | `50387` | `14102` |
| `1259` | `225.76` | `50351` | `14053` |
