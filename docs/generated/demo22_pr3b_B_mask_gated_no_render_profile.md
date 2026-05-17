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
- raw fusion FPS after warmup: `5.44`
- filter output FPS after warmup: `5.44`
- fusion FPS after warmup: `5.44`
- stage period p50 after warmup: `175.36 ms`
- display packet period p50 after warmup: `175.31 ms`
- groups after warmup: `1200`
- complete fused groups after warmup: `473`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.394`
- stage drop count after warmup: `31`
- raw fused pending replacements total: `0`
- render buffer dropped total: `594`
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
| parallel init max wait ms | `9444.84` |
| camera startup ms | `6070.28` |
| EdgeTAM model load ms | `3290.71` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1422.30` |
| EdgeTAM warmup/first forward ms | `80.01` |
| SAM3.1 model load ms | `8689.75` |
| SAM3.1 cam0 segment ms | `557.60` |
| SAM3.1 cam1 segment ms | `270.66` |
| SAM3.1 cam2 segment ms | `185.79` |
| FFS runner init ms | `4939.03` |
| FFS first run ms | `1200.08` |
| session init + prompt add ms | `5.90` |
| SAM3.1 release cleanup ms | `249.13` |
| time to first complete group s | `19.43` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `173`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `49.00` | `57.00` | `59.00` | `63.00` |
| `memory_util_pct` | `15.00` | `20.80` | `21.00` | `27.00` |
| `memory_used_mb` | `11357.10` | `15288.30` | `15785.50` | `16109.10` |
| `power_w` | `119.01` | `168.05` | `184.01` | `246.33` |
| `sm_clock_mhz` | `180.00` | `1110.00` | `1110.00` | `1110.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `64.00` | `67.00` | `68.00` | `69.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.80` | `86.07` | `94.91` | `349.34` |
| `display_packet_publish_period_ms` | `175.31` | `194.87` | `208.01` | `421.54` |
| `edgetam_stage_publish_period_ms` | `175.80` | `192.46` | `204.58` | `406.99` |
| `ffs_stage_publish_period_ms` | `175.42` | `193.89` | `205.13` | `407.82` |
| `filter_output_publish_period_ms` | `175.31` | `194.85` | `208.02` | `421.54` |
| `fusion_publish_period_ms` | `175.31` | `194.85` | `208.01` | `421.54` |
| `gpu_owner_publish_period_ms` | `175.36` | `193.20` | `205.49` | `408.31` |
| `raw_fusion_publish_period_ms` | `175.36` | `193.22` | `205.49` | `408.30` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `175.36` | `193.21` | `205.49` | `408.30` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `15.18` | `49.99` | `54.30` | `66.64` |
| `edgetam_model_ms` | `44.48` | `69.55` | `73.77` | `290.39` |
| `edgetam_preprocess_ms` | `1.25` | `1.66` | `1.83` | `5.29` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.05` | `0.09` | `0.12` | `4.12` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.92` | `9.28` | `10.29` | `12.73` |
| `edgetam_total_ms` | `52.48` | `70.59` | `74.87` | `291.45` |
| `ffs_cycle_ms` | `76.68` | `83.19` | `84.72` | `93.43` |
| `ffs_batch_ms` | `55.96` | `58.92` | `60.27` | `65.25` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `18.67` | `22.70` | `24.20` | `33.36` |
| `edgetam_batch_vision_total_ms` | `28.21` | `32.46` | `34.20` | `42.35` |
| `edgetam_batch_vision_preprocess_ms` | `3.76` | `4.97` | `5.49` | `15.88` |
| `edgetam_cam0_model_ms` | `44.48` | `51.83` | `54.69` | `98.07` |
| `edgetam_cam1_model_ms` | `66.40` | `76.66` | `82.10` | `290.39` |
| `edgetam_cam2_model_ms` | `25.81` | `32.15` | `34.84` | `60.69` |
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
| `ffs_stage_ms` | `2.13` | `3.31` | `3.79` | `8.60` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.13` | `3.31` | `3.78` | `8.60` |
| `ffs_cam1_stage_ms` | `2.13` | `3.31` | `3.78` | `8.60` |
| `ffs_cam2_stage_ms` | `2.13` | `3.31` | `3.78` | `8.60` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `175.72` | `192.12` | `204.57` | `406.97` |
| `gpu_owner_ffs_cycle_ms` | `76.68` | `83.19` | `84.72` | `93.43` |
| `gpu_owner_edgetam_cycle_ms` | `175.72` | `192.12` | `204.57` | `406.97` |
| `raw_fusion_total_ms` | `9.70` | `11.75` | `12.75` | `17.06` |
| `fusion_total_ms` | `53.39` | `60.15` | `63.66` | `274.45` |
| `filter_total_ms` | `43.43` | `49.10` | `52.74` | `263.92` |
| `filter_input_age_ms` | `44.04` | `49.60` | `53.60` | `264.64` |
| `object_enhanced_pt_ms` | `36.11` | `41.27` | `44.43` | `255.27` |
| `controller_pt_filter_ms` | `7.29` | `8.56` | `8.97` | `11.17` |
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
| `1304` | `255.27` | `50453` | `14152` |
| `1385` | `252.01` | `50479` | `14225` |
| `565` | `247.98` | `50432` | `14135` |
| `1461` | `246.63` | `50529` | `14148` |
| `1224` | `245.39` | `50508` | `13999` |
| `1150` | `244.83` | `50537` | `14145` |
| `1076` | `243.96` | `50485` | `14070` |
| `1535` | `243.13` | `50507` | `14126` |
| `1614` | `242.21` | `50509` | `14209` |
| `785` | `235.36` | `50480` | `14175` |
