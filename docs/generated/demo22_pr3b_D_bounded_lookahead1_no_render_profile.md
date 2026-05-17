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
- raw fusion FPS after warmup: `3.84`
- filter output FPS after warmup: `3.72`
- fusion FPS after warmup: `3.72`
- stage period p50 after warmup: `255.08 ms`
- display packet period p50 after warmup: `254.68 ms`
- groups after warmup: `1226`
- complete fused groups after warmup: `323`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.263`
- stage drop count after warmup: `42`
- raw fused pending replacements total: `0`
- render buffer dropped total: `408`
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
| parallel init max wait ms | `3702.13` |
| camera startup ms | `6034.05` |
| EdgeTAM model load ms | `1146.36` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1499.36` |
| EdgeTAM warmup/first forward ms | `200.48` |
| SAM3.1 model load ms | `10032.95` |
| SAM3.1 cam0 segment ms | `827.12` |
| SAM3.1 cam1 segment ms | `530.57` |
| SAM3.1 cam2 segment ms | `437.37` |
| FFS runner init ms | `5016.95` |
| FFS first run ms | `1073.45` |
| session init + prompt add ms | `6.13` |
| SAM3.1 release cleanup ms | `259.89` |
| time to first complete group s | `17.24` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `175`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `62.00` | `70.00` | `73.00` | `77.00` |
| `memory_util_pct` | `23.00` | `26.00` | `27.30` | `39.00` |
| `memory_used_mb` | `8891.10` | `11783.50` | `12193.70` | `12399.10` |
| `power_w` | `127.47` | `161.07` | `171.87` | `248.35` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `1110.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `67.00` | `70.00` | `71.00` | `71.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.84` | `82.03` | `92.31` | `337.15` |
| `display_packet_publish_period_ms` | `254.68` | `286.17` | `403.30` | `765.28` |
| `edgetam_stage_publish_period_ms` | `254.80` | `276.80` | `304.07` | `497.26` |
| `ffs_stage_publish_period_ms` | `79.20` | `108.05` | `112.30` | `349.01` |
| `filter_output_publish_period_ms` | `254.69` | `286.16` | `403.31` | `765.27` |
| `fusion_publish_period_ms` | `254.69` | `286.16` | `403.31` | `765.27` |
| `gpu_owner_publish_period_ms` | `255.08` | `275.81` | `305.02` | `492.30` |
| `raw_fusion_publish_period_ms` | `255.07` | `275.82` | `305.01` | `492.31` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `255.08` | `275.81` | `305.02` | `492.30` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `33.15` | `51.01` | `56.75` | `66.14` |
| `edgetam_model_ms` | `47.03` | `57.08` | `60.40` | `72.94` |
| `edgetam_preprocess_ms` | `1.50` | `2.01` | `2.46` | `79.86` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.05` | `0.11` | `0.15` | `1.66` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `14.91` | `21.62` | `22.90` | `29.41` |
| `edgetam_total_ms` | `62.91` | `72.56` | `75.31` | `96.77` |
| `ffs_cycle_ms` | `78.79` | `107.81` | `111.29` | `339.65` |
| `ffs_batch_ms` | `51.86` | `57.17` | `59.84` | `298.14` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `63.17` | `75.20` | `78.79` | `297.93` |
| `edgetam_batch_vision_total_ms` | `73.10` | `84.97` | `88.90` | `307.84` |
| `edgetam_batch_vision_preprocess_ms` | `4.49` | `6.02` | `7.28` | `239.59` |
| `edgetam_cam0_model_ms` | `41.54` | `54.19` | `60.52` | `70.50` |
| `edgetam_cam1_model_ms` | `47.03` | `56.29` | `58.70` | `72.94` |
| `edgetam_cam2_model_ms` | `50.62` | `57.99` | `61.24` | `71.54` |
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
| `ffs_stage_ms` | `1.86` | `2.84` | `3.36` | `238.27` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.86` | `2.83` | `3.36` | `238.27` |
| `ffs_cam1_stage_ms` | `1.86` | `2.83` | `3.36` | `238.27` |
| `ffs_cam2_stage_ms` | `1.86` | `2.83` | `3.36` | `238.27` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `254.25` | `276.44` | `302.56` | `497.23` |
| `gpu_owner_ffs_cycle_ms` | `77.74` | `88.97` | `96.24` | `319.14` |
| `gpu_owner_edgetam_cycle_ms` | `254.25` | `276.44` | `302.56` | `497.23` |
| `raw_fusion_total_ms` | `9.20` | `11.99` | `12.62` | `16.60` |
| `fusion_total_ms` | `45.49` | `59.31` | `62.33` | `283.20` |
| `filter_total_ms` | `36.84` | `48.15` | `51.90` | `270.62` |
| `filter_input_age_ms` | `37.45` | `48.39` | `52.40` | `270.94` |
| `object_enhanced_pt_ms` | `26.42` | `37.38` | `41.37` | `260.48` |
| `controller_pt_filter_ms` | `10.21` | `12.70` | `13.66` | `21.24` |
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
| `929` | `260.48` | `50691` | `14214` |
| `743` | `248.82` | `36409` | `10018` |
| `1242` | `244.75` | `50588` | `14073` |
| `1081` | `244.13` | `50570` | `14168` |
| `1412` | `238.13` | `30848` | `9513` |
| `1629` | `231.56` | `50841` | `14073` |
| `454` | `228.93` | `30526` | `9291` |
| `305` | `224.72` | `34301` | `9381` |
| `592` | `219.26` | `30683` | `9523` |
| `360` | `47.35` | `50763` | `14119` |
