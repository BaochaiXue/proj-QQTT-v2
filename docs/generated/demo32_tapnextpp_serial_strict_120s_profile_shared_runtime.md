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
- render FPS after warmup: `6.04`
- raw fusion FPS after warmup: `6.03`
- filter output FPS after warmup: `6.03`
- fusion FPS after warmup: `6.03`
- stage period p50 after warmup: `116.22 ms`
- display packet period p50 after warmup: `146.33 ms`
- groups after warmup: `3053`
- complete fused groups after warmup: `601`
- rendered groups after warmup: `601`
- complete group ratio after warmup: `0.197`
- stage drop count after warmup: `4`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `23.96`
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
| camera startup ms | `10699.68` |
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
| time to first complete group s | `34.35` |
| time to first rendered group s | `34.41` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `480`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `86.00` | `95.00` | `98.00` | `99.00` |
| `memory_util_pct` | `46.00` | `59.00` | `61.00` | `62.00` |
| `memory_used_mb` | `6609.00` | `7890.62` | `7909.99` | `8857.00` |
| `power_w` | `300.93` | `354.29` | `365.98` | `377.86` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2655.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `67.00` | `82.00` | `84.00` | `86.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.45` | `54.18` | `61.83` | `239.42` |
| `display_packet_publish_period_ms` | `146.33` | `256.21` | `315.66` | `547.98` |
| `edgetam_stage_publish_period_ms` | `69.61` | `97.77` | `106.74` | `355.02` |
| `ffs_stage_publish_period_ms` | `60.76` | `97.52` | `105.56` | `367.59` |
| `filter_output_publish_period_ms` | `147.15` | `254.71` | `313.03` | `538.88` |
| `fusion_publish_period_ms` | `147.15` | `254.71` | `313.03` | `538.88` |
| `gpu_owner_publish_period_ms` | `116.22` | `187.37` | `286.71` | `529.62` |
| `raw_fusion_publish_period_ms` | `147.06` | `248.63` | `292.53` | `529.51` |
| `render_period_ms` | `146.89` | `258.70` | `315.94` | `547.47` |
| `stage_join_publish_period_ms` | `116.22` | `187.37` | `286.71` | `529.62` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.33` | `24.56` | `27.43` | `54.37` |
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
| `gpu_owner_total_ms` | `64.96` | `69.83` | `72.23` | `105.63` |
| `gpu_owner_ffs_cycle_ms` | `62.45` | `65.38` | `66.52` | `105.63` |
| `gpu_owner_edgetam_cycle_ms` | `64.38` | `69.63` | `71.96` | `84.40` |
| `raw_fusion_total_ms` | `9.82` | `14.75` | `16.14` | `22.02` |
| `fusion_total_ms` | `86.10` | `96.90` | `248.46` | `274.15` |
| `filter_total_ms` | `75.20` | `85.16` | `236.56` | `266.44` |
| `filter_input_age_ms` | `75.22` | `85.19` | `236.59` | `266.47` |
| `object_enhanced_pt_ms` | `44.60` | `50.39` | `53.41` | `229.57` |
| `controller_pt_filter_ms` | `30.99` | `36.61` | `38.57` | `222.04` |
| `render_total_ms` | `3.98` | `5.18` | `5.69` | `17.92` |
| `render_queue_wait_ms` | `40.83` | `50.21` | `53.53` | `88.30` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.32` | `0.54` | `14.00` |
| `render_cpu_format_ms` | `0.39` | `0.82` | `1.80` | `14.64` |
| `render_open3d_points_update_ms` | `0.11` | `0.17` | `0.26` | `2.18` |
| `render_open3d_colors_update_ms` | `0.13` | `0.30` | `0.39` | `2.29` |
| `render_open3d_update_geometry_ms` | `3.41` | `4.30` | `4.58` | `8.52` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2769` | `229.57` | `24000` | `8805` |
| `3141` | `229.01` | `24000` | `8754` |
| `2558` | `227.73` | `24000` | `8773` |
| `1810` | `226.57` | `24000` | `8710` |
| `2186` | `222.57` | `24000` | `8751` |
| `2125` | `221.32` | `24000` | `8720` |
| `2336` | `220.45` | `24000` | `8640` |
| `2482` | `218.81` | `24000` | `8722` |
| `2405` | `218.44` | `24000` | `8744` |
| `1460` | `218.30` | `24000` | `8802` |
