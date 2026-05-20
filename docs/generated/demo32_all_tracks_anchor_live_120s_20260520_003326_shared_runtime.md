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
- render FPS after warmup: `6.90`
- raw fusion FPS after warmup: `15.26`
- filter output FPS after warmup: `15.26`
- fusion FPS after warmup: `15.26`
- stage period p50 after warmup: `65.16 ms`
- display packet period p50 after warmup: `143.89 ms`
- groups after warmup: `3353`
- complete fused groups after warmup: `1623`
- rendered groups after warmup: `719`
- complete group ratio after warmup: `0.484`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `23.10`
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
| camera startup ms | `10838.09` |
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
| time to first complete group s | `27.40` |
| time to first rendered group s | `29.56` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `494`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `87.00` | `92.00` | `93.00` | `95.00` |
| `memory_util_pct` | `47.00` | `82.00` | `85.00` | `88.00` |
| `memory_used_mb` | `6666.69` | `7831.09` | `7862.40` | `8914.69` |
| `power_w` | `302.95` | `328.94` | `332.41` | `337.58` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `66.00` | `75.00` | `77.00` | `79.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.51` | `36.75` | `38.43` | `189.50` |
| `display_packet_publish_period_ms` | `143.89` | `153.93` | `156.51` | `187.24` |
| `edgetam_stage_publish_period_ms` | `69.03` | `83.94` | `97.04` | `231.76` |
| `ffs_stage_publish_period_ms` | `65.91` | `72.59` | `75.04` | `458.08` |
| `filter_output_publish_period_ms` | `65.22` | `84.01` | `88.23` | `272.02` |
| `fusion_publish_period_ms` | `65.23` | `84.01` | `88.24` | `272.02` |
| `gpu_owner_publish_period_ms` | `65.16` | `83.47` | `87.32` | `270.40` |
| `raw_fusion_publish_period_ms` | `65.09` | `83.25` | `87.27` | `270.40` |
| `render_period_ms` | `144.65` | `155.12` | `159.26` | `193.05` |
| `stage_join_publish_period_ms` | `65.16` | `83.47` | `87.32` | `270.40` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `14.38` | `22.45` | `22.74` | `47.65` |
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
| `gpu_owner_total_ms` | `64.73` | `69.77` | `71.27` | `13217.97` |
| `gpu_owner_ffs_cycle_ms` | `62.26` | `64.98` | `65.63` | `71.07` |
| `gpu_owner_edgetam_cycle_ms` | `64.18` | `69.70` | `71.27` | `13217.97` |
| `raw_fusion_total_ms` | `7.99` | `11.26` | `13.64` | `24.93` |
| `fusion_total_ms` | `17.95` | `22.59` | `26.04` | `38.19` |
| `filter_total_ms` | `9.75` | `11.95` | `12.80` | `25.85` |
| `filter_input_age_ms` | `10.37` | `12.59` | `13.62` | `26.15` |
| `object_enhanced_pt_ms` | `4.15` | `5.54` | `6.33` | `19.05` |
| `controller_pt_filter_ms` | `5.47` | `6.59` | `7.34` | `19.04` |
| `render_total_ms` | `6.50` | `12.64` | `13.78` | `26.89` |
| `render_queue_wait_ms` | `203.65` | `249.74` | `257.32` | `2040.48` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.22` | `0.41` | `0.51` | `5.43` |
| `render_cpu_format_ms` | `0.66` | `1.46` | `2.31` | `5.86` |
| `render_open3d_points_update_ms` | `0.16` | `0.29` | `0.34` | `2.70` |
| `render_open3d_colors_update_ms` | `0.21` | `0.59` | `0.95` | `5.40` |
| `render_open3d_update_geometry_ms` | `5.70` | `11.32` | `12.46` | `16.83` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.13` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `790` | `19.05` | `24000` | `6365` |
| `514` | `17.40` | `24000` | `6308` |
| `1613` | `16.94` | `24000` | `6310` |
| `766` | `16.18` | `24000` | `6351` |
| `961` | `16.07` | `24000` | `6350` |
| `632` | `14.54` | `24000` | `6389` |
| `1528` | `14.15` | `24000` | `6396` |
| `1046` | `14.06` | `24000` | `6366` |
| `1571` | `13.87` | `24000` | `6402` |
| `1149` | `13.82` | `24000` | `6356` |
