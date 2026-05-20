# Demo 3.1 performance profile

- preset: `demo2.1.5-live-fast-native`
- canonical preset: `demo2.1.5-live-fast-native`
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
- render FPS after warmup: `0.11`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.49`
- stage period p50 after warmup: `64.43 ms`
- display packet period p50 after warmup: `9619.38 ms`
- groups after warmup: `3399`
- complete fused groups after warmup: `1764`
- rendered groups after warmup: `12`
- complete group ratio after warmup: `0.519`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.89`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `none`
- render filtered only: `False`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `0.01` |
| camera startup ms | `11012.42` |
| EdgeTAM model load ms | `768.75` |
| EdgeTAM compile wrap ms | `425.50` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `126.37` |
| SAM3.1 model load ms | `7525.82` |
| SAM3.1 cam0 segment ms | `597.54` |
| SAM3.1 cam1 segment ms | `120.66` |
| SAM3.1 cam2 segment ms | `118.94` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.69` |
| SAM3.1 release cleanup ms | `303.82` |
| time to first complete group s | `20.31` |
| time to first rendered group s | `30.89` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `474`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `45.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `73.70` | `76.00` | `79.00` |
| `memory_used_mb` | `3539.38` | `24063.36` | `24147.33` | `24448.75` |
| `power_w` | `157.35` | `367.19` | `372.02` | `395.23` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `52.00` | `76.00` | `79.00` | `82.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.50` | `37.01` | `38.71` | `44.08` |
| `display_packet_publish_period_ms` | `9619.38` | `14366.01` | `16734.52` | `19103.03` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.77` | `72.21` | `73.81` | `85.29` |
| `gpu_owner_publish_period_ms` | `64.43` | `69.90` | `72.02` | `81.18` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `9619.07` | `14364.09` | `16734.94` | `19105.79` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.34` | `24.43` | `26.79` | `41.20` |
| `edgetam_model_ms` | `15.53` | `26.84` | `28.48` | `36.96` |
| `edgetam_preprocess_ms` | `0.57` | `0.74` | `0.84` | `2.40` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `6.31` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.52` | `0.59` | `7.20` |
| `edgetam_total_ms` | `16.13` | `27.56` | `29.16` | `37.61` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.22` | `1.47` | `1.63` | `35.32` |
| `edgetam_batch_vision_total_ms` | `7.19` | `9.72` | `10.46` | `44.65` |
| `edgetam_batch_vision_preprocess_ms` | `1.72` | `2.23` | `2.52` | `7.21` |
| `edgetam_cam0_model_ms` | `24.27` | `29.27` | `30.43` | `36.96` |
| `edgetam_cam1_model_ms` | `14.82` | `17.99` | `20.26` | `25.32` |
| `edgetam_cam2_model_ms` | `13.97` | `16.02` | `16.54` | `20.00` |
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
| `gpu_owner_total_ms` | `64.39` | `69.87` | `72.02` | `113.84` |
| `gpu_owner_ffs_cycle_ms` | `0.33` | `0.55` | `0.85` | `2.09` |
| `gpu_owner_edgetam_cycle_ms` | `63.98` | `69.44` | `71.60` | `112.23` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.62` | `20.77` | `21.57` | `27.08` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.68` | `4.19` | `4.47` | `5.75` |
| `controller_pt_filter_ms` | `6.16` | `6.90` | `7.20` | `9.47` |
| `render_total_ms` | `1.91` | `2.20` | `2.58` | `3.04` |
| `render_queue_wait_ms` | `3976.44` | `4034.26` | `4114.14` | `4205.62` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.13` | `0.15` | `0.17` |
| `render_cpu_format_ms` | `0.24` | `0.34` | `0.35` | `0.37` |
| `render_open3d_points_update_ms` | `0.08` | `0.12` | `0.13` | `0.14` |
| `render_open3d_colors_update_ms` | `0.08` | `0.08` | `0.09` | `0.11` |
| `render_open3d_update_geometry_ms` | `1.43` | `1.63` | `1.65` | `1.68` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.03` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2160` | `5.75` | `24000` | `5750` |
| `2896` | `5.46` | `24000` | `5764` |
| `2189` | `5.39` | `24000` | `5764` |
| `661` | `5.38` | `24000` | `5733` |
| `2325` | `5.38` | `24000` | `5728` |
| `2973` | `5.37` | `24000` | `5787` |
| `1224` | `5.37` | `24000` | `5752` |
| `681` | `5.34` | `24000` | `5784` |
| `2546` | `5.26` | `24000` | `5770` |
| `1562` | `5.20` | `24000` | `5750` |
