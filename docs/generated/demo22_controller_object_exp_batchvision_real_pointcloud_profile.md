# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `5.69`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.69`
- filter output FPS after warmup: `5.69`
- fusion FPS after warmup: `5.69`
- stage period p50 after warmup: `168.18 ms`
- display packet period p50 after warmup: `168.32 ms`
- groups after warmup: `1351`
- complete fused groups after warmup: `572`
- rendered groups after warmup: `572`
- complete group ratio after warmup: `0.423`
- stage drop count after warmup: `46`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `9.31`
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
| parallel init max wait ms | `n/a` |
| camera startup ms | `4509.19` |
| EdgeTAM model load ms | `778.93` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1544.33` |
| EdgeTAM warmup/first forward ms | `67.91` |
| SAM3.1 model load ms | `7886.30` |
| SAM3.1 cam0 segment ms | `8154.09` |
| SAM3.1 cam1 segment ms | `185.94` |
| SAM3.1 cam2 segment ms | `178.30` |
| FFS runner init ms | `2504.39` |
| FFS first run ms | `1067.14` |
| session init + prompt add ms | `5.88` |
| SAM3.1 release cleanup ms | `242.63` |
| time to first complete group s | `27.19` |
| time to first rendered group s | `27.21` |

## GPU Sampling

GPU sampling disabled for this run.

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `67.13` | `89.29` | `134.06` | `406.79` |
| `display_packet_publish_period_ms` | `168.32` | `184.89` | `196.62` | `373.87` |
| `filter_output_publish_period_ms` | `168.32` | `184.87` | `196.62` | `373.88` |
| `fusion_publish_period_ms` | `168.32` | `184.87` | `196.62` | `373.88` |
| `gpu_owner_publish_period_ms` | `168.18` | `185.43` | `193.75` | `402.85` |
| `raw_fusion_publish_period_ms` | `168.37` | `185.02` | `193.78` | `403.10` |
| `render_period_ms` | `168.24` | `187.29` | `196.84` | `376.37` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `19.72` | `50.01` | `50.71` | `65.08` |
| `edgetam_model_ms` | `24.29` | `28.66` | `30.52` | `50.85` |
| `edgetam_preprocess_ms` | `1.07` | `1.28` | `1.35` | `1.85` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.12` | `0.16` | `0.57` |
| `edgetam_mask_resize_ms` | `0.04` | `0.07` | `0.10` | `0.53` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.05` | `0.06` | `0.38` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `0.29` | `0.45` | `6.85` |
| `edgetam_total_ms` | `24.77` | `29.17` | `31.03` | `51.30` |
| `ffs_cycle_ms` | `72.73` | `77.92` | `79.79` | `272.34` |
| `ffs_batch_ms` | `50.30` | `53.43` | `54.87` | `247.55` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `12.53` | `15.63` | `16.88` | `24.24` |
| `edgetam_batch_vision_total_ms` | `20.38` | `24.26` | `25.81` | `31.77` |
| `edgetam_batch_vision_preprocess_ms` | `3.21` | `3.85` | `4.05` | `5.55` |
| `edgetam_cam0_model_ms` | `24.74` | `28.71` | `30.03` | `50.85` |
| `edgetam_cam1_model_ms` | `24.18` | `29.07` | `31.06` | `38.46` |
| `edgetam_cam2_model_ms` | `23.75` | `28.32` | `29.86` | `41.87` |
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
| `ffs_stage_ms` | `1.88` | `2.93` | `3.27` | `8.18` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.88` | `2.93` | `3.26` | `8.18` |
| `ffs_cam1_stage_ms` | `1.88` | `2.93` | `3.26` | `8.18` |
| `ffs_cam2_stage_ms` | `1.88` | `2.93` | `3.26` | `8.18` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `168.10` | `184.85` | `193.02` | `402.80` |
| `gpu_owner_ffs_cycle_ms` | `72.73` | `77.92` | `79.79` | `272.34` |
| `gpu_owner_edgetam_cycle_ms` | `94.81` | `106.35` | `111.95` | `143.01` |
| `raw_fusion_total_ms` | `10.56` | `12.48` | `12.98` | `16.25` |
| `fusion_total_ms` | `50.93` | `56.43` | `58.40` | `251.68` |
| `filter_total_ms` | `40.08` | `45.07` | `47.32` | `240.23` |
| `filter_input_age_ms` | `40.64` | `45.58` | `47.86` | `241.13` |
| `object_enhanced_pt_ms` | `32.88` | `37.64` | `40.17` | `233.64` |
| `controller_pt_filter_ms` | `7.08` | `8.05` | `8.41` | `10.56` |
| `render_total_ms` | `2.04` | `2.67` | `2.89` | `4.51` |
| `render_queue_wait_ms` | `4.35` | `10.00` | `10.08` | `10.42` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.10` | `0.19` | `0.23` | `0.70` |
| `render_cpu_format_ms` | `0.27` | `0.44` | `0.53` | `0.94` |
| `render_open3d_points_update_ms` | `0.09` | `0.13` | `0.15` | `0.59` |
| `render_open3d_colors_update_ms` | `0.06` | `0.14` | `0.17` | `0.57` |
| `render_open3d_update_geometry_ms` | `1.44` | `2.07` | `2.31` | `3.76` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.03` | `0.03` | `0.04` | `0.09` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `758` | `233.64` | `49550` | `13994` |
| `1262` | `229.29` | `49316` | `13865` |
| `1341` | `228.39` | `49313` | `13812` |
| `1492` | `227.94` | `49364` | `13925` |
| `1568` | `227.71` | `49304` | `13772` |
| `1117` | `227.01` | `49497` | `13965` |
| `1189` | `224.34` | `49282` | `13767` |
| `480` | `221.73` | `49502` | `13961` |
| `902` | `221.28` | `49562` | `13943` |
| `281` | `220.28` | `49514` | `13932` |
