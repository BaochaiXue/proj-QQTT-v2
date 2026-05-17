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
- render FPS after warmup: `5.37`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.37`
- filter output FPS after warmup: `5.37`
- fusion FPS after warmup: `5.37`
- stage period p50 after warmup: `179.79 ms`
- display packet period p50 after warmup: `179.66 ms`
- groups after warmup: `971`
- complete fused groups after warmup: `377`
- rendered groups after warmup: `377`
- complete group ratio after warmup: `0.388`
- stage drop count after warmup: `19`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `9.63`
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
| parallel init max wait ms | `3401.58` |
| camera startup ms | `8432.09` |
| EdgeTAM model load ms | `2692.81` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1348.44` |
| EdgeTAM warmup/first forward ms | `79.65` |
| SAM3.1 model load ms | `10393.33` |
| SAM3.1 cam0 segment ms | `353.82` |
| SAM3.1 cam1 segment ms | `191.11` |
| SAM3.1 cam2 segment ms | `185.26` |
| FFS runner init ms | `7239.69` |
| FFS first run ms | `1283.58` |
| session init + prompt add ms | `5.46` |
| SAM3.1 release cleanup ms | `244.45` |
| time to first complete group s | `19.66` |
| time to first rendered group s | `19.67` |

## GPU Sampling

GPU sampling disabled for this run.

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.84` | `86.10` | `93.76` | `325.48` |
| `display_packet_publish_period_ms` | `179.66` | `193.68` | `199.73` | `398.15` |
| `edgetam_stage_publish_period_ms` | `179.57` | `190.00` | `195.36` | `400.02` |
| `ffs_stage_publish_period_ms` | `179.47` | `192.35` | `197.99` | `406.90` |
| `filter_output_publish_period_ms` | `179.66` | `193.68` | `199.71` | `398.12` |
| `fusion_publish_period_ms` | `179.66` | `193.68` | `199.71` | `398.13` |
| `gpu_owner_publish_period_ms` | `179.79` | `192.69` | `198.45` | `404.37` |
| `raw_fusion_publish_period_ms` | `179.78` | `192.69` | `198.45` | `404.37` |
| `render_period_ms` | `179.94` | `195.85` | `204.73` | `406.70` |
| `stage_join_publish_period_ms` | `179.79` | `192.69` | `198.45` | `404.37` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `17.75` | `51.58` | `54.18` | `65.08` |
| `edgetam_model_ms` | `44.21` | `71.76` | `75.36` | `280.40` |
| `edgetam_preprocess_ms` | `1.30` | `1.62` | `1.75` | `2.85` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.10` | `0.21` | `0.28` | `4.51` |
| `edgetam_mask_resize_ms` | `0.05` | `0.13` | `0.17` | `2.56` |
| `edgetam_mask_threshold_ms` | `0.04` | `0.08` | `0.10` | `4.18` |
| `edgetam_mask_to_cpu_ms` | `0.24` | `8.12` | `9.64` | `12.90` |
| `edgetam_total_ms` | `50.95` | `72.26` | `75.91` | `280.90` |
| `ffs_cycle_ms` | `77.24` | `84.55` | `87.59` | `99.99` |
| `ffs_batch_ms` | `56.14` | `59.00` | `60.48` | `63.88` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `19.80` | `23.88` | `24.84` | `32.10` |
| `edgetam_batch_vision_total_ms` | `29.59` | `33.58` | `34.67` | `41.23` |
| `edgetam_batch_vision_preprocess_ms` | `3.90` | `4.87` | `5.24` | `8.55` |
| `edgetam_cam0_model_ms` | `44.17` | `53.09` | `54.77` | `93.12` |
| `edgetam_cam1_model_ms` | `68.40` | `77.05` | `79.77` | `280.40` |
| `edgetam_cam2_model_ms` | `28.25` | `35.34` | `38.41` | `68.84` |
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
| `ffs_stage_ms` | `2.31` | `3.35` | `3.70` | `5.71` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.31` | `3.32` | `3.70` | `5.71` |
| `ffs_cam1_stage_ms` | `2.31` | `3.32` | `3.70` | `5.71` |
| `ffs_cam2_stage_ms` | `2.31` | `3.32` | `3.70` | `5.71` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `179.61` | `190.06` | `195.85` | `400.00` |
| `gpu_owner_ffs_cycle_ms` | `77.24` | `84.52` | `87.40` | `99.99` |
| `gpu_owner_edgetam_cycle_ms` | `179.61` | `190.06` | `195.85` | `400.00` |
| `raw_fusion_total_ms` | `9.73` | `12.51` | `13.72` | `17.71` |
| `fusion_total_ms` | `51.55` | `58.01` | `61.92` | `267.37` |
| `filter_total_ms` | `41.39` | `47.45` | `50.68` | `253.02` |
| `filter_input_age_ms` | `41.84` | `48.02` | `50.87` | `253.45` |
| `object_enhanced_pt_ms` | `33.87` | `39.91` | `42.20` | `246.71` |
| `controller_pt_filter_ms` | `7.26` | `8.76` | `9.11` | `12.68` |
| `render_total_ms` | `1.81` | `2.46` | `2.70` | `4.59` |
| `render_queue_wait_ms` | `9.42` | `9.96` | `10.02` | `10.51` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.14` | `0.24` | `0.28` | `0.93` |
| `render_cpu_format_ms` | `0.33` | `0.51` | `0.57` | `1.33` |
| `render_open3d_points_update_ms` | `0.09` | `0.15` | `0.19` | `0.41` |
| `render_open3d_colors_update_ms` | `0.07` | `0.17` | `0.20` | `0.58` |
| `render_open3d_update_geometry_ms` | `1.40` | `2.01` | `2.25` | `3.32` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.24` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `866` | `246.71` | `50190` | `14092` |
| `939` | `242.58` | `50217` | `14052` |
| `1092` | `239.04` | `50288` | `14060` |
| `270` | `235.76` | `50314` | `14182` |
| `1018` | `233.30` | `50220` | `14072` |
| `343` | `233.14` | `50313` | `14084` |
| `1170` | `231.29` | `50205` | `14039` |
| `635` | `230.46` | `50289` | `14069` |
| `560` | `227.75` | `50015` | `14029` |
| `710` | `223.06` | `50224` | `14141` |
