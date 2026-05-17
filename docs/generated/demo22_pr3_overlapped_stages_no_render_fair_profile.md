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
- render FPS after warmup: `0.00`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.25`
- filter output FPS after warmup: `5.25`
- fusion FPS after warmup: `5.25`
- stage period p50 after warmup: `181.38 ms`
- display packet period p50 after warmup: `182.05 ms`
- groups after warmup: `936`
- complete fused groups after warmup: `365`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.390`
- stage drop count after warmup: `24`
- raw fused pending replacements total: `0`
- render buffer dropped total: `424`
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
| parallel init max wait ms | `4361.27` |
| camera startup ms | `8415.33` |
| EdgeTAM model load ms | `965.25` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1455.99` |
| EdgeTAM warmup/first forward ms | `82.61` |
| SAM3.1 model load ms | `9923.83` |
| SAM3.1 cam0 segment ms | `368.83` |
| SAM3.1 cam1 segment ms | `191.11` |
| SAM3.1 cam2 segment ms | `194.33` |
| FFS runner init ms | `7615.40` |
| FFS first run ms | `1230.93` |
| session init + prompt add ms | `5.60` |
| SAM3.1 release cleanup ms | `241.60` |
| time to first complete group s | `19.54` |
| time to first rendered group s | `n/a` |

## GPU Sampling

GPU sampling disabled for this run.

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.85` | `89.34` | `131.02` | `308.56` |
| `display_packet_publish_period_ms` | `182.05` | `203.92` | `226.02` | `415.34` |
| `edgetam_stage_publish_period_ms` | `181.32` | `204.17` | `218.88` | `413.31` |
| `ffs_stage_publish_period_ms` | `181.10` | `203.51` | `224.67` | `422.20` |
| `filter_output_publish_period_ms` | `182.05` | `203.92` | `225.96` | `415.35` |
| `fusion_publish_period_ms` | `182.05` | `203.92` | `226.01` | `415.35` |
| `gpu_owner_publish_period_ms` | `181.38` | `201.80` | `226.69` | `420.90` |
| `raw_fusion_publish_period_ms` | `181.38` | `201.80` | `226.62` | `420.90` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `181.38` | `201.80` | `226.69` | `420.90` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `19.77` | `51.78` | `54.34` | `66.56` |
| `edgetam_model_ms` | `44.99` | `73.15` | `76.32` | `286.82` |
| `edgetam_preprocess_ms` | `1.33` | `1.76` | `1.94` | `13.21` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.11` | `0.22` | `0.27` | `3.33` |
| `edgetam_mask_resize_ms` | `0.06` | `0.14` | `0.18` | `3.25` |
| `edgetam_mask_threshold_ms` | `0.04` | `0.08` | `0.10` | `0.50` |
| `edgetam_mask_to_cpu_ms` | `0.25` | `7.06` | `8.19` | `10.92` |
| `edgetam_total_ms` | `50.96` | `73.71` | `77.04` | `287.30` |
| `ffs_cycle_ms` | `79.01` | `85.50` | `88.33` | `100.16` |
| `ffs_batch_ms` | `57.16` | `60.29` | `61.78` | `67.84` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `20.53` | `25.06` | `26.59` | `33.79` |
| `edgetam_batch_vision_total_ms` | `30.72` | `34.91` | `37.09` | `66.16` |
| `edgetam_batch_vision_preprocess_ms` | `3.98` | `5.29` | `5.80` | `39.62` |
| `edgetam_cam0_model_ms` | `45.07` | `54.06` | `57.64` | `80.46` |
| `edgetam_cam1_model_ms` | `70.44` | `78.30` | `87.77` | `286.82` |
| `edgetam_cam2_model_ms` | `28.22` | `34.79` | `37.59` | `57.51` |
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
| `ffs_stage_ms` | `2.71` | `4.20` | `4.87` | `9.91` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.71` | `4.19` | `4.86` | `9.91` |
| `ffs_cam1_stage_ms` | `2.71` | `4.19` | `4.86` | `9.91` |
| `ffs_cam2_stage_ms` | `2.71` | `4.19` | `4.86` | `9.91` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `181.29` | `199.61` | `218.60` | `413.29` |
| `gpu_owner_ffs_cycle_ms` | `79.02` | `85.51` | `88.34` | `100.16` |
| `gpu_owner_edgetam_cycle_ms` | `181.29` | `199.61` | `218.60` | `413.29` |
| `raw_fusion_total_ms` | `10.37` | `12.72` | `13.74` | `18.09` |
| `fusion_total_ms` | `55.17` | `62.42` | `66.00` | `268.81` |
| `filter_total_ms` | `44.38` | `50.82` | `55.24` | `259.00` |
| `filter_input_age_ms` | `44.97` | `51.59` | `55.83` | `259.87` |
| `object_enhanced_pt_ms` | `36.60` | `42.38` | `46.20` | `251.00` |
| `controller_pt_filter_ms` | `7.70` | `9.04` | `9.50` | `12.25` |
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
| `702` | `251.00` | `50736` | `14210` |
| `856` | `250.53` | `50702` | `14216` |
| `269` | `246.78` | `50706` | `14195` |
| `1083` | `244.44` | `50730` | `14210` |
| `1003` | `239.18` | `50752` | `14205` |
| `929` | `238.54` | `50708` | `14231` |
| `630` | `237.29` | `50677` | `14190` |
| `779` | `236.76` | `50693` | `14291` |
| `1164` | `235.33` | `50718` | `14257` |
| `560` | `234.38` | `50753` | `14128` |
