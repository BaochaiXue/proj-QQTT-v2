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
- raw fusion FPS after warmup: `2.83`
- filter output FPS after warmup: `1.86`
- fusion FPS after warmup: `1.86`
- stage period p50 after warmup: `282.10 ms`
- display packet period p50 after warmup: `448.48 ms`
- groups after warmup: `273`
- complete fused groups after warmup: `15`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.055`
- stage drop count after warmup: `19`
- raw fused pending replacements total: `0`
- render buffer dropped total: `14`
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
| parallel init max wait ms | `5822.29` |
| camera startup ms | `8242.47` |
| EdgeTAM model load ms | `984.76` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1896.49` |
| EdgeTAM warmup/first forward ms | `221.57` |
| SAM3.1 model load ms | `10794.19` |
| SAM3.1 cam0 segment ms | `767.04` |
| SAM3.1 cam1 segment ms | `411.08` |
| SAM3.1 cam2 segment ms | `404.20` |
| FFS runner init ms | `8425.56` |
| FFS first run ms | `1421.67` |
| session init + prompt add ms | `17.75` |
| SAM3.1 release cleanup ms | `278.85` |
| time to first complete group s | `20.57` |
| time to first rendered group s | `n/a` |

## GPU Sampling

GPU sampling disabled for this run.

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.90` | `74.04` | `133.03` | `340.75` |
| `display_packet_publish_period_ms` | `448.48` | `896.25` | `979.81` | `1017.92` |
| `edgetam_stage_publish_period_ms` | `249.66` | `277.78` | `283.19` | `356.71` |
| `ffs_stage_publish_period_ms` | `74.45` | `103.42` | `113.60` | `572.46` |
| `filter_output_publish_period_ms` | `448.51` | `896.24` | `979.81` | `1017.92` |
| `fusion_publish_period_ms` | `448.51` | `896.25` | `979.81` | `1017.91` |
| `gpu_owner_publish_period_ms` | `282.10` | `492.74` | `501.36` | `505.44` |
| `raw_fusion_publish_period_ms` | `282.10` | `492.77` | `501.36` | `505.43` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `282.10` | `492.74` | `501.36` | `505.44` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `23.19` | `41.98` | `49.84` | `66.44` |
| `edgetam_model_ms` | `44.34` | `51.59` | `54.71` | `95.76` |
| `edgetam_preprocess_ms` | `1.47` | `2.35` | `9.91` | `13.98` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `13.92` |
| `edgetam_postprocess_ms` | `0.09` | `0.20` | `0.23` | `0.45` |
| `edgetam_mask_resize_ms` | `0.05` | `0.13` | `0.18` | `0.41` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.06` | `0.07` | `0.27` |
| `edgetam_mask_to_cpu_ms` | `19.23` | `24.23` | `25.84` | `34.02` |
| `edgetam_total_ms` | `65.63` | `73.92` | `76.40` | `130.06` |
| `ffs_cycle_ms` | `73.50` | `102.30` | `111.06` | `1463.23` |
| `ffs_batch_ms` | `51.34` | `61.32` | `65.79` | `1421.67` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `49.58` | `65.98` | `69.77` | `78.78` |
| `edgetam_batch_vision_total_ms` | `61.15` | `76.08` | `80.35` | `91.69` |
| `edgetam_batch_vision_preprocess_ms` | `4.41` | `6.91` | `22.62` | `41.94` |
| `edgetam_cam0_model_ms` | `37.57` | `44.30` | `46.07` | `95.76` |
| `edgetam_cam1_model_ms` | `45.08` | `50.96` | `52.33` | `61.96` |
| `edgetam_cam2_model_ms` | `47.93` | `53.74` | `55.56` | `58.61` |
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
| `ffs_stage_ms` | `2.84` | `5.64` | `8.55` | `33.27` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.84` | `5.51` | `8.41` | `33.27` |
| `ffs_cam1_stage_ms` | `2.84` | `5.51` | `8.41` | `33.27` |
| `ffs_cam2_stage_ms` | `2.84` | `5.51` | `8.41` | `33.27` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `240.68` | `278.28` | `288.84` | `356.66` |
| `gpu_owner_ffs_cycle_ms` | `80.27` | `105.42` | `108.60` | `133.62` |
| `gpu_owner_edgetam_cycle_ms` | `240.68` | `278.28` | `288.84` | `356.66` |
| `raw_fusion_total_ms` | `12.43` | `15.12` | `15.64` | `17.60` |
| `fusion_total_ms` | `54.43` | `60.89` | `61.80` | `63.08` |
| `filter_total_ms` | `42.60` | `47.87` | `48.45` | `48.77` |
| `filter_input_age_ms` | `43.22` | `48.11` | `48.92` | `49.77` |
| `object_enhanced_pt_ms` | `33.70` | `37.25` | `37.47` | `37.67` |
| `controller_pt_filter_ms` | `10.90` | `11.28` | `11.54` | `12.06` |
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
| `234` | `37.67` | `50728` | `14271` |
| `264` | `37.39` | `50714` | `14208` |
| `158` | `37.04` | `30551` | `9421` |
| `219` | `36.64` | `50694` | `14116` |
| `181` | `35.36` | `50703` | `14256` |
| `250` | `34.89` | `50706` | `14088` |
| `191` | `34.53` | `50738` | `14237` |
| `164` | `33.70` | `50527` | `14175` |
| `204` | `31.44` | `50780` | `14189` |
| `265` | `30.66` | `30719` | `9649` |
