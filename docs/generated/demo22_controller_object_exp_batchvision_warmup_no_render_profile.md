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
- raw fusion FPS after warmup: `6.10`
- filter output FPS after warmup: `6.11`
- fusion FPS after warmup: `6.11`
- stage period p50 after warmup: `158.00 ms`
- display packet period p50 after warmup: `158.09 ms`
- groups after warmup: `599`
- complete fused groups after warmup: `204`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.341`
- stage drop count after warmup: `13`
- raw fused pending replacements total: `0`
- render buffer dropped total: `204`
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
| parallel init max wait ms | `n/a` |
| camera startup ms | `4466.37` |
| EdgeTAM model load ms | `1074.21` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1811.71` |
| EdgeTAM warmup/first forward ms | `130.36` |
| SAM3.1 model load ms | `10679.25` |
| SAM3.1 cam0 segment ms | `11222.23` |
| SAM3.1 cam1 segment ms | `256.42` |
| SAM3.1 cam2 segment ms | `268.11` |
| FFS runner init ms | `8065.89` |
| FFS first run ms | `1319.71` |
| session init + prompt add ms | `29.93` |
| SAM3.1 release cleanup ms | `246.98` |
| time to first complete group s | `31.06` |
| time to first rendered group s | `n/a` |

## GPU Sampling

GPU sampling disabled for this run.

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.99` | `83.86` | `92.03` | `940.62` |
| `display_packet_publish_period_ms` | `158.09` | `168.94` | `174.10` | `362.09` |
| `filter_output_publish_period_ms` | `158.09` | `168.94` | `174.10` | `362.08` |
| `fusion_publish_period_ms` | `158.09` | `168.94` | `174.10` | `362.08` |
| `gpu_owner_publish_period_ms` | `158.00` | `168.33` | `172.11` | `358.74` |
| `raw_fusion_publish_period_ms` | `158.33` | `168.54` | `173.55` | `360.51` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `33.68` | `49.14` | `54.31` | `65.84` |
| `edgetam_model_ms` | `22.74` | `26.20` | `27.38` | `37.10` |
| `edgetam_preprocess_ms` | `1.00` | `1.20` | `1.26` | `1.62` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.10` | `0.13` | `0.49` |
| `edgetam_mask_resize_ms` | `0.04` | `0.07` | `0.09` | `0.30` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.04` | `0.05` | `0.45` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `1.79` | `3.02` | `12.84` |
| `edgetam_total_ms` | `23.55` | `27.16` | `28.42` | `37.56` |
| `ffs_cycle_ms` | `67.85` | `71.78` | `74.67` | `261.97` |
| `ffs_batch_ms` | `49.36` | `52.39` | `54.63` | `238.40` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `11.42` | `14.02` | `15.04` | `18.82` |
| `edgetam_batch_vision_total_ms` | `18.95` | `22.36` | `23.01` | `27.30` |
| `edgetam_batch_vision_preprocess_ms` | `2.99` | `3.61` | `3.78` | `4.86` |
| `edgetam_cam0_model_ms` | `23.19` | `26.68` | `28.18` | `37.10` |
| `edgetam_cam1_model_ms` | `22.61` | `26.30` | `27.10` | `32.18` |
| `edgetam_cam2_model_ms` | `22.48` | `25.20` | `26.74` | `33.82` |
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
| `ffs_stage_ms` | `2.08` | `3.33` | `3.85` | `27.35` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.08` | `3.27` | `3.85` | `27.35` |
| `ffs_cam1_stage_ms` | `2.08` | `3.27` | `3.85` | `27.35` |
| `ffs_cam2_stage_ms` | `2.08` | `3.27` | `3.85` | `27.35` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `158.00` | `168.62` | `172.15` | `358.69` |
| `gpu_owner_ffs_cycle_ms` | `67.85` | `71.78` | `74.67` | `261.97` |
| `gpu_owner_edgetam_cycle_ms` | `89.38` | `97.56` | `100.04` | `118.48` |
| `raw_fusion_total_ms` | `10.15` | `11.52` | `11.95` | `14.86` |
| `fusion_total_ms` | `47.31` | `53.25` | `54.23` | `243.34` |
| `filter_total_ms` | `37.19` | `42.85` | `44.55` | `232.02` |
| `filter_input_age_ms` | `37.78` | `43.41` | `44.73` | `232.92` |
| `object_enhanced_pt_ms` | `30.35` | `35.83` | `37.48` | `224.60` |
| `controller_pt_filter_ms` | `6.78` | `7.70` | `7.90` | `11.01` |
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
| `628` | `224.60` | `49505` | `13967` |
| `357` | `223.73` | `49523` | `14009` |
| `692` | `222.48` | `49522` | `13935` |
| `487` | `222.20` | `49491` | `13991` |
| `422` | `216.24` | `49528` | `13975` |
| `556` | `215.33` | `49313` | `13907` |
| `116` | `43.51` | `48692` | `13497` |
| `598` | `39.42` | `49551` | `14036` |
| `680` | `38.31` | `49523` | `14067` |
| `511` | `38.10` | `49498` | `13975` |
