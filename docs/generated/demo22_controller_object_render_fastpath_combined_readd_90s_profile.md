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
- render copy mode: `async-pinned`
- render FPS after warmup: `5.63`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.63`
- filter output FPS after warmup: `5.63`
- fusion FPS after warmup: `5.63`
- groups after warmup: `895`
- complete fused groups after warmup: `396`
- rendered groups after warmup: `396`
- complete group ratio after warmup: `0.442`
- target deficit: `9.37`
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
| camera startup ms | `4500.52` |
| EdgeTAM model load ms | `794.56` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `103.26` |
| SAM3.1 model load ms | `7754.79` |
| SAM3.1 cam0 segment ms | `8199.72` |
| SAM3.1 cam1 segment ms | `176.97` |
| SAM3.1 cam2 segment ms | `177.05` |
| FFS runner init ms | `2446.78` |
| FFS first run ms | `1074.06` |
| session init + prompt add ms | `6.13` |
| SAM3.1 release cleanup ms | `242.24` |
| time to first complete group s | `26.10` |
| time to first rendered group s | `26.12` |

## GPU Sampling

GPU sampling disabled for this run.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `20.65` | `52.80` | `59.25` | `66.67` |
| `edgetam_model_ms` | `23.40` | `28.23` | `29.94` | `51.22` |
| `edgetam_preprocess_ms` | `1.03` | `1.32` | `1.44` | `2.31` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.11` | `0.13` | `2.30` |
| `edgetam_mask_resize_ms` | `0.04` | `0.07` | `0.08` | `0.32` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.04` | `0.05` | `2.11` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `0.35` | `0.99` | `16.59` |
| `edgetam_total_ms` | `24.02` | `28.90` | `30.78` | `51.59` |
| `ffs_cycle_ms` | `76.36` | `81.71` | `85.29` | `266.31` |
| `ffs_batch_ms` | `51.01` | `57.45` | `59.23` | `235.84` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `12.28` | `15.97` | `17.08` | `24.52` |
| `edgetam_batch_vision_total_ms` | `19.83` | `24.47` | `26.39` | `33.33` |
| `edgetam_batch_vision_preprocess_ms` | `3.09` | `3.96` | `4.30` | `6.92` |
| `edgetam_cam0_model_ms` | `24.23` | `28.63` | `30.39` | `40.33` |
| `edgetam_cam1_model_ms` | `22.88` | `27.78` | `29.01` | `51.22` |
| `edgetam_cam2_model_ms` | `22.91` | `28.10` | `29.63` | `37.84` |
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
| `ffs_stage_ms` | `2.34` | `4.00` | `4.75` | `12.56` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.34` | `3.99` | `4.74` | `12.56` |
| `ffs_cam1_stage_ms` | `2.34` | `3.99` | `4.74` | `12.56` |
| `ffs_cam2_stage_ms` | `2.34` | `3.99` | `4.74` | `12.56` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `168.97` | `183.92` | `195.19` | `369.78` |
| `gpu_owner_ffs_cycle_ms` | `76.36` | `81.71` | `85.29` | `266.31` |
| `gpu_owner_edgetam_cycle_ms` | `92.38` | `103.59` | `107.88` | `137.22` |
| `raw_fusion_total_ms` | `11.15` | `13.02` | `13.63` | `16.33` |
| `fusion_total_ms` | `54.90` | `61.41` | `64.18` | `247.95` |
| `filter_total_ms` | `43.53` | `49.59` | `52.29` | `237.39` |
| `filter_input_age_ms` | `44.13` | `50.28` | `52.90` | `238.01` |
| `object_enhanced_pt_ms` | `29.37` | `34.39` | `36.74` | `222.34` |
| `controller_pt_filter_ms` | `14.04` | `16.28` | `17.00` | `20.29` |
| `render_total_ms` | `2.63` | `4.17` | `4.69` | `7.39` |
| `render_queue_wait_ms` | `9.11` | `9.79` | `9.86` | `10.45` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.12` | `0.20` | `0.25` | `0.70` |
| `render_cpu_format_ms` | `0.30` | `0.46` | `0.63` | `1.02` |
| `render_open3d_points_update_ms` | `0.09` | `0.14` | `0.16` | `0.73` |
| `render_open3d_colors_update_ms` | `0.06` | `0.15` | `0.20` | `0.66` |
| `render_open3d_update_geometry_ms` | `2.21` | `3.75` | `4.29` | `6.65` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.03` | `0.03` | `0.04` | `0.08` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1128` | `222.34` | `45392` | `12517` |
| `970` | `220.57` | `49007` | `12993` |
| `1047` | `216.05` | `42514` | `11500` |
| `821` | `214.99` | `47693` | `12584` |
| `740` | `213.12` | `47723` | `12645` |
| `350` | `212.73` | `47706` | `12570` |
| `664` | `212.36` | `47693` | `12485` |
| `431` | `211.79` | `47738` | `12540` |
| `895` | `210.80` | `47694` | `12584` |
| `587` | `210.37` | `47732` | `12600` |
