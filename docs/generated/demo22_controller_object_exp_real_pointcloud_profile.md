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
- render FPS after warmup: `4.71`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `4.71`
- filter output FPS after warmup: `4.71`
- fusion FPS after warmup: `4.71`
- groups after warmup: `1412`
- complete fused groups after warmup: `464`
- rendered groups after warmup: `464`
- complete group ratio after warmup: `0.329`
- target deficit: `10.29`
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
| camera startup ms | `4701.20` |
| EdgeTAM model load ms | `817.11` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1433.28` |
| EdgeTAM warmup/first forward ms | `131.16` |
| SAM3.1 model load ms | `8427.83` |
| SAM3.1 cam0 segment ms | `8710.82` |
| SAM3.1 cam1 segment ms | `193.45` |
| SAM3.1 cam2 segment ms | `185.98` |
| FFS runner init ms | `2628.32` |
| FFS first run ms | `1107.59` |
| session init + prompt add ms | `4.10` |
| SAM3.1 release cleanup ms | `254.97` |
| time to first complete group s | `25.99` |
| time to first rendered group s | `26.00` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `197`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `48.00` | `49.20` | `53.00` |
| `memory_util_pct` | `11.00` | `13.00` | `15.20` | `20.00` |
| `memory_used_mb` | `8594.47` | `12938.58` | `13460.58` | `13736.18` |
| `power_w` | `123.78` | `141.13` | `155.11` | `212.23` |
| `sm_clock_mhz` | `180.00` | `232.00` | `1110.00` | `1110.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `66.00` | `69.00` | `70.00` | `72.00` |

- sampler errors: `nvml: NVMLError_Unknown: Unknown Error; nvml: NVMLError_Unknown: Unknown Error; nvml: NVMLError_Unknown: Unknown Error`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `36.72` | `48.76` | `49.65` | `66.66` |
| `edgetam_model_ms` | `39.33` | `47.05` | `50.34` | `73.49` |
| `edgetam_preprocess_ms` | `1.45` | `1.89` | `2.07` | `4.55` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.08` | `0.14` | `0.18` | `0.57` |
| `edgetam_mask_resize_ms` | `0.05` | `0.09` | `0.11` | `0.35` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.06` | `0.07` | `0.50` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `0.26` | `0.29` | `0.88` |
| `edgetam_total_ms` | `42.90` | `50.68` | `54.08` | `77.29` |
| `ffs_cycle_ms` | `75.81` | `79.57` | `81.42` | `283.76` |
| `ffs_batch_ms` | `54.28` | `57.07` | `58.57` | `254.18` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `39.97` | `47.71` | `50.65` | `68.76` |
| `edgetam_cam1_model_ms` | `37.77` | `46.03` | `48.85` | `70.95` |
| `edgetam_cam2_model_ms` | `40.19` | `47.70` | `50.59` | `73.49` |
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
| `ffs_stage_ms` | `2.13` | `3.08` | `3.52` | `7.83` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.13` | `3.08` | `3.52` | `7.83` |
| `ffs_cam1_stage_ms` | `2.13` | `3.08` | `3.52` | `7.83` |
| `ffs_cam2_stage_ms` | `2.13` | `3.08` | `3.52` | `7.83` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `205.41` | `228.77` | `237.72` | `426.49` |
| `gpu_owner_ffs_cycle_ms` | `75.81` | `79.57` | `81.42` | `283.76` |
| `gpu_owner_edgetam_cycle_ms` | `129.67` | `147.56` | `153.53` | `193.43` |
| `raw_fusion_total_ms` | `11.59` | `13.39` | `13.79` | `16.32` |
| `fusion_total_ms` | `52.01` | `56.32` | `58.65` | `267.60` |
| `filter_total_ms` | `40.35` | `43.77` | `45.42` | `256.40` |
| `filter_input_age_ms` | `40.95` | `44.42` | `46.03` | `257.13` |
| `object_enhanced_pt_ms` | `25.14` | `28.40` | `30.29` | `241.57` |
| `controller_pt_filter_ms` | `14.92` | `16.28` | `16.93` | `19.48` |
| `render_total_ms` | `2.24` | `2.89` | `3.37` | `7.60` |
| `render_queue_wait_ms` | `9.05` | `9.68` | `9.82` | `10.72` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.11` | `0.19` | `0.24` | `0.62` |
| `render_cpu_format_ms` | `0.30` | `0.46` | `0.55` | `1.06` |
| `render_open3d_points_update_ms` | `0.10` | `0.14` | `0.16` | `0.72` |
| `render_open3d_colors_update_ms` | `0.07` | `0.14` | `0.17` | `0.68` |
| `render_open3d_update_geometry_ms` | `1.59` | `2.21` | `2.64` | `6.20` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.03` | `0.04` | `0.04` | `0.09` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1606` | `241.57` | `43342` | `11753` |
| `1037` | `236.35` | `43321` | `11728` |
| `1165` | `235.29` | `43277` | `11741` |
| `1505` | `232.89` | `43313` | `11797` |
| `577` | `230.44` | `41143` | `11320` |
| `816` | `229.12` | `43287` | `11733` |
| `1285` | `225.33` | `43331` | `11750` |
| `933` | `220.66` | `43263` | `11712` |
| `1398` | `217.45` | `43314` | `11715` |
| `693` | `217.27` | `41683` | `11409` |
