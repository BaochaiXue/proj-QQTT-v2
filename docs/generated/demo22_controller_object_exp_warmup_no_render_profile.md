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
- raw fusion FPS after warmup: `5.16`
- filter output FPS after warmup: `5.16`
- fusion FPS after warmup: `5.16`
- groups after warmup: `598`
- complete fused groups after warmup: `206`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.344`
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
| camera startup ms | `4614.11` |
| EdgeTAM model load ms | `933.78` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1262.52` |
| EdgeTAM warmup/first forward ms | `123.97` |
| SAM3.1 model load ms | `9384.72` |
| SAM3.1 cam0 segment ms | `9682.17` |
| SAM3.1 cam1 segment ms | `184.14` |
| SAM3.1 cam2 segment ms | `177.57` |
| FFS runner init ms | `4603.07` |
| FFS first run ms | `1113.79` |
| session init + prompt add ms | `23.82` |
| SAM3.1 release cleanup ms | `269.84` |
| time to first complete group s | `24.51` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `92`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `46.00` | `50.90` | `52.00` | `84.00` |
| `memory_util_pct` | `11.00` | `14.90` | `17.00` | `21.00` |
| `memory_used_mb` | `5151.10` | `7040.70` | `7314.00` | `7315.10` |
| `power_w` | `119.29` | `151.21` | `167.54` | `236.43` |
| `sm_clock_mhz` | `232.00` | `232.00` | `627.10` | `1110.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `58.50` | `64.00` | `65.45` | `67.00` |

- sampler errors: `nvml: NVMLError_Unknown: Unknown Error; nvml: NVMLError_Unknown: Unknown Error`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `20.20` | `57.41` | `60.83` | `69.85` |
| `edgetam_model_ms` | `35.15` | `41.18` | `43.77` | `59.31` |
| `edgetam_preprocess_ms` | `1.36` | `1.73` | `1.87` | `2.87` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.12` | `0.16` | `0.53` |
| `edgetam_mask_resize_ms` | `0.04` | `0.07` | `0.09` | `0.48` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.05` | `0.07` | `0.28` |
| `edgetam_mask_to_cpu_ms` | `0.20` | `0.25` | `0.27` | `0.81` |
| `edgetam_total_ms` | `38.58` | `44.60` | `47.11` | `63.64` |
| `ffs_cycle_ms` | `70.73` | `75.88` | `78.33` | `249.20` |
| `ffs_batch_ms` | `52.40` | `56.39` | `57.81` | `224.40` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `35.35` | `41.27` | `43.75` | `59.31` |
| `edgetam_cam1_model_ms` | `35.18` | `41.17` | `43.89` | `54.42` |
| `edgetam_cam2_model_ms` | `34.99` | `40.75` | `42.58` | `54.82` |
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
| `ffs_stage_ms` | `2.64` | `4.27` | `4.83` | `30.44` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.64` | `4.27` | `4.74` | `30.44` |
| `ffs_cam1_stage_ms` | `2.64` | `4.27` | `4.74` | `30.44` |
| `ffs_cam2_stage_ms` | `2.64` | `4.27` | `4.74` | `30.44` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `186.15` | `205.75` | `220.46` | `369.45` |
| `gpu_owner_ffs_cycle_ms` | `70.73` | `75.88` | `78.33` | `249.20` |
| `gpu_owner_edgetam_cycle_ms` | `115.11` | `129.60` | `134.95` | `167.50` |
| `raw_fusion_total_ms` | `11.11` | `12.65` | `13.72` | `15.87` |
| `fusion_total_ms` | `49.87` | `54.71` | `57.18` | `235.91` |
| `filter_total_ms` | `38.44` | `43.05` | `44.49` | `225.94` |
| `filter_input_age_ms` | `38.91` | `43.72` | `45.24` | `226.04` |
| `object_enhanced_pt_ms` | `23.74` | `27.00` | `28.85` | `211.87` |
| `controller_pt_filter_ms` | `14.67` | `16.46` | `16.91` | `18.75` |
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
| `741` | `211.87` | `41167` | `11310` |
| `334` | `205.54` | `41118` | `11265` |
| `534` | `205.35` | `41111` | `11384` |
| `431` | `205.26` | `41150` | `11338` |
| `636` | `199.54` | `41122` | `11317` |
| `754` | `31.41` | `41141` | `11310` |
| `83` | `29.89` | `40380` | `10971` |
| `666` | `29.70` | `41167` | `11327` |
| `606` | `29.06` | `41192` | `11290` |
| `624` | `29.00` | `41155` | `11273` |
