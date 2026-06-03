# Demo 3.3 performance profile

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
- render FPS after warmup: `4.01`
- raw fusion FPS after warmup: `4.01`
- filter output FPS after warmup: `4.01`
- fusion FPS after warmup: `4.01`
- stage period p50 after warmup: `92.54 ms`
- display packet period p50 after warmup: `234.00 ms`
- groups after warmup: `1091`
- complete fused groups after warmup: `96`
- rendered groups after warmup: `95`
- complete group ratio after warmup: `0.088`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.99`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Shape Prior Warmup

- enabled: `True`
- status: `case_ready`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-172055/case`
- object points0: `77105`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10746.50` |
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
| time to first complete group s | `34.64` |
| time to first rendered group s | `34.81` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `194`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `64.00` | `92.00` | `95.35` | `97.00` |
| `memory_util_pct` | `38.00` | `57.00` | `60.35` | `62.00` |
| `memory_used_mb` | `6105.44` | `7759.87` | `7764.19` | `8353.44` |
| `power_w` | `270.00` | `297.68` | `303.78` | `342.98` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `61.00` | `66.00` | `68.00` | `70.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.37` | `39.85` | `54.89` | `221.48` |
| `display_packet_publish_period_ms` | `234.00` | `286.62` | `371.91` | `407.80` |
| `edgetam_stage_publish_period_ms` | `69.08` | `94.55` | `106.30` | `1584.24` |
| `ffs_stage_publish_period_ms` | `60.37` | `96.38` | `107.18` | `1636.84` |
| `filter_output_publish_period_ms` | `234.68` | `285.58` | `378.43` | `424.85` |
| `fusion_publish_period_ms` | `234.68` | `285.59` | `378.43` | `424.85` |
| `gpu_owner_publish_period_ms` | `92.54` | `245.47` | `278.93` | `547.80` |
| `raw_fusion_publish_period_ms` | `233.85` | `281.67` | `370.39` | `393.58` |
| `render_period_ms` | `234.73` | `287.26` | `368.26` | `407.18` |
| `stage_join_publish_period_ms` | `92.54` | `245.47` | `278.93` | `547.80` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `17.17` | `22.67` | `27.32` | `43.38` |
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
| `gpu_owner_total_ms` | `64.67` | `70.50` | `72.79` | `92.57` |
| `gpu_owner_ffs_cycle_ms` | `62.46` | `65.15` | `65.85` | `74.64` |
| `gpu_owner_edgetam_cycle_ms` | `63.82` | `70.04` | `72.45` | `92.57` |
| `raw_fusion_total_ms` | `11.34` | `17.52` | `18.82` | `25.88` |
| `fusion_total_ms` | `88.52` | `99.53` | `235.33` | `247.13` |
| `filter_total_ms` | `76.62` | `84.33` | `222.65` | `237.72` |
| `filter_input_age_ms` | `76.64` | `84.35` | `222.68` | `237.75` |
| `object_enhanced_pt_ms` | `47.08` | `52.84` | `192.05` | `210.85` |
| `controller_pt_filter_ms` | `29.82` | `34.32` | `36.02` | `177.43` |
| `render_total_ms` | `4.01` | `5.51` | `7.45` | `17.64` |
| `render_queue_wait_ms` | `146.66` | `161.02` | `164.47` | `173.46` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.15` | `0.46` | `0.70` | `13.39` |
| `render_cpu_format_ms` | `0.41` | `0.96` | `1.65` | `14.22` |
| `render_open3d_points_update_ms` | `0.10` | `0.16` | `0.24` | `1.83` |
| `render_open3d_colors_update_ms` | `0.12` | `0.32` | `0.35` | `1.99` |
| `render_open3d_update_geometry_ms` | `3.41` | `4.40` | `4.66` | `12.63` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1146` | `210.85` | `24000` | `9503` |
| `694` | `207.10` | `24000` | `9617` |
| `957` | `197.42` | `24000` | `9637` |
| `1048` | `196.45` | `24000` | `9559` |
| `782` | `192.87` | `24000` | `9493` |
| `1232` | `191.78` | `24000` | `9392` |
| `406` | `54.19` | `24000` | `9530` |
| `688` | `54.05` | `24000` | `9472` |
| `944` | `53.18` | `24000` | `9501` |
| `902` | `52.91` | `24000` | `9476` |
