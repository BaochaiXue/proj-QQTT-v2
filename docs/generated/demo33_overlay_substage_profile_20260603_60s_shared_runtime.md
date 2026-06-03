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
- render FPS after warmup: `4.03`
- raw fusion FPS after warmup: `4.02`
- filter output FPS after warmup: `4.02`
- fusion FPS after warmup: `4.02`
- stage period p50 after warmup: `91.41 ms`
- display packet period p50 after warmup: `232.09 ms`
- groups after warmup: `1509`
- complete fused groups after warmup: `154`
- rendered groups after warmup: `153`
- complete group ratio after warmup: `0.102`
- stage drop count after warmup: `2`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.97`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-184537/case`
- object points0: `76975`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10709.79` |
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
| time to first complete group s | `35.54` |
| time to first rendered group s | `35.72` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `256`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `77.00` | `95.00` | `98.00` | `98.00` |
| `memory_util_pct` | `43.00` | `58.00` | `60.00` | `60.00` |
| `memory_used_mb` | `6105.44` | `8004.94` | `8006.00` | `8365.44` |
| `power_w` | `285.34` | `311.53` | `322.99` | `350.91` |
| `sm_clock_mhz` | `2655.00` | `2662.50` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `63.00` | `73.00` | `74.00` | `76.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.40` | `41.48` | `56.36` | `232.28` |
| `display_packet_publish_period_ms` | `232.09` | `278.28` | `387.56` | `699.43` |
| `edgetam_stage_publish_period_ms` | `68.91` | `100.48` | `106.15` | `1476.37` |
| `ffs_stage_publish_period_ms` | `58.82` | `100.18` | `109.64` | `1502.59` |
| `filter_output_publish_period_ms` | `233.36` | `277.64` | `386.17` | `676.53` |
| `fusion_publish_period_ms` | `233.36` | `277.64` | `386.16` | `676.53` |
| `gpu_owner_publish_period_ms` | `91.41` | `217.30` | `248.43` | `831.41` |
| `raw_fusion_publish_period_ms` | `234.69` | `285.83` | `384.84` | `682.68` |
| `render_period_ms` | `233.89` | `277.44` | `385.95` | `710.65` |
| `stage_join_publish_period_ms` | `91.41` | `217.30` | `248.43` | `831.41` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `3.87` | `28.25` | `29.51` | `62.38` |
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
| `gpu_owner_total_ms` | `64.53` | `70.07` | `71.75` | `106.05` |
| `gpu_owner_ffs_cycle_ms` | `62.22` | `64.84` | `65.69` | `82.69` |
| `gpu_owner_edgetam_cycle_ms` | `63.95` | `70.07` | `71.75` | `106.05` |
| `raw_fusion_total_ms` | `12.21` | `17.18` | `17.90` | `23.01` |
| `fusion_total_ms` | `89.76` | `99.93` | `241.63` | `261.35` |
| `filter_total_ms` | `77.07` | `88.51` | `227.20` | `253.70` |
| `filter_input_age_ms` | `77.09` | `88.54` | `227.22` | `253.72` |
| `object_enhanced_pt_ms` | `46.68` | `52.95` | `57.06` | `221.63` |
| `controller_pt_filter_ms` | `29.84` | `35.71` | `36.54` | `197.62` |
| `render_total_ms` | `4.16` | `6.02` | `6.66` | `17.76` |
| `render_queue_wait_ms` | `147.98` | `161.80` | `164.34` | `181.05` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `1.35` | `2.43` | `13.74` |
| `render_cpu_format_ms` | `0.41` | `2.22` | `2.88` | `14.16` |
| `render_open3d_points_update_ms` | `0.11` | `0.21` | `0.33` | `1.69` |
| `render_open3d_colors_update_ms` | `0.11` | `0.25` | `0.38` | `10.63` |
| `render_open3d_update_geometry_ms` | `3.38` | `4.36` | `4.56` | `7.49` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.07` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1611` | `221.63` | `24000` | `9562` |
| `1524` | `211.56` | `24000` | `9525` |
| `1338` | `208.72` | `24000` | `9520` |
| `1428` | `206.60` | `24000` | `9547` |
| `794` | `202.23` | `24000` | `9617` |
| `966` | `197.70` | `24000` | `9561` |
| `1051` | `194.68` | `24000` | `9579` |
| `1213` | `57.33` | `24000` | `9522` |
| `1426` | `56.92` | `24000` | `9504` |
| `1114` | `56.36` | `24000` | `9634` |
