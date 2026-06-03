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
- render FPS after warmup: `3.48`
- raw fusion FPS after warmup: `3.48`
- filter output FPS after warmup: `3.48`
- fusion FPS after warmup: `3.48`
- stage period p50 after warmup: `102.09 ms`
- display packet period p50 after warmup: `239.21 ms`
- groups after warmup: `2350`
- complete fused groups after warmup: `244`
- rendered groups after warmup: `243`
- complete group ratio after warmup: `0.104`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `26.52`
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
- status: `running`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-171225/case`
- object points0: `0`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10741.81` |
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
| time to first complete group s | `33.70` |
| time to first rendered group s | `33.85` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `374`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `78.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `42.00` | `57.00` | `59.35` | `62.00` |
| `memory_used_mb` | `7773.62` | `10126.75` | `13800.75` | `16477.44` |
| `power_w` | `287.14` | `346.11` | `347.22` | `372.28` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `64.00` | `70.00` | `72.00` | `75.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.37` | `43.50` | `57.22` | `232.15` |
| `display_packet_publish_period_ms` | `239.21` | `393.05` | `618.26` | `1119.42` |
| `edgetam_stage_publish_period_ms` | `74.68` | `111.64` | `127.69` | `376.40` |
| `ffs_stage_publish_period_ms` | `65.62` | `105.45` | `119.25` | `383.52` |
| `filter_output_publish_period_ms` | `238.60` | `390.14` | `599.88` | `1115.21` |
| `fusion_publish_period_ms` | `238.61` | `390.14` | `599.88` | `1115.21` |
| `gpu_owner_publish_period_ms` | `102.09` | `266.95` | `385.45` | `1208.95` |
| `raw_fusion_publish_period_ms` | `240.22` | `393.90` | `597.80` | `1114.29` |
| `render_period_ms` | `240.49` | `389.08` | `624.62` | `1123.60` |
| `stage_join_publish_period_ms` | `102.09` | `266.95` | `385.45` | `1208.95` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `7.41` | `27.49` | `31.65` | `65.14` |
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
| `gpu_owner_total_ms` | `66.43` | `99.18` | `100.41` | `110.02` |
| `gpu_owner_ffs_cycle_ms` | `63.25` | `91.07` | `93.26` | `110.02` |
| `gpu_owner_edgetam_cycle_ms` | `66.05` | `98.71` | `100.15` | `104.87` |
| `raw_fusion_total_ms` | `11.68` | `17.95` | `19.97` | `24.21` |
| `fusion_total_ms` | `89.08` | `105.89` | `239.72` | `290.71` |
| `filter_total_ms` | `76.73` | `94.26` | `229.45` | `279.30` |
| `filter_input_age_ms` | `76.75` | `94.28` | `229.47` | `279.33` |
| `object_enhanced_pt_ms` | `47.23` | `55.37` | `65.18` | `218.81` |
| `controller_pt_filter_ms` | `29.61` | `36.08` | `39.40` | `215.57` |
| `render_total_ms` | `3.90` | `5.47` | `6.44` | `19.27` |
| `render_queue_wait_ms` | `147.52` | `162.11` | `167.45` | `203.53` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.39` | `0.64` | `13.81` |
| `render_cpu_format_ms` | `0.38` | `0.84` | `1.80` | `16.72` |
| `render_open3d_points_update_ms` | `0.11` | `0.20` | `0.26` | `10.85` |
| `render_open3d_colors_update_ms` | `0.10` | `0.20` | `0.32` | `10.89` |
| `render_open3d_update_geometry_ms` | `3.37` | `4.35` | `5.54` | `7.38` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1751` | `218.81` | `24000` | `9459` |
| `1931` | `216.82` | `24000` | `9520` |
| `1661` | `211.43` | `24000` | `9537` |
| `2210` | `208.37` | `24000` | `9503` |
| `1105` | `207.36` | `24000` | `9577` |
| `1840` | `204.86` | `24000` | `9677` |
| `849` | `204.83` | `24000` | `9482` |
| `2306` | `203.77` | `24000` | `9483` |
| `2114` | `201.70` | `24000` | `9523` |
| `663` | `199.48` | `24000` | `9443` |
