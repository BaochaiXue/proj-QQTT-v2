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
- render FPS after warmup: `5.33`
- raw fusion FPS after warmup: `5.33`
- filter output FPS after warmup: `5.31`
- fusion FPS after warmup: `5.31`
- stage period p50 after warmup: `96.91 ms`
- display packet period p50 after warmup: `159.37 ms`
- groups after warmup: `5096`
- complete fused groups after warmup: `176`
- rendered groups after warmup: `175`
- complete group ratio after warmup: `0.035`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.67`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10728.67` |
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
| time to first complete group s | `160.80` |
| time to first rendered group s | `160.86` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `728`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `32.00` | `92.00` | `99.00` | `100.00` |
| `memory_util_pct` | `19.00` | `50.30` | `56.00` | `63.00` |
| `memory_used_mb` | `4935.44` | `15598.81` | `20264.75` | `24250.75` |
| `power_w` | `60.47` | `301.65` | `336.25` | `390.72` |
| `sm_clock_mhz` | `2565.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `48.00` | `65.00` | `70.00` | `73.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `34.11` | `37.26` | `226.77` |
| `display_packet_publish_period_ms` | `159.37` | `297.32` | `317.81` | `680.97` |
| `edgetam_stage_publish_period_ms` | `69.49` | `95.25` | `103.53` | `126535.61` |
| `ffs_stage_publish_period_ms` | `59.66` | `94.32` | `101.52` | `126573.99` |
| `filter_output_publish_period_ms` | `157.44` | `299.51` | `321.21` | `662.45` |
| `fusion_publish_period_ms` | `157.44` | `299.50` | `321.20` | `662.45` |
| `gpu_owner_publish_period_ms` | `96.91` | `259.40` | `290.18` | `606.94` |
| `raw_fusion_publish_period_ms` | `157.43` | `284.04` | `298.00` | `650.90` |
| `render_period_ms` | `158.17` | `298.51` | `320.50` | `685.23` |
| `stage_join_publish_period_ms` | `96.91` | `259.40` | `290.18` | `606.94` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.15` | `16.23` | `20.38` | `62.84` |
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
| `gpu_owner_total_ms` | `64.70` | `71.73` | `75.75` | `98.16` |
| `gpu_owner_ffs_cycle_ms` | `62.21` | `65.61` | `66.97` | `94.13` |
| `gpu_owner_edgetam_cycle_ms` | `64.04` | `70.88` | `74.46` | `98.16` |
| `raw_fusion_total_ms` | `9.19` | `14.58` | `17.94` | `26.60` |
| `fusion_total_ms` | `88.12` | `100.59` | `238.31` | `257.38` |
| `filter_total_ms` | `78.04` | `89.76` | `228.91` | `248.17` |
| `filter_input_age_ms` | `78.07` | `89.78` | `228.94` | `248.21` |
| `object_enhanced_pt_ms` | `45.95` | `51.89` | `54.63` | `221.51` |
| `controller_pt_filter_ms` | `32.35` | `37.77` | `42.20` | `201.99` |
| `render_total_ms` | `3.64` | `5.00` | `5.96` | `16.16` |
| `render_queue_wait_ms` | `39.01` | `47.34` | `49.38` | `57.71` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.12` | `0.28` | `0.39` | `12.12` |
| `render_cpu_format_ms` | `0.40` | `0.76` | `1.29` | `13.15` |
| `render_open3d_points_update_ms` | `0.10` | `0.14` | `0.22` | `2.24` |
| `render_open3d_colors_update_ms` | `0.13` | `0.37` | `0.41` | `2.31` |
| `render_open3d_update_geometry_ms` | `3.09` | `3.96` | `4.28` | `8.19` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `4733` | `221.51` | `24000` | `9340` |
| `4960` | `216.83` | `24000` | `9316` |
| `5038` | `213.81` | `24000` | `9424` |
| `4593` | `213.22` | `24000` | `9359` |
| `4887` | `208.46` | `24000` | `9314` |
| `4526` | `199.29` | `24000` | `9372` |
| `4815` | `197.71` | `24000` | `9304` |
| `4155` | `60.03` | `24000` | `9394` |
| `4637` | `54.93` | `24000` | `9267` |
| `4550` | `54.53` | `24000` | `9448` |
