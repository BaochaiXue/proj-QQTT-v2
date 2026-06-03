# Demo 3.1 performance profile

- preset: `demo2.1.5-live-fast-native`
- canonical preset: `demo2.1.5-live-fast-native`
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
- render FPS after warmup: `5.51`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `5.51`
- stage period p50 after warmup: `98.08 ms`
- display packet period p50 after warmup: `168.85 ms`
- groups after warmup: `4613`
- complete fused groups after warmup: `953`
- rendered groups after warmup: `953`
- complete group ratio after warmup: `0.207`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.49`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `none`
- render filtered only: `False`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `0.01` |
| camera startup ms | `11293.99` |
| EdgeTAM model load ms | `582.66` |
| EdgeTAM compile wrap ms | `588.90` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `117.07` |
| SAM3.1 model load ms | `6817.41` |
| SAM3.1 cam0 segment ms | `537.17` |
| SAM3.1 cam1 segment ms | `122.00` |
| SAM3.1 cam2 segment ms | `116.77` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.23` |
| SAM3.1 release cleanup ms | `289.92` |
| time to first complete group s | `20.58` |
| time to first rendered group s | `20.59` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `690`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `45.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `75.00` | `75.00` | `77.00` |
| `memory_used_mb` | `2976.12` | `8211.06` | `8229.10` | `8275.50` |
| `power_w` | `128.58` | `362.47` | `364.61` | `369.59` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `51.00` | `81.00` | `82.00` | `84.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `49.23` | `54.42` | `362.26` |
| `display_packet_publish_period_ms` | `168.85` | `179.57` | `292.76` | `494.50` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `168.76` | `179.45` | `292.71` | `494.58` |
| `gpu_owner_publish_period_ms` | `98.08` | `125.52` | `129.89` | `506.77` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `168.87` | `184.12` | `295.28` | `498.60` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.64` | `24.08` | `26.17` | `45.10` |
| `edgetam_model_ms` | `16.70` | `66.25` | `76.75` | `395.40` |
| `edgetam_preprocess_ms` | `0.58` | `0.90` | `2.26` | `107.13` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.05` | `15.21` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.57` | `0.67` | `321.89` |
| `edgetam_total_ms` | `17.32` | `70.39` | `77.52` | `395.96` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.22` | `1.54` | `1.73` | `23.38` |
| `edgetam_batch_vision_total_ms` | `7.25` | `25.88` | `36.11` | `337.01` |
| `edgetam_batch_vision_preprocess_ms` | `1.74` | `2.71` | `6.71` | `321.39` |
| `edgetam_cam0_model_ms` | `18.96` | `72.50` | `77.51` | `393.80` |
| `edgetam_cam1_model_ms` | `15.60` | `62.64` | `76.43` | `395.40` |
| `edgetam_cam2_model_ms` | `14.97` | `64.28` | `76.14` | `386.78` |
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
| `gpu_owner_total_ms` | `98.02` | `125.47` | `129.85` | `506.73` |
| `gpu_owner_ffs_cycle_ms` | `0.32` | `0.81` | `1.43` | `22.69` |
| `gpu_owner_edgetam_cycle_ms` | `95.80` | `124.81` | `129.41` | `506.32` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `82.00` | `89.87` | `169.15` | `404.22` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `40.13` | `44.39` | `46.04` | `360.95` |
| `controller_pt_filter_ms` | `28.02` | `31.70` | `33.39` | `347.07` |
| `render_total_ms` | `1.72` | `1.99` | `2.09` | `5.96` |
| `render_queue_wait_ms` | `13.69` | `16.41` | `17.16` | `25.83` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.14` | `0.18` | `1.87` |
| `render_cpu_format_ms` | `0.26` | `0.38` | `0.43` | `3.51` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.12` | `3.32` |
| `render_open3d_colors_update_ms` | `0.08` | `0.16` | `0.18` | `1.49` |
| `render_open3d_update_geometry_ms` | `1.36` | `1.58` | `1.65` | `5.61` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `4165` | `360.95` | `24000` | `8736` |
| `4647` | `359.40` | `24000` | `8769` |
| `4404` | `357.03` | `24000` | `8845` |
| `3801` | `350.21` | `24000` | `8710` |
| `3336` | `345.40` | `24000` | `8729` |
| `2876` | `341.78` | `24000` | `8760` |
| `2424` | `334.46` | `24000` | `8817` |
| `2206` | `333.44` | `24000` | `8672` |
| `1666` | `333.31` | `24000` | `8769` |
| `1882` | `331.43` | `24000` | `8766` |
