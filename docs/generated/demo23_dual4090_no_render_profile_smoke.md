# Demo 2.1 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `13.27`
- filter output FPS after warmup: `14.02`
- fusion FPS after warmup: `14.02`
- stage period p50 after warmup: `68.10 ms`
- display packet period p50 after warmup: `67.72 ms`
- groups after warmup: `609`
- complete fused groups after warmup: `216`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.355`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `18`
- render buffer dropped total: `215`
- target deficit: `15.00`
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
| camera startup ms | `11038.81` |
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
| time to first complete group s | `41.96` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `200`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `26.00` | `47.00` | `54.05` | `100.00` |
| `memory_util_pct` | `9.00` | `34.00` | `41.25` | `100.00` |
| `memory_used_mb` | `3671.88` | `8235.31` | `9237.64` | `9249.75` |
| `power_w` | `145.08` | `213.35` | `218.31` | `272.31` |
| `sm_clock_mhz` | `2610.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `50.00` | `56.10` | `57.00` | `58.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.78` | `70.76` | `73.13` | `248.72` |
| `display_packet_publish_period_ms` | `67.72` | `72.78` | `74.88` | `221.80` |
| `edgetam_stage_publish_period_ms` | `68.00` | `80.80` | `108.11` | `314.15` |
| `ffs_stage_publish_period_ms` | `66.86` | `78.52` | `84.74` | `380.39` |
| `filter_output_publish_period_ms` | `67.73` | `72.79` | `74.89` | `221.80` |
| `fusion_publish_period_ms` | `67.73` | `72.79` | `74.88` | `221.80` |
| `gpu_owner_publish_period_ms` | `68.10` | `73.85` | `115.17` | `336.78` |
| `raw_fusion_publish_period_ms` | `68.10` | `73.84` | `115.17` | `336.79` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `68.10` | `73.85` | `115.17` | `336.78` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.01` | `23.93` | `29.30` | `66.54` |
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
| `gpu_owner_total_ms` | `53.65` | `57.83` | `58.55` | `88.57` |
| `gpu_owner_ffs_cycle_ms` | `39.71` | `45.63` | `46.63` | `50.34` |
| `gpu_owner_edgetam_cycle_ms` | `53.65` | `57.83` | `58.55` | `88.57` |
| `raw_fusion_total_ms` | `18.04` | `37.12` | `53.68` | `290.16` |
| `fusion_total_ms` | `77.17` | `92.45` | `110.90` | `1938.29` |
| `filter_total_ms` | `59.42` | `62.58` | `65.52` | `1913.15` |
| `filter_input_age_ms` | `60.03` | `63.19` | `66.57` | `1913.50` |
| `object_enhanced_pt_ms` | `29.52` | `42.69` | `46.54` | `1887.38` |
| `controller_pt_filter_ms` | `29.59` | `31.80` | `32.37` | `36.41` |
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
| `413` | `1887.38` | `47968` | `11607` |
| `460` | `186.81` | `48087` | `11636` |
| `541` | `172.79` | `48013` | `11658` |
| `596` | `171.51` | `48128` | `11619` |
| `623` | `167.61` | `48109` | `11543` |
| `487` | `164.65` | `48012` | `11627` |
| `515` | `164.35` | `48029` | `11571` |
| `568` | `163.34` | `48062` | `11680` |
| `442` | `48.30` | `48035` | `11706` |
| `597` | `47.98` | `48101` | `11700` |
