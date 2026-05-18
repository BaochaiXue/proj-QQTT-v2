# Demo 2.1 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
- target FPS: `30.00`
- capture group target FPS: `30.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `5.00`
- filter output FPS after warmup: `10.17`
- fusion FPS after warmup: `10.17`
- stage period p50 after warmup: `99.92 ms`
- display packet period p50 after warmup: `97.79 ms`
- groups after warmup: `626`
- complete fused groups after warmup: `24`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.038`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `23`
- target deficit: `30.00`
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
| camera startup ms | `11512.89` |
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
| time to first complete group s | `32.50` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `154`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `1.00` | `60.40` | `63.70` | `98.00` |
| `memory_util_pct` | `0.00` | `48.00` | `51.35` | `66.00` |
| `memory_used_mb` | `3671.88` | `5853.24` | `5880.07` | `8007.12` |
| `power_w` | `55.91` | `285.27` | `291.72` | `297.75` |
| `sm_clock_mhz` | `2565.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10501.00` | `10501.00` |
| `temperature_c` | `45.00` | `57.00` | `59.00` | `60.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.46` | `41.82` | `46.98` | `252.25` |
| `display_packet_publish_period_ms` | `97.79` | `129.05` | `138.05` | `138.83` |
| `edgetam_stage_publish_period_ms` | `59.07` | `119.14` | `161.50` | `304.58` |
| `ffs_stage_publish_period_ms` | `44.35` | `61.29` | `76.59` | `279.59` |
| `filter_output_publish_period_ms` | `97.78` | `129.05` | `138.04` | `138.84` |
| `fusion_publish_period_ms` | `97.78` | `129.05` | `138.04` | `138.84` |
| `gpu_owner_publish_period_ms` | `99.92` | `164.90` | `708.89` | `1754.79` |
| `raw_fusion_publish_period_ms` | `99.93` | `164.91` | `708.91` | `1754.80` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `99.92` | `164.90` | `708.90` | `1754.79` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `7.15` | `24.43` | `25.06` | `38.64` |
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
| `gpu_owner_total_ms` | `55.13` | `60.55` | `63.48` | `83.19` |
| `gpu_owner_ffs_cycle_ms` | `40.14` | `42.09` | `42.26` | `43.15` |
| `gpu_owner_edgetam_cycle_ms` | `55.13` | `60.55` | `63.48` | `83.19` |
| `raw_fusion_total_ms` | `48.55` | `95.71` | `218.37` | `249.09` |
| `fusion_total_ms` | `116.96` | `159.30` | `310.61` | `2777.63` |
| `filter_total_ms` | `64.48` | `77.04` | `86.53` | `2530.04` |
| `filter_input_age_ms` | `68.15` | `107.23` | `1534.78` | `2530.15` |
| `object_enhanced_pt_ms` | `45.03` | `51.60` | `56.16` | `2502.22` |
| `controller_pt_filter_ms` | `23.16` | `34.39` | `34.95` | `35.78` |
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
| `498` | `2502.22` | `47915` | `8026` |
| `594` | `56.70` | `48083` | `8242` |
| `556` | `53.07` | `48073` | `7963` |
| `609` | `48.17` | `48035` | `10662` |
| `583` | `46.85` | `48041` | `7980` |
| `606` | `46.73` | `48100` | `8079` |
| `596` | `46.69` | `48148` | `10897` |
| `604` | `46.34` | `48099` | `8033` |
| `612` | `45.60` | `48018` | `8002` |
| `565` | `45.55` | `48120` | `8102` |
