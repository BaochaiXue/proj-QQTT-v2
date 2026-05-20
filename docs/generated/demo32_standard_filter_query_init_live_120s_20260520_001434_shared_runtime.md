# Demo 3.2 performance profile

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
- render FPS after warmup: `5.81`
- raw fusion FPS after warmup: `15.32`
- filter output FPS after warmup: `15.32`
- fusion FPS after warmup: `15.32`
- stage period p50 after warmup: `63.80 ms`
- display packet period p50 after warmup: `145.68 ms`
- groups after warmup: `3366`
- complete fused groups after warmup: `1642`
- rendered groups after warmup: `611`
- complete group ratio after warmup: `0.488`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.19`
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
| camera startup ms | `10828.09` |
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
| time to first complete group s | `26.51` |
| time to first rendered group s | `28.60` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `494`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `87.00` | `92.00` | `94.00` | `95.00` |
| `memory_util_pct` | `48.00` | `81.00` | `84.00` | `87.00` |
| `memory_used_mb` | `6666.69` | `7773.12` | `7803.94` | `8594.69` |
| `power_w` | `303.50` | `326.64` | `329.79` | `336.46` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `65.00` | `75.00` | `76.00` | `79.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.52` | `36.15` | `37.36` | `194.64` |
| `display_packet_publish_period_ms` | `145.68` | `157.47` | `160.40` | `15123.36` |
| `edgetam_stage_publish_period_ms` | `68.85` | `78.46` | `94.72` | `222.98` |
| `ffs_stage_publish_period_ms` | `66.11` | `71.27` | `72.72` | `224.44` |
| `filter_output_publish_period_ms` | `63.74` | `81.06` | `83.36` | `238.76` |
| `fusion_publish_period_ms` | `63.74` | `81.06` | `83.36` | `238.76` |
| `gpu_owner_publish_period_ms` | `63.80` | `80.77` | `82.69` | `236.83` |
| `raw_fusion_publish_period_ms` | `63.79` | `80.70` | `82.64` | `236.56` |
| `render_period_ms` | `146.13` | `158.31` | `162.13` | `15126.25` |
| `stage_join_publish_period_ms` | `63.80` | `80.77` | `82.69` | `236.83` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `5.47` | `26.08` | `26.57` | `46.31` |
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
| `gpu_owner_total_ms` | `64.63` | `69.84` | `71.48` | `12677.87` |
| `gpu_owner_ffs_cycle_ms` | `62.06` | `64.56` | `65.21` | `111.97` |
| `gpu_owner_edgetam_cycle_ms` | `64.21` | `69.83` | `71.48` | `12677.87` |
| `raw_fusion_total_ms` | `7.96` | `10.27` | `11.16` | `18.48` |
| `fusion_total_ms` | `18.56` | `21.62` | `22.63` | `34.19` |
| `filter_total_ms` | `10.47` | `12.27` | `13.05` | `16.48` |
| `filter_input_age_ms` | `11.12` | `12.97` | `13.69` | `17.07` |
| `object_enhanced_pt_ms` | `4.19` | `5.34` | `5.92` | `9.11` |
| `controller_pt_filter_ms` | `6.14` | `7.39` | `7.98` | `11.30` |
| `render_total_ms` | `1.64` | `2.00` | `2.29` | `6.75` |
| `render_queue_wait_ms` | `208.21` | `262.40` | `269.20` | `2081.60` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.07` | `0.19` | `0.36` | `5.18` |
| `render_cpu_format_ms` | `0.23` | `0.45` | `0.65` | `5.31` |
| `render_open3d_points_update_ms` | `0.07` | `0.11` | `0.14` | `2.63` |
| `render_open3d_colors_update_ms` | `0.06` | `0.14` | `0.16` | `1.68` |
| `render_open3d_update_geometry_ms` | `1.29` | `1.56` | `1.64` | `4.15` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1427` | `9.11` | `24000` | `6217` |
| `582` | `8.75` | `24000` | `6283` |
| `2579` | `7.89` | `24000` | `6054` |
| `1379` | `7.76` | `24000` | `6189` |
| `3335` | `7.69` | `24000` | `5994` |
| `1606` | `7.48` | `24000` | `5956` |
| `1604` | `7.40` | `24000` | `5835` |
| `3203` | `7.28` | `24000` | `6103` |
| `1514` | `7.21` | `24000` | `6225` |
| `2049` | `7.16` | `24000` | `5994` |
