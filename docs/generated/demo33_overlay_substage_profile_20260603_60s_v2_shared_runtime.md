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
- stage period p50 after warmup: `89.22 ms`
- display packet period p50 after warmup: `237.02 ms`
- groups after warmup: `1517`
- complete fused groups after warmup: `157`
- rendered groups after warmup: `156`
- complete group ratio after warmup: `0.103`
- stage drop count after warmup: `0`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-185224/case`
- object points0: `77019`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10703.36` |
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
| time to first complete group s | `34.70` |
| time to first rendered group s | `34.86` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `256`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `78.00` | `97.00` | `97.00` | `99.00` |
| `memory_util_pct` | `42.00` | `59.00` | `60.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `7994.69` | `8011.19` | `8067.50` |
| `power_w` | `282.00` | `303.70` | `318.15` | `347.77` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `63.00` | `71.00` | `72.00` | `75.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `44.14` | `56.72` | `221.94` |
| `display_packet_publish_period_ms` | `237.02` | `260.02` | `393.07` | `424.79` |
| `edgetam_stage_publish_period_ms` | `68.96` | `99.44` | `108.20` | `1147.42` |
| `ffs_stage_publish_period_ms` | `58.83` | `100.35` | `107.85` | `1194.18` |
| `filter_output_publish_period_ms` | `238.41` | `263.70` | `388.26` | `420.81` |
| `fusion_publish_period_ms` | `238.41` | `263.70` | `388.26` | `420.80` |
| `gpu_owner_publish_period_ms` | `89.22` | `225.90` | `276.94` | `493.21` |
| `raw_fusion_publish_period_ms` | `237.52` | `258.06` | `388.04` | `431.03` |
| `render_period_ms` | `237.97` | `267.72` | `388.29` | `413.41` |
| `stage_join_publish_period_ms` | `89.22` | `225.90` | `276.94` | `493.21` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `14.04` | `23.28` | `23.56` | `50.91` |
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
| `gpu_owner_total_ms` | `64.59` | `70.49` | `72.51` | `102.21` |
| `gpu_owner_ffs_cycle_ms` | `62.60` | `65.24` | `66.43` | `91.99` |
| `gpu_owner_edgetam_cycle_ms` | `63.94` | `70.18` | `72.22` | `102.21` |
| `raw_fusion_total_ms` | `12.70` | `17.81` | `18.64` | `20.69` |
| `fusion_total_ms` | `90.44` | `102.72` | `243.96` | `260.68` |
| `filter_total_ms` | `77.82` | `90.07` | `228.95` | `248.01` |
| `filter_input_age_ms` | `77.85` | `90.10` | `228.98` | `248.04` |
| `object_enhanced_pt_ms` | `47.38` | `54.77` | `57.93` | `216.60` |
| `controller_pt_filter_ms` | `30.65` | `35.55` | `38.64` | `196.00` |
| `render_total_ms` | `4.15` | `5.28` | `6.22` | `18.29` |
| `render_queue_wait_ms` | `151.68` | `165.63` | `167.61` | `176.35` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.58` | `1.02` | `2.63` |
| `render_cpu_format_ms` | `0.44` | `1.16` | `2.12` | `13.91` |
| `render_open3d_points_update_ms` | `0.11` | `0.19` | `0.28` | `1.64` |
| `render_open3d_colors_update_ms` | `0.14` | `0.35` | `0.44` | `13.41` |
| `render_open3d_update_geometry_ms` | `3.48` | `4.26` | `4.43` | `6.91` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `772` | `216.60` | `24000` | `9562` |
| `954` | `215.62` | `24000` | `9671` |
| `1314` | `209.03` | `24000` | `9643` |
| `685` | `202.75` | `24000` | `9543` |
| `1044` | `202.14` | `24000` | `9550` |
| `862` | `196.18` | `24000` | `9565` |
| `1588` | `60.29` | `24000` | `9626` |
| `1436` | `60.06` | `24000` | `9699` |
| `970` | `57.40` | `24000` | `9477` |
| `898` | `56.48` | `24000` | `9631` |
