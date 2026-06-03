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
- render FPS after warmup: `4.07`
- raw fusion FPS after warmup: `4.07`
- filter output FPS after warmup: `4.07`
- fusion FPS after warmup: `4.07`
- stage period p50 after warmup: `89.20 ms`
- display packet period p50 after warmup: `233.14 ms`
- groups after warmup: `1506`
- complete fused groups after warmup: `159`
- rendered groups after warmup: `158`
- complete group ratio after warmup: `0.106`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.93`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-173440/case`
- object points0: `76681`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10749.01` |
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
| time to first complete group s | `34.69` |
| time to first rendered group s | `34.85` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `256`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `74.00` | `91.00` | `94.00` | `98.00` |
| `memory_util_pct` | `40.00` | `57.00` | `59.00` | `62.00` |
| `memory_used_mb` | `6105.44` | `7763.94` | `7763.94` | `8365.44` |
| `power_w` | `280.69` | `310.73` | `324.65` | `363.09` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `62.00` | `71.00` | `73.00` | `74.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.40` | `45.53` | `57.43` | `203.99` |
| `display_packet_publish_period_ms` | `233.14` | `251.40` | `386.40` | `463.70` |
| `edgetam_stage_publish_period_ms` | `69.08` | `96.96` | `104.33` | `1270.03` |
| `ffs_stage_publish_period_ms` | `60.19` | `98.68` | `108.35` | `1266.28` |
| `filter_output_publish_period_ms` | `234.00` | `252.64` | `388.03` | `463.32` |
| `fusion_publish_period_ms` | `234.00` | `252.64` | `388.03` | `463.33` |
| `gpu_owner_publish_period_ms` | `89.20` | `221.51` | `271.50` | `435.51` |
| `raw_fusion_publish_period_ms` | `232.82` | `256.69` | `383.25` | `411.23` |
| `render_period_ms` | `234.47` | `257.19` | `381.67` | `464.63` |
| `stage_join_publish_period_ms` | `89.20` | `221.51` | `271.50` | `435.51` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.81` | `24.61` | `36.78` | `63.32` |
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
| `gpu_owner_total_ms` | `64.51` | `70.04` | `71.79` | `81.37` |
| `gpu_owner_ffs_cycle_ms` | `62.40` | `65.18` | `66.03` | `80.80` |
| `gpu_owner_edgetam_cycle_ms` | `63.77` | `70.00` | `71.64` | `81.37` |
| `raw_fusion_total_ms` | `12.27` | `17.57` | `18.93` | `21.81` |
| `fusion_total_ms` | `90.78` | `98.21` | `243.08` | `262.86` |
| `filter_total_ms` | `77.83` | `87.65` | `228.44` | `251.68` |
| `filter_input_age_ms` | `77.85` | `87.68` | `228.47` | `251.71` |
| `object_enhanced_pt_ms` | `46.55` | `50.69` | `53.87` | `202.46` |
| `controller_pt_filter_ms` | `32.02` | `37.25` | `176.36` | `203.04` |
| `render_total_ms` | `3.98` | `5.42` | `6.52` | `15.72` |
| `render_queue_wait_ms` | `147.53` | `159.62` | `162.86` | `171.42` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.61` | `0.87` | `2.64` |
| `render_cpu_format_ms` | `0.39` | `0.98` | `1.46` | `11.52` |
| `render_open3d_points_update_ms` | `0.11` | `0.16` | `0.20` | `0.76` |
| `render_open3d_colors_update_ms` | `0.11` | `0.20` | `0.32` | `11.27` |
| `render_open3d_update_geometry_ms` | `3.41` | `4.37` | `4.75` | `9.05` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `949` | `202.46` | `24000` | `9541` |
| `866` | `199.12` | `24000` | `9488` |
| `1496` | `196.21` | `24000` | `9596` |
| `960` | `58.02` | `24000` | `9545` |
| `1072` | `57.99` | `24000` | `9704` |
| `1611` | `54.95` | `24000` | `9655` |
| `1411` | `54.57` | `24000` | `9586` |
| `1500` | `54.20` | `24000` | `9592` |
| `1476` | `53.83` | `24000` | `9603` |
| `667` | `53.62` | `24000` | `9409` |
