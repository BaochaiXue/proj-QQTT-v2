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
- render FPS after warmup: `4.04`
- raw fusion FPS after warmup: `4.04`
- filter output FPS after warmup: `4.05`
- fusion FPS after warmup: `4.05`
- stage period p50 after warmup: `91.64 ms`
- display packet period p50 after warmup: `235.98 ms`
- groups after warmup: `1501`
- complete fused groups after warmup: `162`
- rendered groups after warmup: `161`
- complete group ratio after warmup: `0.108`
- stage drop count after warmup: `2`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.96`
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
- status: `snapshot_ready`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-171655/case`
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
| camera startup ms | `10727.69` |
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
| time to first complete group s | `33.74` |
| time to first rendered group s | `33.90` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `255`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `75.00` | `91.00` | `94.00` | `98.00` |
| `memory_util_pct` | `43.00` | `57.00` | `59.00` | `62.00` |
| `memory_used_mb` | `6105.44` | `7770.44` | `7771.50` | `8365.44` |
| `power_w` | `284.73` | `308.52` | `324.33` | `357.86` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `63.00` | `70.00` | `71.00` | `73.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `44.95` | `58.33` | `221.70` |
| `display_packet_publish_period_ms` | `235.98` | `263.06` | `383.22` | `412.84` |
| `edgetam_stage_publish_period_ms` | `68.79` | `99.55` | `107.45` | `365.97` |
| `ffs_stage_publish_period_ms` | `60.13` | `100.08` | `108.01` | `386.89` |
| `filter_output_publish_period_ms` | `236.23` | `267.31` | `382.88` | `413.50` |
| `fusion_publish_period_ms` | `236.23` | `267.31` | `382.88` | `413.49` |
| `gpu_owner_publish_period_ms` | `91.64` | `225.18` | `277.56` | `545.85` |
| `raw_fusion_publish_period_ms` | `236.63` | `260.33` | `383.28` | `406.47` |
| `render_period_ms` | `235.04` | `269.21` | `381.54` | `430.30` |
| `stage_join_publish_period_ms` | `91.64` | `225.18` | `277.56` | `545.85` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.63` | `20.26` | `29.72` | `54.63` |
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
| `gpu_owner_total_ms` | `64.70` | `70.06` | `71.58` | `94.23` |
| `gpu_owner_ffs_cycle_ms` | `61.90` | `65.33` | `66.39` | `88.40` |
| `gpu_owner_edgetam_cycle_ms` | `63.99` | `69.90` | `71.55` | `94.23` |
| `raw_fusion_total_ms` | `12.14` | `16.90` | `18.85` | `23.23` |
| `fusion_total_ms` | `89.61` | `100.09` | `234.06` | `263.15` |
| `filter_total_ms` | `78.09` | `88.30` | `224.00` | `249.81` |
| `filter_input_age_ms` | `78.11` | `88.33` | `224.02` | `249.83` |
| `object_enhanced_pt_ms` | `47.71` | `54.85` | `58.93` | `203.85` |
| `controller_pt_filter_ms` | `30.44` | `34.98` | `39.17` | `200.18` |
| `render_total_ms` | `3.93` | `6.58` | `8.50` | `18.14` |
| `render_queue_wait_ms` | `149.20` | `163.07` | `169.55` | `186.31` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.60` | `0.86` | `8.36` |
| `render_cpu_format_ms` | `0.41` | `1.22` | `3.79` | `15.04` |
| `render_open3d_points_update_ms` | `0.11` | `0.21` | `0.25` | `14.58` |
| `render_open3d_colors_update_ms` | `0.13` | `0.30` | `0.61` | `11.25` |
| `render_open3d_update_geometry_ms` | `3.37` | `4.29` | `5.34` | `7.48` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1565` | `203.85` | `24000` | `9459` |
| `1469` | `200.71` | `24000` | `9538` |
| `1011` | `199.83` | `24000` | `9546` |
| `834` | `196.89` | `24000` | `9422` |
| `1194` | `194.48` | `24000` | `9571` |
| `1100` | `191.52` | `24000` | `9566` |
| `751` | `185.34` | `24000` | `9475` |
| `375` | `60.20` | `24000` | `9409` |
| `1411` | `58.95` | `24000` | `9560` |
| `1298` | `58.39` | `24000` | `9622` |
