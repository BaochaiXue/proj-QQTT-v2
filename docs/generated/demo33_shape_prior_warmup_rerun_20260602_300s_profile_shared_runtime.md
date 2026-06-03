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
- render FPS after warmup: `5.70`
- raw fusion FPS after warmup: `5.71`
- filter output FPS after warmup: `5.71`
- fusion FPS after warmup: `5.71`
- stage period p50 after warmup: `116.91 ms`
- display packet period p50 after warmup: `149.18 ms`
- groups after warmup: `8399`
- complete fused groups after warmup: `663`
- rendered groups after warmup: `660`
- complete group ratio after warmup: `0.079`
- stage drop count after warmup: `4`
- raw fused pending replacements total: `0`
- render buffer dropped total: `2`
- target deficit: `24.30`
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
- status: `ready`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260602-201602/case`
- object points0: `9308`
- surface points: `494`
- interior points: `1048`
- structure points: `10850`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10760.86` |
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
| time to first complete group s | `197.89` |
| time to first rendered group s | `197.94` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `1200`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `31.50` | `92.10` | `97.05` | `100.00` |
| `memory_util_pct` | `20.00` | `56.00` | `58.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `15602.75` | `15618.81` | `24534.75` |
| `power_w` | `71.42` | `330.99` | `356.91` | `381.63` |
| `sm_clock_mhz` | `2565.00` | `2655.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `55.00` | `77.00` | `81.00` | `84.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `37.55` | `53.12` | `244.76` |
| `display_packet_publish_period_ms` | `149.18` | `283.95` | `312.30` | `458.78` |
| `edgetam_stage_publish_period_ms` | `69.71` | `99.78` | `108.66` | `164204.03` |
| `ffs_stage_publish_period_ms` | `60.88` | `100.86` | `112.37` | `164223.31` |
| `filter_output_publish_period_ms` | `150.48` | `283.19` | `311.42` | `464.73` |
| `fusion_publish_period_ms` | `150.47` | `283.19` | `311.42` | `464.73` |
| `gpu_owner_publish_period_ms` | `116.91` | `257.86` | `294.65` | `470.70` |
| `raw_fusion_publish_period_ms` | `150.41` | `275.09` | `293.22` | `468.46` |
| `render_period_ms` | `149.97` | `284.81` | `312.79` | `464.03` |
| `stage_join_publish_period_ms` | `116.91` | `257.86` | `294.65` | `470.70` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.62` | `22.94` | `30.41` | `44.52` |
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
| `gpu_owner_total_ms` | `64.84` | `71.10` | `72.65` | `101.31` |
| `gpu_owner_ffs_cycle_ms` | `61.85` | `65.11` | `66.35` | `101.31` |
| `gpu_owner_edgetam_cycle_ms` | `64.22` | `70.90` | `72.43` | `99.99` |
| `raw_fusion_total_ms` | `8.86` | `15.34` | `17.27` | `27.76` |
| `fusion_total_ms` | `86.45` | `98.85` | `240.92` | `277.26` |
| `filter_total_ms` | `76.53` | `85.66` | `233.44` | `265.81` |
| `filter_input_age_ms` | `76.56` | `85.69` | `233.47` | `265.84` |
| `object_enhanced_pt_ms` | `46.26` | `52.26` | `54.18` | `230.86` |
| `controller_pt_filter_ms` | `30.08` | `35.77` | `39.25` | `217.51` |
| `render_total_ms` | `3.54` | `4.53` | `4.97` | `18.22` |
| `render_queue_wait_ms` | `38.59` | `47.93` | `50.77` | `126.93` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.11` | `0.33` | `0.49` | `12.12` |
| `render_cpu_format_ms` | `0.36` | `0.71` | `0.97` | `14.40` |
| `render_open3d_points_update_ms` | `0.10` | `0.15` | `0.23` | `2.19` |
| `render_open3d_colors_update_ms` | `0.10` | `0.28` | `0.37` | `12.21` |
| `render_open3d_update_geometry_ms` | `3.03` | `3.82` | `3.94` | `7.46` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.96` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `8409` | `230.86` | `24000` | `9498` |
| `7875` | `230.83` | `24000` | `9383` |
| `8345` | `229.85` | `24000` | `9498` |
| `7805` | `223.20` | `24000` | `9506` |
| `8002` | `222.51` | `24000` | `9562` |
| `6464` | `220.67` | `24000` | `9528` |
| `8200` | `220.38` | `24000` | `9656` |
| `5918` | `215.55` | `24000` | `9426` |
| `6383` | `214.75` | `24000` | `9560` |
| `7672` | `213.20` | `24000` | `9542` |
