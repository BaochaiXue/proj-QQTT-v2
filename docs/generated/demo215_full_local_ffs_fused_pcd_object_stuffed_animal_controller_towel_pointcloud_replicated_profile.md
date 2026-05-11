# Demo 2.1 performance profile

- preset: `demo2.1.5-live-quality-ffs`
- canonical preset: `demo2.1.5-live-quality-ffs`
- target FPS: `25.00`
- capture group target FPS: `15.00`
- compile mode: `none`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `2.50`
- raw fusion FPS after warmup: `5.01`
- filter output FPS after warmup: `5.00`
- fusion FPS after warmup: `5.00`
- groups after warmup: `989`
- complete fused groups after warmup: `364`
- rendered groups after warmup: `181`
- complete group ratio after warmup: `0.368`
- target deficit: `22.50`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `3764.93` |
| camera startup ms | `7837.91` |
| EdgeTAM model load ms | `2119.05` |
| EdgeTAM compile wrap ms | `0.00` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `594.05` |
| SAM3.1 model load ms | `6771.52` |
| SAM3.1 cam0 segment ms | `7201.35` |
| SAM3.1 cam1 segment ms | `166.09` |
| SAM3.1 cam2 segment ms | `172.18` |
| FFS runner init ms | `7612.13` |
| FFS first run ms | `925.55` |
| session init + prompt add ms | `4.10` |
| SAM3.1 release cleanup ms | `226.49` |
| time to first complete group s | `22.95` |
| time to first rendered group s | `23.71` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.200`
- samples after warmup: `358`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `43.00` | `47.00` | `48.00` | `54.00` |
| `memory_util_pct` | `11.00` | `16.00` | `17.00` | `22.00` |
| `memory_used_mb` | `9602.01` | `13166.14` | `13508.62` | `13847.42` |
| `power_w` | `120.74` | `153.95` | `170.53` | `249.67` |
| `sm_clock_mhz` | `180.00` | `1110.00` | `1110.00` | `1597.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `66.00` | `69.00` | `69.00` | `70.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `24.25` | `49.91` | `57.84` | `66.11` |
| `edgetam_model_ms` | `40.79` | `52.09` | `56.98` | `111.66` |
| `edgetam_preprocess_ms` | `1.25` | `1.75` | `1.92` | `6.57` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.13` | `0.16` | `0.41` |
| `edgetam_mask_resize_ms` | `0.04` | `0.08` | `0.10` | `0.32` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.05` | `0.06` | `0.31` |
| `edgetam_mask_to_cpu_ms` | `0.20` | `0.24` | `0.26` | `1.55` |
| `edgetam_total_ms` | `39.82` | `50.39` | `54.85` | `108.75` |
| `ffs_cycle_ms` | `68.95` | `75.73` | `79.75` | `259.85` |
| `ffs_batch_ms` | `44.70` | `51.21` | `53.79` | `233.71` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `43.15` | `53.68` | `57.60` | `111.16` |
| `edgetam_cam1_model_ms` | `39.68` | `49.56` | `56.33` | `110.08` |
| `edgetam_cam2_model_ms` | `39.63` | `51.56` | `56.17` | `111.66` |
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
| `ffs_stage_ms` | `1.77` | `2.81` | `3.50` | `37.60` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.77` | `2.80` | `3.48` | `37.60` |
| `ffs_cam1_stage_ms` | `1.77` | `2.80` | `3.48` | `37.60` |
| `ffs_cam2_stage_ms` | `1.77` | `2.80` | `3.48` | `37.60` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `189.31` | `222.20` | `259.38` | `409.39` |
| `gpu_owner_ffs_cycle_ms` | `68.95` | `75.73` | `79.75` | `259.85` |
| `gpu_owner_edgetam_cycle_ms` | `119.69` | `144.70` | `156.81` | `237.14` |
| `raw_fusion_total_ms` | `11.78` | `13.96` | `14.73` | `49.61` |
| `fusion_total_ms` | `54.65` | `61.64` | `64.18` | `244.41` |
| `filter_total_ms` | `42.71` | `48.77` | `51.49` | `231.90` |
| `filter_input_age_ms` | `43.22` | `49.28` | `52.37` | `232.65` |
| `object_enhanced_pt_ms` | `29.14` | `34.60` | `37.01` | `219.51` |
| `controller_pt_filter_ms` | `12.94` | `15.79` | `16.88` | `23.01` |
| `render_total_ms` | `0.46` | `0.72` | `0.96` | `1.75` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.04` | `1.04` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `1.35` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `842` | `219.51` | `73240` | `7928` |
| `747` | `218.63` | `73178` | `8012` |
| `373` | `218.45` | `73441` | `8001` |
| `560` | `217.79` | `73469` | `7955` |
| `1390` | `217.23` | `73510` | `7962` |
| `941` | `216.17` | `73266` | `8013` |
| `1128` | `216.08` | `73453` | `7932` |
| `656` | `213.33` | `73290` | `7984` |
| `287` | `211.16` | `73441` | `7911` |
| `1035` | `210.86` | `73183` | `7967` |
