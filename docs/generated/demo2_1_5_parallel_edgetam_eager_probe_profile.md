# Demo 2.1.5 Parallel EdgeTAM Eager Probe

- preset: `demo2.1.5-compiled-parallel-edgetam-5fps`
- canonical preset: `demo2.1.5-compiled-parallel-edgetam-5fps`
- compile mode: `none`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `0.65`
- filter output FPS after warmup: `0.65`
- fusion FPS after warmup: `0.65`
- groups after warmup: `722`
- complete fused groups after warmup: `27`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.037`
- target deficit: `15.00`
- bottleneck class: `upstream_supply`
- GPU pipeline: `separate-workers`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `edge`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `2590.65` |
| camera startup ms | `5082.15` |
| EdgeTAM model load ms | `8424.25` |
| EdgeTAM compile wrap ms | `0.00` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `1528.05` |
| SAM3.1 model load ms | `9533.20` |
| SAM3.1 cam0 segment ms | `n/a` |
| SAM3.1 cam1 segment ms | `n/a` |
| SAM3.1 cam2 segment ms | `n/a` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `60.14` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `16.99` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.200`
- samples after warmup: `258`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `24.00` | `27.00` | `28.15` | `31.00` |
| `memory_util_pct` | `4.00` | `5.00` | `6.00` | `8.00` |
| `memory_used_mb` | `5574.10` | `5835.10` | `5855.10` | `5855.10` |
| `power_w` | `135.98` | `137.84` | `139.67` | `204.55` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `892.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `69.00` | `70.00` | `70.00` | `71.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `19.43` | `52.25` | `58.04` | `66.61` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `184.41` | `208.36` | `215.77` | `390.98` |
| `edgetam_cam1_model_ms` | `188.11` | `209.47` | `219.68` | `393.45` |
| `edgetam_cam2_model_ms` | `189.92` | `209.90` | `223.42` | `388.81` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `0.51` | `0.79` | `0.96` | `1.90` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.46` | `0.73` | `0.85` | `1.75` |
| `edge_cam1_pin_copy_ms` | `0.54` | `0.79` | `0.93` | `1.90` |
| `edge_cam2_pin_copy_ms` | `0.52` | `0.88` | `1.06` | `1.54` |
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
| `gpu_owner_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_total_ms` | `20.58` | `23.93` | `24.43` | `26.18` |
| `fusion_total_ms` | `65.34` | `70.95` | `71.89` | `257.22` |
| `filter_total_ms` | `45.29` | `49.01` | `50.62` | `240.85` |
| `filter_input_age_ms` | `45.97` | `49.22` | `51.01` | `241.10` |
| `object_enhanced_pt_ms` | `29.27` | `32.56` | `32.86` | `223.33` |
| `controller_pt_filter_ms` | `16.20` | `17.56` | `17.99` | `18.22` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `129` | `897.42` | `46531` | `8516` |
| `1216` | `223.33` | `46579` | `8481` |
| `243` | `218.75` | `46577` | `8458` |
| `194` | `38.47` | `46608` | `8468` |
| `197` | `37.15` | `46602` | `8466` |
| `236` | `33.05` | `46569` | `8435` |
| `1200` | `32.99` | `46621` | `8518` |
| `1227` | `32.56` | `46568` | `8528` |
| `715` | `32.55` | `46540` | `8433` |
| `434` | `31.68` | `46565` | `8478` |
