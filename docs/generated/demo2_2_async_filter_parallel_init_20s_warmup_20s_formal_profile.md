# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `4.92`
- raw fusion FPS after warmup: `4.92`
- filter output FPS after warmup: `4.92`
- fusion FPS after warmup: `4.92`
- groups after warmup: `319`
- complete fused groups after warmup: `110`
- rendered groups after warmup: `110`
- complete group ratio after warmup: `0.345`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `10.08`
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
| parallel init max wait ms | `3352.61` |
| camera startup ms | `5119.36` |
| EdgeTAM model load ms | `821.22` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1300.89` |
| EdgeTAM warmup/first forward ms | `118.34` |
| SAM3.1 model load ms | `7741.03` |
| SAM3.1 cam0 segment ms | `334.71` |
| SAM3.1 cam1 segment ms | `158.32` |
| SAM3.1 cam2 segment ms | `160.03` |
| FFS runner init ms | `5932.12` |
| FFS first run ms | `917.11` |
| session init + prompt add ms | `2.67` |
| SAM3.1 release cleanup ms | `215.72` |
| time to first complete group s | `13.78` |
| time to first rendered group s | `13.80` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `28.98` | `52.32` | `55.59` | `66.68` |
| `ffs_cycle_ms` | `90.67` | `97.44` | `99.07` | `274.95` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `32.92` | `39.65` | `40.92` | `49.19` |
| `edgetam_cam1_model_ms` | `31.27` | `36.96` | `39.01` | `44.12` |
| `edgetam_cam2_model_ms` | `31.52` | `37.20` | `38.13` | `41.56` |
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
| `ffs_stage_ms` | `0.82` | `7.62` | `8.12` | `9.02` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.69` | `1.03` | `1.16` | `1.31` |
| `ffs_cam1_stage_ms` | `7.25` | `8.29` | `8.48` | `9.02` |
| `ffs_cam2_stage_ms` | `0.81` | `1.32` | `1.56` | `1.85` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `197.08` | `212.09` | `223.95` | `381.09` |
| `gpu_owner_ffs_cycle_ms` | `90.67` | `97.44` | `99.07` | `274.95` |
| `gpu_owner_edgetam_cycle_ms` | `106.33` | `119.51` | `123.94` | `135.87` |
| `raw_fusion_total_ms` | `8.67` | `10.32` | `11.35` | `11.91` |
| `fusion_total_ms` | `45.58` | `49.40` | `50.95` | `228.86` |
| `filter_total_ms` | `36.67` | `39.35` | `40.42` | `220.50` |
| `filter_input_age_ms` | `37.34` | `39.90` | `40.82` | `221.28` |
| `object_enhanced_pt_ms` | `23.39` | `25.89` | `26.93` | `207.55` |
| `controller_pt_filter_ms` | `13.05` | `14.32` | `14.69` | `16.00` |
| `render_total_ms` | `0.39` | `0.56` | `1.51` | `3.35` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.03` | `0.03` | `0.08` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.01` | `0.02` | `2.94` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `495` | `207.55` | `45840` | `7995` |
| `401` | `204.70` | `45759` | `8039` |
| `203` | `203.53` | `45794` | `7940` |
| `303` | `201.30` | `45817` | `8009` |
| `38` | `36.27` | `45249` | `7739` |
| `431` | `27.41` | `45802` | `8028` |
| `434` | `27.20` | `45838` | `8018` |
| `330` | `27.17` | `45823` | `7971` |
| `371` | `26.63` | `45797` | `8004` |
| `506` | `26.29` | `45775` | `8023` |
