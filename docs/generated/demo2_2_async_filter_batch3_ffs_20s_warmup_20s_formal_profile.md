# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `5.37`
- raw fusion FPS after warmup: `5.38`
- filter output FPS after warmup: `5.37`
- fusion FPS after warmup: `5.37`
- groups after warmup: `340`
- complete fused groups after warmup: `137`
- rendered groups after warmup: `137`
- complete group ratio after warmup: `0.403`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `9.63`
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
| parallel init max wait ms | `3981.24` |
| camera startup ms | `7933.45` |
| EdgeTAM model load ms | `991.65` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1495.29` |
| EdgeTAM warmup/first forward ms | `145.32` |
| SAM3.1 model load ms | `9821.58` |
| SAM3.1 cam0 segment ms | `899.32` |
| SAM3.1 cam1 segment ms | `185.90` |
| SAM3.1 cam2 segment ms | `192.49` |
| FFS runner init ms | `8886.81` |
| FFS first run ms | `990.83` |
| session init + prompt add ms | `4.72` |
| SAM3.1 release cleanup ms | `234.43` |
| time to first complete group s | `18.51` |
| time to first rendered group s | `18.53` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `25.92` | `57.02` | `58.90` | `65.55` |
| `ffs_cycle_ms` | `67.38` | `73.71` | `75.46` | `254.87` |
| `ffs_batch_ms` | `49.02` | `53.33` | `55.49` | `231.72` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `33.64` | `40.60` | `43.96` | `53.25` |
| `edgetam_cam1_model_ms` | `32.60` | `39.94` | `41.32` | `50.74` |
| `edgetam_cam2_model_ms` | `31.97` | `38.66` | `44.06` | `54.43` |
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
| `ffs_stage_ms` | `2.17` | `3.67` | `4.06` | `6.70` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.17` | `3.66` | `4.00` | `6.70` |
| `ffs_cam1_stage_ms` | `2.17` | `3.66` | `4.00` | `6.70` |
| `ffs_cam2_stage_ms` | `2.17` | `3.66` | `4.00` | `6.70` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `177.28` | `200.34` | `209.75` | `359.78` |
| `gpu_owner_ffs_cycle_ms` | `67.38` | `73.71` | `75.46` | `254.87` |
| `gpu_owner_edgetam_cycle_ms` | `109.17` | `125.70` | `132.76` | `152.28` |
| `raw_fusion_total_ms` | `10.40` | `12.33` | `13.10` | `14.54` |
| `fusion_total_ms` | `47.88` | `52.96` | `55.68` | `243.98` |
| `filter_total_ms` | `37.40` | `42.04` | `44.00` | `233.65` |
| `filter_input_age_ms` | `38.04` | `42.55` | `44.45` | `233.73` |
| `object_enhanced_pt_ms` | `24.55` | `27.94` | `30.73` | `219.82` |
| `controller_pt_filter_ms` | `12.64` | `14.07` | `14.64` | `15.45` |
| `render_total_ms` | `0.47` | `0.86` | `1.20` | `6.57` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.06` | `3.02` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `6.13` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `386` | `219.82` | `45645` | `7956` |
| `471` | `204.05` | `45646` | `7983` |
| `296` | `202.81` | `45637` | `7938` |
| `212` | `201.97` | `45585` | `7922` |
| `55` | `32.26` | `45225` | `7760` |
| `453` | `31.16` | `45649` | `8031` |
| `477` | `31.02` | `45654` | `7943` |
| `487` | `30.91` | `45622` | `7933` |
| `328` | `30.68` | `45625` | `7901` |
| `342` | `30.06` | `45618` | `7904` |
