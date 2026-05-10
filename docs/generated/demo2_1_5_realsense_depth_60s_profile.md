# Demo 2.1 performance profile

- preset: `demo2.1.5-async-filter-5fps`
- canonical preset: `demo2.1.5-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `6.38`
- raw fusion FPS after warmup: `6.38`
- filter output FPS after warmup: `6.38`
- fusion FPS after warmup: `6.38`
- groups after warmup: `381`
- complete fused groups after warmup: `171`
- rendered groups after warmup: `170`
- complete group ratio after warmup: `0.449`
- target deficit: `8.62`
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
| parallel init max wait ms | `27565.26` |
| camera startup ms | `5835.83` |
| EdgeTAM model load ms | `3191.30` |
| EdgeTAM compile wrap ms | `616.15` |
| EdgeTAM compile prewarm ms | `16418.37` |
| EdgeTAM warmup/first forward ms | `1008.37` |
| SAM3.1 model load ms | `9834.70` |
| SAM3.1 cam0 segment ms | `742.04` |
| SAM3.1 cam1 segment ms | `204.05` |
| SAM3.1 cam2 segment ms | `202.64` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `6.29` |
| SAM3.1 release cleanup ms | `403.99` |
| time to first complete group s | `38.62` |
| time to first rendered group s | `38.64` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `15.22` | `54.67` | `58.93` | `66.37` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `67.05` | `87.83` | `104.24` | `935.21` |
| `edgetam_cam1_model_ms` | `32.01` | `45.45` | `51.35` | `60.58` |
| `edgetam_cam2_model_ms` | `31.38` | `42.70` | `48.99` | `65.56` |
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
| `gpu_owner_total_ms` | `141.80` | `192.59` | `219.31` | `2572.52` |
| `gpu_owner_ffs_cycle_ms` | `0.59` | `0.93` | `0.99` | `3.62` |
| `gpu_owner_edgetam_cycle_ms` | `141.24` | `191.90` | `218.73` | `2568.84` |
| `raw_fusion_total_ms` | `12.09` | `16.93` | `18.21` | `24.19` |
| `fusion_total_ms` | `53.33` | `71.90` | `81.05` | `392.46` |
| `filter_total_ms` | `40.92` | `53.83` | `63.08` | `374.18` |
| `filter_input_age_ms` | `41.38` | `53.93` | `64.40` | `374.44` |
| `object_enhanced_pt_ms` | `26.59` | `36.02` | `40.96` | `360.61` |
| `controller_pt_filter_ms` | `14.33` | `17.74` | `19.82` | `31.19` |
| `render_total_ms` | `0.54` | `1.16` | `2.87` | `13.97` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.06` | `1.33` | `8.45` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.03` | `5.30` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `461` | `360.61` | `46357` | `8427` |
| `558` | `337.68` | `46293` | `8400` |
| `654` | `335.64` | `46360` | `8434` |
| `413` | `47.02` | `46300` | `8454` |
| `392` | `46.53` | `46362` | `8467` |
| `408` | `43.80` | `46296` | `8456` |
| `415` | `43.24` | `46327` | `8487` |
| `405` | `42.66` | `46296` | `8457` |
| `398` | `42.27` | `46352` | `8461` |
| `395` | `39.66` | `46348` | `8438` |
