# Demo 2.2 performance profile

- preset: `demo2.2-staged-parallel-5fps`
- canonical preset: `demo2.2-staged-parallel-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `3.46`
- raw fusion FPS after warmup: `3.46`
- filter output FPS after warmup: `3.46`
- fusion FPS after warmup: `3.46`
- groups after warmup: `362`
- complete fused groups after warmup: `93`
- rendered groups after warmup: `93`
- complete group ratio after warmup: `0.257`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `11.54`
- bottleneck class: `upstream_supply`
- GPU pipeline: `staged`
- single-owner order: `ffs-then-parallel-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `all`
- FFS input staging: `pinned`
- H2D stream mode: `dedicated`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `4726.96` |
| camera startup ms | `5103.40` |
| EdgeTAM model load ms | `2604.13` |
| EdgeTAM compile wrap ms | `0.02` |
| EdgeTAM compile prewarm ms | `1565.37` |
| EdgeTAM warmup/first forward ms | `361.17` |
| SAM3.1 model load ms | `24034.38` |
| SAM3.1 cam0 segment ms | `7758.70` |
| SAM3.1 cam1 segment ms | `7879.63` |
| SAM3.1 cam2 segment ms | `9094.24` |
| FFS runner init ms | `4607.17` |
| FFS first run ms | `1396.77` |
| session init + prompt add ms | `8.47` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `37.63` |
| time to first rendered group s | `37.64` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `19.41` | `55.90` | `58.33` | `66.32` |
| `ffs_cycle_ms` | `98.41` | `109.57` | `114.90` | `298.12` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `158.50` | `183.55` | `189.71` | `226.81` |
| `edgetam_cam1_model_ms` | `160.64` | `181.01` | `188.82` | `220.34` |
| `edgetam_cam2_model_ms` | `161.69` | `181.95` | `187.07` | `221.42` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `1.14` | `1.69` | `2.03` | `5.62` |
| `edge_h2d_wait_ms` | `0.01` | `0.11` | `0.22` | `1.04` |
| `edge_cam0_pin_copy_ms` | `1.15` | `1.76` | `2.13` | `3.57` |
| `edge_cam1_pin_copy_ms` | `1.13` | `1.65` | `2.06` | `2.38` |
| `edge_cam2_pin_copy_ms` | `1.13` | `1.57` | `1.91` | `5.62` |
| `edge_cam0_h2d_wait_ms` | `0.01` | `0.14` | `0.28` | `1.04` |
| `edge_cam1_h2d_wait_ms` | `0.01` | `0.07` | `0.19` | `0.56` |
| `edge_cam2_h2d_wait_ms` | `0.01` | `0.12` | `0.20` | `0.52` |
| `ffs_stage_ms` | `0.97` | `8.11` | `9.00` | `11.32` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.85` | `1.41` | `1.79` | `6.08` |
| `ffs_cam1_stage_ms` | `4.37` | `9.20` | `9.79` | `11.32` |
| `ffs_cam2_stage_ms` | `0.92` | `1.63` | `2.18` | `2.35` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `283.66` | `313.51` | `321.42` | `472.28` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_total_ms` | `9.90` | `13.12` | `14.22` | `17.23` |
| `fusion_total_ms` | `51.74` | `61.13` | `64.40` | `248.59` |
| `filter_total_ms` | `42.43` | `50.49` | `52.17` | `238.24` |
| `filter_input_age_ms` | `42.66` | `50.69` | `52.95` | `238.49` |
| `object_enhanced_pt_ms` | `27.45` | `34.54` | `36.30` | `223.60` |
| `controller_pt_filter_ms` | `14.61` | `16.90` | `18.37` | `37.27` |
| `render_total_ms` | `0.43` | `0.83` | `1.08` | `2.33` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.03` | `0.06` | `1.76` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `1.45` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `582` | `235.38` | `45821` | `7961` |
| `721` | `223.60` | `45772` | `7917` |
| `856` | `213.45` | `45800` | `8008` |
| `556` | `40.19` | `45763` | `7985` |
| `653` | `37.63` | `45757` | `7950` |
| `756` | `36.41` | `45753` | `7845` |
| `739` | `36.30` | `45749` | `7872` |
| `699` | `36.29` | `45785` | `7934` |
| `674` | `36.28` | `45763` | `7954` |
| `730` | `36.01` | `45772` | `7951` |
