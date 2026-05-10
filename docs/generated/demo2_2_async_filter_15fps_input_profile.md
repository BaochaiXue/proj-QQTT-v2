# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `5.13`
- raw fusion FPS after warmup: `5.12`
- filter output FPS after warmup: `5.12`
- fusion FPS after warmup: `5.12`
- groups after warmup: `381`
- complete fused groups after warmup: `78`
- rendered groups after warmup: `78`
- complete group ratio after warmup: `0.205`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `9.87`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.43` | `48.65` | `58.16` | `105.55` |
| `ffs_cycle_ms` | `89.26` | `95.71` | `101.34` | `926.76` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `30.78` | `38.05` | `38.77` | `96.11` |
| `edgetam_cam1_model_ms` | `29.20` | `33.45` | `34.86` | `38.53` |
| `edgetam_cam2_model_ms` | `29.43` | `32.86` | `34.52` | `49.77` |
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
| `ffs_stage_ms` | `1.12` | `7.74` | `8.19` | `9.78` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.94` | `1.52` | `1.76` | `2.09` |
| `ffs_cam1_stage_ms` | `7.30` | `8.47` | `8.95` | `9.78` |
| `ffs_cam2_stage_ms` | `1.05` | `1.86` | `2.07` | `2.54` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `188.88` | `208.28` | `218.07` | `9045.82` |
| `gpu_owner_ffs_cycle_ms` | `89.26` | `95.71` | `101.34` | `926.76` |
| `gpu_owner_edgetam_cycle_ms` | `99.71` | `108.49` | `116.22` | `8118.99` |
| `raw_fusion_total_ms` | `8.80` | `10.34` | `10.64` | `15.70` |
| `fusion_total_ms` | `45.28` | `47.74` | `50.81` | `217.13` |
| `filter_total_ms` | `36.30` | `38.77` | `40.45` | `207.07` |
| `filter_input_age_ms` | `36.77` | `39.54` | `41.38` | `207.24` |
| `object_enhanced_pt_ms` | `22.95` | `25.63` | `26.06` | `193.60` |
| `controller_pt_filter_ms` | `12.94` | `14.22` | `14.66` | `16.03` |
| `render_total_ms` | `0.37` | `2.04` | `2.84` | `7.77` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.06` | `1.53` | `3.17` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.17` | `3.51` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `353` | `193.60` | `45622` | `7905` |
| `257` | `186.30` | `45643` | `7907` |
| `49` | `33.08` | `45291` | `7721` |
| `215` | `26.22` | `45640` | `7954` |
| `350` | `26.04` | `45618` | `7930` |
| `316` | `25.89` | `45628` | `7905` |
| `376` | `25.70` | `45652` | `7870` |
| `264` | `25.64` | `45643` | `7962` |
| `367` | `25.63` | `45655` | `7934` |
| `283` | `25.48` | `45621` | `7863` |
