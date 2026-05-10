# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `5.00`
- render FPS after warmup: `2.64`
- raw fusion FPS after warmup: `2.64`
- filter output FPS after warmup: `2.64`
- fusion FPS after warmup: `2.64`
- groups after warmup: `221`
- complete fused groups after warmup: `213`
- rendered groups after warmup: `213`
- complete group ratio after warmup: `0.964`
- target deficit: `2.36`
- Demo 2.2 PASS threshold: `4.80 FPS`
- Demo 2.2 result: `FAIL`
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
| `capture_temporal_skew_ms` | `43.00` | `61.67` | `63.99` | `66.45` |
| `ffs_cycle_ms` | `73.50` | `91.57` | `95.38` | `302.33` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `30.75` | `37.04` | `39.42` | `70.52` |
| `edgetam_cam1_model_ms` | `30.62` | `35.97` | `38.13` | `49.50` |
| `edgetam_cam2_model_ms` | `31.02` | `36.79` | `39.33` | `57.74` |
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
| `ffs_stage_ms` | `0.79` | `1.76` | `7.77` | `17.96` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.77` | `1.87` | `7.85` | `17.96` |
| `ffs_cam1_stage_ms` | `0.81` | `7.53` | `8.40` | `10.09` |
| `ffs_cam2_stage_ms` | `0.78` | `1.45` | `1.72` | `3.13` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `177.91` | `206.76` | `216.46` | `397.69` |
| `gpu_owner_ffs_cycle_ms` | `73.50` | `91.57` | `95.38` | `302.33` |
| `gpu_owner_edgetam_cycle_ms` | `102.20` | `116.45` | `123.63` | `150.69` |
| `raw_fusion_total_ms` | `7.75` | `9.32` | `9.93` | `13.11` |
| `fusion_total_ms` | `42.37` | `46.94` | `48.85` | `247.84` |
| `filter_total_ms` | `34.30` | `38.08` | `40.07` | `236.96` |
| `filter_input_age_ms` | `34.85` | `38.82` | `40.29` | `237.09` |
| `object_enhanced_pt_ms` | `21.62` | `24.30` | `25.63` | `223.31` |
| `controller_pt_filter_ms` | `12.55` | `14.23` | `14.53` | `18.72` |
| `render_total_ms` | `0.41` | `0.62` | `1.83` | `4.32` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.03` | `0.04` | `2.43` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.01` | `0.02` | `1.50` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `232` | `223.31` | `45262` | `7656` |
| `271` | `202.03` | `45538` | `7761` |
| `118` | `196.67` | `45521` | `7867` |
| `156` | `196.20` | `45458` | `7825` |
| `195` | `195.31` | `45489` | `7774` |
| `22` | `29.99` | `44624` | `7408` |
| `200` | `29.03` | `45452` | `7862` |
| `277` | `27.51` | `45569` | `7799` |
| `275` | `27.06` | `45651` | `7910` |
| `199` | `26.03` | `45369` | `7722` |
