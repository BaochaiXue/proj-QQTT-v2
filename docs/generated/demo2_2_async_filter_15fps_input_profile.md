# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `5.00`
- render FPS after warmup: `4.52`
- raw fusion FPS after warmup: `4.52`
- filter output FPS after warmup: `4.52`
- fusion FPS after warmup: `4.52`
- groups after warmup: `354`
- complete fused groups after warmup: `351`
- rendered groups after warmup: `350`
- complete group ratio after warmup: `0.992`
- target deficit: `0.48`
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
| `capture_temporal_skew_ms` | `19.12` | `42.47` | `52.26` | `62.58` |
| `ffs_cycle_ms` | `87.09` | `95.91` | `98.91` | `278.01` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `31.70` | `37.63` | `39.40` | `59.10` |
| `edgetam_cam1_model_ms` | `30.63` | `36.03` | `37.40` | `49.32` |
| `edgetam_cam2_model_ms` | `31.41` | `37.99` | `39.91` | `50.64` |
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
| `ffs_stage_ms` | `0.92` | `7.39` | `8.09` | `12.92` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.82` | `1.54` | `4.87` | `12.92` |
| `ffs_cam1_stage_ms` | `1.11` | `8.27` | `8.62` | `9.96` |
| `ffs_cam2_stage_ms` | `0.94` | `1.70` | `2.05` | `3.66` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `189.80` | `211.28` | `218.25` | `387.43` |
| `gpu_owner_ffs_cycle_ms` | `87.09` | `95.91` | `98.91` | `278.01` |
| `gpu_owner_edgetam_cycle_ms` | `104.79` | `117.42` | `121.64` | `157.73` |
| `raw_fusion_total_ms` | `8.45` | `9.97` | `10.69` | `12.24` |
| `fusion_total_ms` | `45.39` | `48.80` | `49.94` | `234.20` |
| `filter_total_ms` | `36.59` | `39.49` | `40.60` | `226.83` |
| `filter_input_age_ms` | `37.10` | `40.07` | `41.17` | `227.61` |
| `object_enhanced_pt_ms` | `23.54` | `25.72` | `26.73` | `212.44` |
| `controller_pt_filter_ms` | `12.97` | `14.10` | `14.66` | `17.69` |
| `render_total_ms` | `0.39` | `0.66` | `1.47` | `4.29` |
| `open3d_object_update_geometry_ms` | `0.02` | `0.03` | `0.05` | `3.80` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.01` | `0.02` | `2.17` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `205` | `212.44` | `45761` | `7944` |
| `421` | `208.95` | `45745` | `7952` |
| `277` | `208.44` | `45758` | `7918` |
| `349` | `204.98` | `45753` | `7979` |
| `313` | `204.79` | `45761` | `8004` |
| `241` | `202.91` | `45792` | `7925` |
| `385` | `202.20` | `45772` | `7898` |
| `457` | `202.02` | `45760` | `7916` |
| `169` | `192.32` | `45759` | `7946` |
| `37` | `32.49` | `45232` | `7652` |
