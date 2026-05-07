# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.61`
- fusion FPS after warmup: `0.61`
- groups after warmup: `194`
- complete fused groups after warmup: `12`
- rendered groups after warmup: `12`
- complete group ratio after warmup: `0.062`
- target deficit: `4.39`
- bottleneck class: `upstream_supply`
- GPU pipeline: `separate-workers`
- single-owner order: `ffs-then-edgetam`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.94` | `25.32` | `27.19` | `33.39` |
| `ffs_cycle_ms` | `198.77` | `539.72` | `555.12` | `698.92` |
| `ffs_gate_wait_ms` | `100.30` | `437.57` | `448.70` | `569.35` |
| `edgetam_cam0_model_ms` | `151.40` | `177.14` | `181.47` | `209.20` |
| `edgetam_cam1_model_ms` | `141.67` | `179.78` | `207.98` | `616.81` |
| `edgetam_cam2_model_ms` | `150.37` | `183.42` | `204.10` | `508.25` |
| `edgetam_cam0_gate_wait_ms` | `93.46` | `151.92` | `160.21` | `190.67` |
| `edgetam_cam1_gate_wait_ms` | `15.78` | `115.92` | `132.84` | `174.47` |
| `edgetam_cam2_gate_wait_ms` | `25.62` | `138.97` | `149.45` | `191.74` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `1.08` | `1.88` | `2.20` | `3.92` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.19` | `1.89` | `2.40` | `3.31` |
| `ffs_cam1_stage_ms` | `0.97` | `1.79` | `2.13` | `3.68` |
| `ffs_cam2_stage_ms` | `1.01` | `1.79` | `2.14` | `3.92` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `61.64` | `69.53` | `70.86` | `72.23` |
| `object_enhanced_pt_ms` | `30.40` | `34.01` | `34.10` | `34.12` |
| `controller_pt_filter_ms` | `15.09` | `17.73` | `19.07` | `20.42` |
| `render_total_ms` | `2.08` | `5.08` | `9.24` | `14.17` |
| `open3d_object_update_geometry_ms` | `0.03` | `4.02` | `4.53` | `4.61` |
| `open3d_controller_update_geometry_ms` | `0.59` | `3.08` | `5.25` | `7.85` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `154` | `34.12` | `47288` | `11586` |
| `219` | `34.07` | `47314` | `11529` |
| `179` | `33.49` | `47299` | `11567` |
| `221` | `31.79` | `47205` | `11578` |
| `205` | `31.71` | `47216` | `11537` |
| `175` | `31.38` | `47284` | `11615` |
| `202` | `29.42` | `47278` | `11567` |
| `201` | `29.12` | `47301` | `11575` |
| `227` | `28.50` | `47237` | `11545` |
| `155` | `28.43` | `47389` | `11472` |
