# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps-single-owner`
- target FPS: `5.00`
- render FPS after warmup: `3.74`
- fusion FPS after warmup: `3.74`
- groups after warmup: `360`
- complete fused groups after warmup: `313`
- rendered groups after warmup: `312`
- complete group ratio after warmup: `0.869`
- target deficit: `1.26`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `edgetam-then-ffs`
- pin memory mode: `off`
- FFS input staging: `pageable`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.90` | `26.64` | `28.61` | `33.28` |
| `ffs_cycle_ms` | `67.84` | `72.86` | `74.19` | `83.61` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `79.30` | `93.19` | `102.35` | `271.40` |
| `edgetam_cam1_model_ms` | `42.71` | `58.43` | `65.42` | `82.33` |
| `edgetam_cam2_model_ms` | `42.58` | `56.67` | `62.90` | `77.46` |
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
| `gpu_owner_total_ms` | `245.40` | `286.12` | `304.84` | `449.22` |
| `gpu_owner_ffs_cycle_ms` | `67.84` | `72.86` | `74.19` | `83.61` |
| `gpu_owner_edgetam_cycle_ms` | `176.20` | `218.06` | `232.68` | `382.83` |
| `fusion_total_ms` | `51.81` | `60.82` | `65.94` | `259.48` |
| `object_enhanced_pt_ms` | `25.80` | `31.08` | `33.54` | `236.97` |
| `controller_pt_filter_ms` | `14.54` | `17.21` | `18.49` | `24.66` |
| `render_total_ms` | `0.54` | `0.87` | `1.23` | `4.58` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.06` | `3.69` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.03` | `4.01` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `485` | `236.97` | `47323` | `11613` |
| `239` | `215.06` | `47317` | `11575` |
| `319` | `214.24` | `47371` | `11624` |
| `360` | `210.99` | `47296` | `11623` |
| `400` | `205.45` | `47298` | `11583` |
| `200` | `204.49` | `47360` | `11631` |
| `442` | `200.91` | `47417` | `11624` |
| `162` | `191.75` | `47396` | `11534` |
| `280` | `191.08` | `47278` | `11527` |
| `238` | `38.58` | `47357` | `11577` |
