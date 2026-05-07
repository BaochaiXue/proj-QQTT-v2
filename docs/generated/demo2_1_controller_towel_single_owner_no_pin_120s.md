# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps-single-owner`
- target FPS: `5.00`
- render FPS after warmup: `3.85`
- fusion FPS after warmup: `3.85`
- groups after warmup: `367`
- complete fused groups after warmup: `315`
- rendered groups after warmup: `314`
- complete group ratio after warmup: `0.858`
- target deficit: `1.15`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- pin memory mode: `off`
- FFS input staging: `pageable`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `16.90` | `23.13` | `29.78` | `32.84` |
| `ffs_cycle_ms` | `95.02` | `103.95` | `106.73` | `291.01` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `44.04` | `61.88` | `69.92` | `90.05` |
| `edgetam_cam1_model_ms` | `44.26` | `58.55` | `64.54` | `77.39` |
| `edgetam_cam2_model_ms` | `44.65` | `60.23` | `65.25` | `84.04` |
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
| `gpu_owner_total_ms` | `242.13` | `282.13` | `298.80` | `480.33` |
| `gpu_owner_ffs_cycle_ms` | `95.02` | `103.95` | `106.73` | `291.01` |
| `gpu_owner_edgetam_cycle_ms` | `147.78` | `181.30` | `191.67` | `239.62` |
| `fusion_total_ms` | `50.24` | `57.24` | `59.85` | `236.13` |
| `object_enhanced_pt_ms` | `24.93` | `28.86` | `31.20` | `210.23` |
| `controller_pt_filter_ms` | `14.86` | `17.16` | `18.34` | `21.25` |
| `render_total_ms` | `0.47` | `0.79` | `1.01` | `7.53` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.05` | `4.37` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.01` | `0.02` | `6.26` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `289` | `210.23` | `47310` | `11610` |
| `459` | `208.04` | `47329` | `11562` |
| `416` | `199.79` | `47313` | `11645` |
| `333` | `197.58` | `47301` | `11578` |
| `172` | `194.17` | `47378` | `11654` |
| `249` | `193.26` | `47315` | `11565` |
| `374` | `191.80` | `47386` | `11655` |
| `210` | `190.25` | `47338` | `11548` |
| `455` | `36.56` | `47278` | `11581` |
| `170` | `35.43` | `47298` | `11672` |
