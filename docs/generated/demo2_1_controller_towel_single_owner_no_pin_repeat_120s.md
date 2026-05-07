# Demo 2.1 performance profile

- preset: `visual-5fps-single-owner`
- canonical preset: `perf-5fps-single-owner`
- target FPS: `5.00`
- render FPS after warmup: `4.44`
- fusion FPS after warmup: `4.44`
- groups after warmup: `349`
- complete fused groups after warmup: `334`
- rendered groups after warmup: `334`
- complete group ratio after warmup: `0.957`
- target deficit: `0.56`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- pin memory mode: `off`
- FFS input staging: `pageable`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `22.24` | `51.07` | `59.30` | `65.92` |
| `ffs_cycle_ms` | `85.10` | `94.57` | `97.06` | `271.56` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `32.24` | `43.48` | `48.52` | `75.56` |
| `edgetam_cam1_model_ms` | `33.09` | `42.92` | `47.45` | `74.51` |
| `edgetam_cam2_model_ms` | `33.33` | `43.26` | `46.96` | `65.71` |
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
| `gpu_owner_total_ms` | `194.31` | `229.86` | `243.55` | `398.92` |
| `gpu_owner_ffs_cycle_ms` | `85.10` | `94.57` | `97.06` | `271.56` |
| `gpu_owner_edgetam_cycle_ms` | `110.13` | `135.45` | `147.35` | `197.63` |
| `fusion_total_ms` | `46.72` | `55.82` | `58.66` | `237.92` |
| `object_enhanced_pt_ms` | `23.92` | `28.90` | `30.39` | `209.20` |
| `controller_pt_filter_ms` | `13.73` | `16.16` | `17.56` | `20.07` |
| `render_total_ms` | `0.39` | `0.86` | `1.85` | `5.10` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.09` | `4.66` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `3.53` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `454` | `209.20` | `45767` | `11900` |
| `217` | `206.69` | `45768` | `11848` |
| `378` | `206.09` | `45775` | `11881` |
| `336` | `203.76` | `45806` | `11839` |
| `414` | `198.22` | `45802` | `11876` |
| `295` | `198.06` | `45774` | `11757` |
| `257` | `194.95` | `45802` | `11834` |
| `177` | `194.06` | `45776` | `11809` |
| `456` | `37.95` | `45811` | `11869` |
| `465` | `35.18` | `45807` | `11889` |
