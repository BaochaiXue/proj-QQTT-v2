# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.00`
- fusion FPS after warmup: `0.00`
- groups after warmup: `101`
- complete fused groups after warmup: `0`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.000`
- target deficit: `5.00`
- bottleneck class: `upstream_supply`
- GPU pipeline: `separate-workers`
- single-owner order: `ffs-then-edgetam`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

Warning: this profile has no complete fused groups after warmup. Treat it as an initialization or missing-packet run, not as a valid visual FPS comparison.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `16.06` | `30.07` | `31.13` | `33.10` |
| `ffs_cycle_ms` | `114.60` | `152.73` | `214.58` | `2347.79` |
| `ffs_gate_wait_ms` | `0.01` | `0.01` | `0.02` | `114.29` |
| `edgetam_cam0_model_ms` | `94.21` | `144.05` | `150.27` | `156.50` |
| `edgetam_cam1_model_ms` | `75.60` | `131.93` | `153.72` | `442.22` |
| `edgetam_cam2_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_gate_wait_ms` | `0.01` | `0.01` | `0.01` | `0.01` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.01` | `11.29` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `1.11` | `2.00` | `2.67` | `16.65` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.22` | `2.35` | `2.91` | `9.02` |
| `ffs_cam1_stage_ms` | `1.02` | `1.92` | `2.45` | `3.88` |
| `ffs_cam2_stage_ms` | `1.06` | `1.86` | `2.15` | `16.65` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `0` | `0.00` | `0` | `0` |
| `1` | `0.00` | `0` | `0` |
| `2` | `0.00` | `0` | `0` |
| `3` | `0.00` | `0` | `0` |
| `4` | `0.00` | `0` | `0` |
| `5` | `0.00` | `0` | `0` |
| `6` | `0.00` | `0` | `0` |
| `7` | `0.00` | `0` | `0` |
| `8` | `0.00` | `0` | `0` |
| `9` | `0.00` | `0` | `0` |
