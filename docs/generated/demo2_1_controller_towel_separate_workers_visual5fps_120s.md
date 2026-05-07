# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.51`
- fusion FPS after warmup: `0.51`
- groups after warmup: `367`
- complete fused groups after warmup: `38`
- rendered groups after warmup: `38`
- complete group ratio after warmup: `0.104`
- target deficit: `4.49`
- bottleneck class: `upstream_supply`
- GPU pipeline: `separate-workers`
- single-owner order: `ffs-then-edgetam`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.40` | `26.24` | `28.04` | `32.92` |
| `ffs_cycle_ms` | `419.08` | `525.25` | `561.82` | `1116.33` |
| `ffs_gate_wait_ms` | `305.00` | `410.43` | `439.27` | `978.84` |
| `edgetam_cam0_model_ms` | `140.55` | `169.41` | `178.17` | `527.84` |
| `edgetam_cam1_model_ms` | `136.18` | `162.37` | `170.49` | `195.82` |
| `edgetam_cam2_model_ms` | `138.71` | `166.82` | `178.12` | `629.28` |
| `edgetam_cam0_gate_wait_ms` | `64.22` | `125.86` | `137.72` | `180.51` |
| `edgetam_cam1_gate_wait_ms` | `64.44` | `132.76` | `143.02` | `180.51` |
| `edgetam_cam2_gate_wait_ms` | `62.90` | `130.04` | `143.37` | `176.27` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `0.90` | `1.64` | `1.89` | `3.33` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.94` | `1.72` | `1.89` | `3.02` |
| `ffs_cam1_stage_ms` | `0.87` | `1.51` | `1.69` | `2.18` |
| `ffs_cam2_stage_ms` | `0.89` | `1.69` | `2.00` | `3.33` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `53.07` | `65.09` | `69.16` | `230.34` |
| `object_enhanced_pt_ms` | `26.74` | `31.50` | `32.85` | `205.15` |
| `controller_pt_filter_ms` | `14.57` | `18.11` | `19.09` | `20.84` |
| `render_total_ms` | `0.59` | `2.38` | `3.02` | `7.26` |
| `open3d_object_update_geometry_ms` | `0.03` | `1.55` | `1.64` | `2.56` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.03` | `1.39` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `476` | `205.15` | `47264` | `11603` |
| `346` | `36.43` | `47293` | `11608` |
| `452` | `32.22` | `47319` | `11615` |
| `488` | `31.53` | `47312` | `11547` |
| `329` | `31.48` | `47360` | `11649` |
| `495` | `28.95` | `47323` | `11522` |
| `424` | `28.55` | `47222` | `11629` |
| `237` | `28.37` | `47331` | `11683` |
| `218` | `28.33` | `47315` | `11619` |
| `327` | `28.33` | `47373` | `11663` |
