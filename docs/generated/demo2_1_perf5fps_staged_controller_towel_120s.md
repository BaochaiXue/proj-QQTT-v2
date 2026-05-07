# Demo 2.1 performance profile

- preset: `perf-5fps-staged`
- canonical preset: `perf-5fps-staged`
- target FPS: `5.00`
- render FPS after warmup: `3.87`
- fusion FPS after warmup: `3.87`
- groups after warmup: `516`
- complete fused groups after warmup: `305`
- rendered groups after warmup: `305`
- complete group ratio after warmup: `0.591`
- target deficit: `1.13`
- bottleneck class: `upstream_supply`
- GPU pipeline: `staged`
- single-owner order: `ffs-then-parallel-edgetam`
- pin memory mode: `off`
- FFS input staging: `pageable`
- H2D stream mode: `default`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `14.76` | `43.85` | `52.90` | `66.54` |
| `ffs_cycle_ms` | `91.24` | `101.30` | `104.22` | `905.06` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `135.94` | `156.25` | `164.77` | `182.92` |
| `edgetam_cam1_model_ms` | `142.29` | `157.94` | `165.31` | `191.86` |
| `edgetam_cam2_model_ms` | `147.26` | `164.03` | `168.36` | `184.71` |
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
| `gpu_owner_total_ms` | `248.00` | `272.24` | `283.90` | `25428.48` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `50.26` | `58.99` | `64.87` | `263.52` |
| `object_enhanced_pt_ms` | `25.59` | `31.19` | `35.90` | `234.56` |
| `controller_pt_filter_ms` | `14.56` | `17.43` | `18.28` | `21.64` |
| `render_total_ms` | `0.42` | `0.73` | `1.58` | `6.84` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.07` | `5.35` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `3.21` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `231` | `234.56` | `45842` | `11846` |
| `362` | `233.73` | `45860` | `11838` |
| `274` | `211.13` | `45826` | `11879` |
| `407` | `206.72` | `45837` | `11844` |
| `502` | `205.54` | `45865` | `11866` |
| `188` | `205.46` | `45826` | `11890` |
| `453` | `201.92` | `45848` | `11875` |
| `319` | `196.19` | `45838` | `11831` |
| `364` | `38.14` | `45859` | `11887` |
| `466` | `37.63` | `45837` | `11874` |
