# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `5.07`
- raw fusion FPS after warmup: `5.06`
- filter output FPS after warmup: `5.06`
- fusion FPS after warmup: `5.06`
- groups after warmup: `381`
- complete fused groups after warmup: `64`
- rendered groups after warmup: `64`
- complete group ratio after warmup: `0.168`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `9.93`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| camera startup ms | `4430.62` |
| EdgeTAM model load ms | `792.01` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1490.59` |
| EdgeTAM warmup/first forward ms | `93.02` |
| SAM3.1 model load ms | `7445.50` |
| SAM3.1 cam0 segment ms | `7721.65` |
| SAM3.1 cam1 segment ms | `172.75` |
| SAM3.1 cam2 segment ms | `170.20` |
| FFS runner init ms | `3168.81` |
| FFS first run ms | `1277.43` |
| session init + prompt add ms | `3.32` |
| SAM3.1 release cleanup ms | `213.36` |
| time to first complete group s | `25.18` |
| time to first rendered group s | `25.20` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `21.34` | `49.28` | `57.85` | `65.90` |
| `ffs_cycle_ms` | `88.59` | `96.06` | `101.71` | `1315.02` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `31.80` | `36.29` | `40.98` | `44.37` |
| `edgetam_cam1_model_ms` | `31.72` | `36.83` | `38.45` | `45.90` |
| `edgetam_cam2_model_ms` | `30.50` | `36.18` | `38.85` | `42.25` |
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
| `ffs_stage_ms` | `0.78` | `7.73` | `8.08` | `8.78` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.68` | `1.12` | `1.27` | `2.17` |
| `ffs_cam1_stage_ms` | `7.33` | `8.26` | `8.56` | `8.78` |
| `ffs_cam2_stage_ms` | `0.75` | `1.17` | `1.47` | `2.27` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `192.84` | `210.26` | `214.11` | `9821.29` |
| `gpu_owner_ffs_cycle_ms` | `88.59` | `96.06` | `101.71` | `1315.02` |
| `gpu_owner_edgetam_cycle_ms` | `103.69` | `116.60` | `121.49` | `8506.17` |
| `raw_fusion_total_ms` | `8.64` | `10.79` | `11.05` | `14.94` |
| `fusion_total_ms` | `44.38` | `46.76` | `47.82` | `212.70` |
| `filter_total_ms` | `35.63` | `37.77` | `38.54` | `204.02` |
| `filter_input_age_ms` | `36.09` | `38.43` | `38.76` | `204.31` |
| `object_enhanced_pt_ms` | `22.43` | `24.51` | `25.36` | `191.20` |
| `controller_pt_filter_ms` | `12.90` | `13.94` | `14.39` | `14.97` |
| `render_total_ms` | `0.39` | `2.09` | `3.03` | `5.66` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.06` | `1.31` | `2.77` |
| `open3d_controller_update_geometry_ms` | `0.01` | `1.27` | `1.60` | `2.76` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `313` | `191.20` | `45767` | `7966` |
| `88` | `32.66` | `45221` | `7726` |
| `337` | `25.41` | `45816` | `8010` |
| `261` | `25.37` | `45779` | `7972` |
| `332` | `25.31` | `45823` | `7998` |
| `277` | `25.08` | `45743` | `7960` |
| `395` | `24.55` | `45811` | `7971` |
| `307` | `24.42` | `45815` | `8045` |
| `320` | `24.34` | `45773` | `7944` |
| `326` | `24.16` | `45814` | `7942` |
