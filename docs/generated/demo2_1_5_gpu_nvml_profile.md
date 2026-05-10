# Demo 2.1 performance profile

- preset: `demo2.1.5-async-filter-5fps`
- canonical preset: `demo2.1.5-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `7.29`
- raw fusion FPS after warmup: `7.34`
- filter output FPS after warmup: `7.29`
- fusion FPS after warmup: `7.29`
- groups after warmup: `387`
- complete fused groups after warmup: `212`
- rendered groups after warmup: `212`
- complete group ratio after warmup: `0.548`
- target deficit: `7.71`
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
| parallel init max wait ms | `6249.65` |
| camera startup ms | `4710.83` |
| EdgeTAM model load ms | `814.77` |
| EdgeTAM compile wrap ms | `1249.05` |
| EdgeTAM compile prewarm ms | `5801.85` |
| EdgeTAM warmup/first forward ms | `964.94` |
| SAM3.1 model load ms | `13597.20` |
| SAM3.1 cam0 segment ms | `517.99` |
| SAM3.1 cam1 segment ms | `185.97` |
| SAM3.1 cam2 segment ms | `168.61` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `3.89` |
| SAM3.1 release cleanup ms | `253.24` |
| time to first complete group s | `18.76` |
| time to first rendered group s | `18.78` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.100`
- samples after warmup: `284`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `30.00` | `35.70` | `37.00` | `39.00` |
| `memory_util_pct` | `4.00` | `6.00` | `7.00` | `8.00` |
| `memory_used_mb` | `6193.48` | `8554.88` | `8668.48` | `8668.48` |
| `power_w` | `105.24` | `109.75` | `111.71` | `133.56` |
| `sm_clock_mhz` | `232.00` | `232.00` | `232.00` | `232.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `59.00` | `61.00` | `61.00` | `63.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `23.86` | `52.64` | `52.97` | `65.56` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `62.30` | `69.43` | `72.49` | `280.84` |
| `edgetam_cam1_model_ms` | `27.54` | `34.51` | `37.74` | `46.88` |
| `edgetam_cam2_model_ms` | `26.37` | `34.36` | `38.80` | `46.06` |
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
| `gpu_owner_total_ms` | `129.16` | `148.82` | `154.81` | `343.09` |
| `gpu_owner_ffs_cycle_ms` | `0.58` | `0.74` | `0.81` | `1.18` |
| `gpu_owner_edgetam_cycle_ms` | `128.52` | `148.16` | `154.12` | `342.54` |
| `raw_fusion_total_ms` | `11.67` | `14.04` | `15.10` | `20.17` |
| `fusion_total_ms` | `53.38` | `58.87` | `61.16` | `269.90` |
| `filter_total_ms` | `40.89` | `46.86` | `49.52` | `258.61` |
| `filter_input_age_ms` | `41.57` | `47.51` | `49.77` | `258.83` |
| `object_enhanced_pt_ms` | `26.44` | `31.42` | `32.86` | `244.31` |
| `controller_pt_filter_ms` | `14.24` | `16.55` | `17.43` | `21.34` |
| `render_total_ms` | `0.29` | `0.46` | `0.52` | `3.87` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.04` | `2.27` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `3.51` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `486` | `244.31` | `46348` | `8414` |
| `684` | `243.58` | `46323` | `8407` |
| `282` | `237.86` | `46336` | `8427` |
| `349` | `237.24` | `46329` | `8429` |
| `417` | `234.23` | `46313` | `8490` |
| `212` | `231.98` | `46337` | `8447` |
| `624` | `230.63` | `46372` | `8515` |
| `555` | `229.67` | `46348` | `8437` |
| `441` | `35.46` | `46338` | `8419` |
| `74` | `34.62` | `46043` | `8300` |
