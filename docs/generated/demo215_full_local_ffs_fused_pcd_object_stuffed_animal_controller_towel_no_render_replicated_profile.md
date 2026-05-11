# Demo 2.1 performance profile

- preset: `demo2.1.5-live-quality-ffs`
- canonical preset: `demo2.1.5-live-quality-ffs`
- target FPS: `25.00`
- capture group target FPS: `15.00`
- compile mode: `none`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `4.49`
- filter output FPS after warmup: `4.49`
- fusion FPS after warmup: `4.49`
- groups after warmup: `1119`
- complete fused groups after warmup: `372`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.332`
- target deficit: `25.00`
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
| parallel init max wait ms | `4499.91` |
| camera startup ms | `8052.01` |
| EdgeTAM model load ms | `2474.26` |
| EdgeTAM compile wrap ms | `0.00` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `573.03` |
| SAM3.1 model load ms | `6960.50` |
| SAM3.1 cam0 segment ms | `7381.34` |
| SAM3.1 cam1 segment ms | `173.45` |
| SAM3.1 cam2 segment ms | `160.92` |
| FFS runner init ms | `8140.46` |
| FFS first run ms | `1089.15` |
| session init + prompt add ms | `2.89` |
| SAM3.1 release cleanup ms | `241.38` |
| time to first complete group s | `23.71` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.200`
- samples after warmup: `407`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `43.00` | `90.40` | `96.00` | `98.00` |
| `memory_util_pct` | `11.00` | `27.80` | `41.00` | `50.00` |
| `memory_used_mb` | `10521.10` | `13146.20` | `13519.30` | `13942.20` |
| `power_w` | `120.02` | `143.64` | `163.92` | `243.31` |
| `sm_clock_mhz` | `180.00` | `1597.00` | `1597.00` | `1597.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `65.00` | `68.00` | `69.00` | `69.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `26.93` | `48.98` | `55.90` | `66.30` |
| `edgetam_model_ms` | `42.75` | `65.02` | `101.72` | `241.77` |
| `edgetam_preprocess_ms` | `1.34` | `1.81` | `1.98` | `5.80` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.08` | `0.18` | `0.24` | `0.65` |
| `edgetam_mask_resize_ms` | `0.05` | `0.11` | `0.17` | `0.57` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.06` | `0.08` | `0.21` |
| `edgetam_mask_to_cpu_ms` | `0.20` | `0.28` | `0.38` | `5.71` |
| `edgetam_total_ms` | `41.70` | `62.81` | `97.42` | `223.82` |
| `ffs_cycle_ms` | `67.98` | `83.21` | `110.16` | `302.48` |
| `ffs_batch_ms` | `45.47` | `56.36` | `84.85` | `275.30` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `42.95` | `71.21` | `106.61` | `241.77` |
| `edgetam_cam1_model_ms` | `42.65` | `63.83` | `108.53` | `185.32` |
| `edgetam_cam2_model_ms` | `42.67` | `64.12` | `83.93` | `220.01` |
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
| `ffs_stage_ms` | `2.05` | `3.14` | `3.79` | `7.25` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.05` | `3.14` | `3.76` | `7.25` |
| `ffs_cam1_stage_ms` | `2.05` | `3.14` | `3.76` | `7.25` |
| `ffs_cam2_stage_ms` | `2.05` | `3.14` | `3.76` | `7.25` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `193.89` | `304.94` | `395.10` | `694.22` |
| `gpu_owner_ffs_cycle_ms` | `67.98` | `83.21` | `110.16` | `302.48` |
| `gpu_owner_edgetam_cycle_ms` | `124.21` | `183.16` | `278.99` | `572.51` |
| `raw_fusion_total_ms` | `11.81` | `14.40` | `15.56` | `22.19` |
| `fusion_total_ms` | `51.99` | `63.87` | `83.12` | `287.24` |
| `filter_total_ms` | `40.31` | `50.71` | `63.00` | `270.01` |
| `filter_input_age_ms` | `40.78` | `51.16` | `63.25` | `270.92` |
| `object_enhanced_pt_ms` | `27.22` | `34.51` | `43.91` | `254.99` |
| `controller_pt_filter_ms` | `12.98` | `16.00` | `17.54` | `40.46` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1115` | `254.99` | `73348` | `7907` |
| `972` | `244.59` | `73351` | `7981` |
| `663` | `226.55` | `73333` | `7875` |
| `1429` | `224.07` | `73385` | `7911` |
| `770` | `223.15` | `73369` | `7958` |
| `564` | `222.65` | `73295` | `7978` |
| `1517` | `220.70` | `73372` | `7912` |
| `293` | `215.59` | `73342` | `7908` |
| `1250` | `213.42` | `73384` | `7921` |
| `386` | `212.89` | `73360` | `7964` |
